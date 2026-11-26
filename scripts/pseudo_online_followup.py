import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reach_or_cue.config import (FEATURE_SETS, MAIN_WINDOW, MAX_CENSORED, SEARCH, SIM_CUE_LAG,
                                 SIM_MOV_LAG, WINDOWS)
from reach_or_cue.online import (detect_events, first_triggers, prep_windows, score_events,
                                 sign_flip_p, tobit)

# Operating curves.
CURVE_QUANTILES = 1 - np.logspace(np.log10(2e-5), np.log10(0.6), 90)
FP_GRID         = np.logspace(-1, np.log10(30), 60) # per min
FP_POINTS       = (1.0, 2.0, 5.0)                   # per min

# Slope test at looser operating points.
FP_LEVELS = (1.0, 2.0, 5.0, 10.0) # per min
N_BOOT    = 100
SEED      = 1

# First-trigger latency by RT.
RT_BINS = [0.1, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 10.0] # s


def main():
    parser = argparse.ArgumentParser(description='Operating curves, slope test at looser operating '
                                                 'points and first-trigger latency by RT.')
    parser.add_argument('online_dir', help='pseudo_online.py output folder')
    parser.add_argument('--participants', default='1,2,3,4,5,6,7,8,9,10,11,12')
    parser.add_argument('--out', default='pseudo_online_followup')
    parser.add_argument('--stage', choices=('compute', 'summarize'), default='compute')
    args = parser.parse_args()

    online_dir = Path(args.online_dir)
    out        = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pids = [int(p) for p in args.participants.split(',')]
    if args.stage == 'compute':
        for pid in pids:
            compute_participant(pid, online_dir, out)
    else:
        summarize(pids, online_dir, out)


def compute_participant(pid, online_dir, out_dir):
    if (out_dir / f'slope_levels_P{pid}.csv').exists():
        print(f'P{pid}: cached')
        return
    rng    = np.random.default_rng(SEED + pid)
    scores = np.load(online_dir / f'scores_P{pid}.npz')
    events = pd.read_csv(online_dir / f'lifts_P{pid}.csv')
    runs   = sorted(int(k.split('|')[1]) for k in scores.files if k.startswith('t|'))
    t      = {r: scores[f't|{r}'] for r in runs}
    ev     = {r: events[events['run'] == r] for r in runs}
    preps  = {r: {name: prep_windows(ev[r], t[r], *w) for name, w in WINDOWS.items()} for r in runs}

    curve_rows, level_rows, trial_tabs = [], [], []
    for fset in FEATURE_SETS:
        s    = {r: scores[f'{fset}|{r}'] for r in runs}
        thrs = np.unique(np.quantile(np.concatenate(list(s.values())), CURVE_QUANTILES))[::-1]

        # Operating curve, one threshold shared across series.
        rows = []
        for th in thrs:
            trig = {r: detect_events(t[r], s[r], th) for r in runs}
            for name in WINDOWS:
                pooled = _pool([score_events(trig[r], preps[r][name]) for r in runs])
                rows.append({'pid': pid, 'features': fset, 'window': name, 'threshold': th,
                             **pooled})
        curve_rows += rows
        main_curve = pd.DataFrame([r for r in rows if r['window'] == MAIN_WINDOW])

        # Slope test at looser operating points.
        for level in FP_LEVELS:
            ok    = main_curve[main_curve['fp_per_min'] <= level]
            point = ok.loc[ok['detect_pct'].idxmax()] if len(ok) else main_curve.iloc[0]
            th    = point['threshold']
            tr    = []
            for r in runs:
                tr += first_triggers(pid, r, fset, detect_events(t[r], s[r], th), ev[r])
            g = pd.DataFrame(tr)
            g['level'] = level
            trial_tabs.append(g)
            pre_pct = 100 * g['pre_cue'].mean()
            g       = g[~g['pre_cue']]
            x, y, c = g['rt'].to_numpy(), g['y'].to_numpy(), g['censored'].to_numpy()
            cens    = 100 * c.mean()
            usable  = cens <= MAX_CENSORED
            beta    = tobit(x, y, c)[1] if usable else np.nan
            boot    = []
            for _ in range(N_BOOT if usable else 0):
                idx = rng.integers(0, len(g), len(g))
                boot.append(tobit(x[idx], y[idx], c[idx])[1])
            level_rows.append({'pid': pid, 'features': fset, 'level': level, 'threshold': th,
                               'fp_per_min': point['fp_per_min'], 'detect_pct': point['detect_pct'],
                               'lead_med_ms': point['lead_med_ms'], 'beta': beta,
                               'beta_lo': np.nanpercentile(boot, 2.5) if boot else np.nan,
                               'beta_hi': np.nanpercentile(boot, 97.5) if boot else np.nan,
                               'censored_pct': cens, 'pre_cue_pct': pre_pct})
    pd.DataFrame(curve_rows).to_csv(out_dir / f'curves_P{pid}.csv', index=False)
    pd.concat(trial_tabs, ignore_index=True).to_csv(out_dir / f'trials_P{pid}.csv', index=False)
    pd.DataFrame(level_rows).to_csv(out_dir / f'slope_levels_P{pid}.csv', index=False)
    print(f'P{pid}: done', flush=True)


def summarize(pids, online_dir, out_dir):
    curves = pd.concat([pd.read_csv(out_dir / f'curves_P{p}.csv') for p in pids], ignore_index=True)
    levels = pd.concat([pd.read_csv(out_dir / f'slope_levels_P{p}.csv') for p in pids],
                       ignore_index=True)
    trials = pd.concat([pd.read_csv(out_dir / f'trials_P{p}.csv') for p in pids], ignore_index=True)
    ops    = pd.read_csv(online_dir / 'operating_points.csv')
    ops    = ops[ops['pid'].isin(pids)]

    # Curves on a common false-positive grid.
    grid_rows, point_rows = [], []
    for (pid, fset, win), g in curves.groupby(['pid', 'features', 'window']):
        det = _envelope(g)
        for fp, d in zip(FP_GRID, det(FP_GRID)):
            grid_rows.append({'pid': pid, 'features': fset, 'window': win, 'fp_per_min': fp,
                              'detect_pct': d})
        for fp in FP_POINTS:
            point_rows.append({'pid': pid, 'features': fset, 'window': win, 'fp_per_min': fp,
                               'detect_pct': float(det(np.array([fp]))[0])})
    grid   = pd.DataFrame(grid_rows)
    points = pd.DataFrame(point_rows)
    band   = (grid.groupby(['features', 'window', 'fp_per_min'])['detect_pct']
                  .quantile([0.25, 0.5, 0.75]).unstack().reset_index())
    band.to_csv(out_dir / 'curves_group.csv', index=False)
    points.to_csv(out_dir / 'curve_points.csv', index=False)
    point_sum = (points.groupby(['window', 'features', 'fp_per_min'])['detect_pct']
                       .agg(['mean', 'min', 'max']))

    # Slope by operating point.
    levels.to_csv(out_dir / 'slope_levels.csv', index=False)
    tests = []
    for (fset, level), g in levels.groupby(['features', 'level']):
        b = g['beta'].to_numpy()
        b = b[~np.isnan(b)]
        for null in (0.0, 1.0):
            tests.append({'features': fset, 'level': level, 'null': null, 'n': len(b),
                          'mean_beta': b.mean() if len(b) else np.nan,
                          'n_above': int((b > null).sum()), 'p_two_sided': sign_flip_p(b - null),
                          'median_censored_pct': g['censored_pct'].median(),
                          'mean_detect_pct': g['detect_pct'].mean(),
                          'mean_fp_per_min': g['fp_per_min'].mean()})
    tests = pd.DataFrame(tests)
    tests.to_csv(out_dir / 'slope_level_tests.csv', index=False)

    # Cue timer against detectors in both windows.
    timer_tests = []
    for win in WINDOWS:
        x = ops[ops['window'] == win].pivot(index='pid', columns='detector')
        for other in ('EEG+EMG', 'EMG'):
            for col in ('detect_pct', 'lead_med_ms', 'fp_per_min'):
                d = (x[col]['Cue timer'] - x[col][other]).to_numpy()
                timer_tests.append({'window': win, 'test': f'Cue timer - {other}, {col}',
                                    'mean': np.nanmean(d), 'n_above': int((d > 0).sum()),
                                    'n': len(d), 'p_two_sided': sign_flip_p(d)})
    timer_tests = pd.DataFrame(timer_tests)
    timer_tests.to_csv(out_dir / 'timer_tests.csv', index=False)

    # First-trigger latency by RT.
    nested = pd.concat([pd.read_csv(online_dir / f'slope_trials_P{p}.csv') for p in pids],
                       ignore_index=True)
    nested['level'] = 'nested 1'
    trials['level'] = trials['level'].map(lambda v: f'shared {v:g}')
    allt = pd.concat([nested, trials], ignore_index=True)
    allt = allt[~allt['pre_cue']].copy()
    allt['rt_bin'] = pd.cut(allt['rt'], RT_BINS)
    allt['early']  = ~allt['censored'] & (allt['y'] < allt['rt'] - WINDOWS['B'][0])
    binned = (allt.groupby(['features', 'level', 'rt_bin'], observed=True)
                  .agg(n=('rt', 'size'), rt_med=('rt', 'median'), first_med=('y', 'median'),
                       censored=('censored', 'mean'), early=('early', 'mean'))
                  .reset_index())
    binned.to_csv(out_dir / 'rt_binned.csv', index=False)

    with pd.option_context('display.width', 250, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(point_sum.to_string())
        print(tests.to_string())
        print(timer_tests.to_string())
        print(binned.to_string())
    _plot(band, ops, levels, binned, out_dir)


def _pool(results):
    n_hit   = sum(r['n_hit'] for r in results)
    n_trial = sum(r['n_trials'] for r in results)
    n_fp    = sum(r['n_fp'] for r in results)
    free    = sum(r['free_min'] for r in results)
    leads   = np.concatenate([r['leads'] for r in results])
    return {'detect_pct': 100 * n_hit / max(n_trial, 1), 'fp_per_min': n_fp / free,
            'lead_med_ms': 1000 * np.median(leads) if len(leads) else np.nan}


def _envelope(g):
    g   = g.sort_values(['fp_per_min', 'detect_pct'])
    fp  = g['fp_per_min'].to_numpy()
    det = np.maximum.accumulate(g['detect_pct'].to_numpy())
    return lambda q: np.where(q < fp[0], np.nan, np.interp(q, fp, det, right=np.nan))


def _plot(band, ops, levels, binned, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    colors    = {'EMG': 'tab:blue', 'EEG': 'tab:green', 'EEG+EMG': 'tab:orange'}

    # Operating curves, main window.
    ax = axes[0]
    for fset, g in band[band['window'] == MAIN_WINDOW].groupby('features'):
        ax.plot(g['fp_per_min'], g[0.5], color=colors[fset], label=f'{fset} (median)')
        ax.fill_between(g['fp_per_min'], g[0.25], g[0.75], color=colors[fset], alpha=0.15)
    tm = ops[(ops['window'] == MAIN_WINDOW) & (ops['detector'] == 'Cue timer')]
    ax.scatter(tm['fp_per_min'].clip(lower=0.1), tm['detect_pct'], color='tab:red', s=18,
               label='Cue timer (subjects)')
    ax.axvline(1.0, color='k', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('False positives per min')
    ax.set_ylabel('Detection rate (%)')
    ax.set_ylim(-2, 102)
    ax.set_title('Window [HS-1.0, HS+0.15] s, threshold swept')
    ax.legend(fontsize=8, loc='lower right')

    # Slope against operating point.
    ax = axes[1]
    for k, (fset, g) in enumerate(levels.groupby('features')):
        for level, h in g.groupby('level'):
            b = h['beta'].dropna()
            x = np.log10(level) + (k - 1) * 0.04
            ax.scatter(np.full(len(b), x), b.clip(-1, 4), color=colors[fset], s=8, alpha=0.5)
            if len(b):
                ax.scatter([x], [b.mean()], color=colors[fset], marker='D', s=40, edgecolor='k',
                           label=fset if level == 1.0 else None)
            ax.annotate(f'{len(b)}', (x, -0.95), ha='center', fontsize=7, color=colors[fset])
    ax.axhline(0, color='tab:red', ls='--', lw=1)
    ax.axhline(1, color='k', ls='--', lw=1)
    ax.set_xticks(np.log10(FP_LEVELS))
    ax.set_xticklabels([f'{v:g}' for v in FP_LEVELS])
    ax.set_ylim(-1, 4)
    ax.set_xlabel('False positives per min (threshold shared across series)')
    ax.set_ylabel('Slope of first trigger latency on RT')
    ax.set_title('Slope test by operating point (n = usable subjects)')
    ax.legend(fontsize=8)

    # First-trigger latency by RT.
    ax = axes[2]
    show = (('EMG', 'nested 1', '-'), ('EEG+EMG', 'nested 1', '-'), ('EEG', 'shared 5', '--'),
            ('EEG', 'shared 10', ':'))
    for fset, level, ls in show:
        g = binned[(binned['features'] == fset) & (binned['level'] == level)]
        ax.plot(g['rt_med'], g['first_med'], ls, marker='o', color=colors[fset],
                label=f'{fset}, {level}/min')
    r = np.linspace(0.1, 3.0, 10)
    ax.plot(r, np.minimum(r - SIM_MOV_LAG, SEARCH), color='k', lw=1, label='movement-locked')
    ax.axhline(SIM_CUE_LAG, color='tab:red', lw=1, label='cue-locked')
    ax.set_xlabel('Reaction time, bin median (s)')
    ax.set_ylabel('Median first trigger after LED (s)')
    ax.set_ylim(0, SEARCH + 0.05)
    ax.set_title('First trigger latency by RT (censored at 2 s)')
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / 'pseudo_online_followup.png', dpi=150)


if __name__ == '__main__':
    main()
