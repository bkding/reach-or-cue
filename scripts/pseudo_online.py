import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reach_or_cue.config import FEATURE_SETS, MAIN_WINDOW, MAX_CENSORED, TARGET_FP, WINDOWS
from reach_or_cue.data import label_times, load_lifts, load_series, select_test_runs
from reach_or_cue.evaluation import make_model
from reach_or_cue.features import series_features
from reach_or_cue.online import (detect_events, first_triggers, ols_slope, prep_windows,
                                 random_events, score_events, sign_flip_p, simulate, timer_delay,
                                 tobit)

# Threshold search.
THR_QUANTILES = 1 - np.logspace(np.log10(5e-5), np.log10(0.5), 160)
STOP_FP       = 3.0 # x target, end of downward scan
STOP_RUN      = 3   # consecutive candidates above STOP_FP

# Bootstrap and random seed.
N_BOOT = 300
SEED   = 0

# References and simulation.
N_RANDOM = 100
SIM_REPS = 20


def main():
    parser = argparse.ArgumentParser(description='Pseudo-online event detection, references and '
                                                 'cue/movement slope test on WAY-EEG-GAL dev '
                                                 'series.')
    parser.add_argument('data_root', help='folder holding P1, P2, ... subfolders')
    parser.add_argument('baseline_dir', help='folder holding P#/baseline_auc_folds.csv')
    parser.add_argument('--participants', default='1,2,3,4,5,6,7,8,9,10,11,12')
    parser.add_argument('--out', default='pseudo_online_out')
    parser.add_argument('--stage', choices=('score', 'evaluate', 'summarize'), default='score')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = [int(p) for p in args.participants.split(',')]
    for pid in pids:
        if args.stage == 'score':
            score_participant(Path(args.data_root) / f'P{pid}', pid, Path(args.baseline_dir),
                              out_dir)
        elif args.stage == 'evaluate':
            evaluate_participant(pid, out_dir)
    if args.stage == 'summarize':
        summarize(pids, out_dir)


def score_participant(pdir, pid, baseline_dir, out_dir):
    if (out_dir / f'scores_P{pid}.npz').exists():
        print(f'P{pid}: scores cached')
        return
    events    = load_lifts(pdir, pid)
    test_runs = select_test_runs(events)
    runs      = sorted(int(r) for r in events['run'].unique() if r not in test_runs)
    c_table   = pd.read_csv(baseline_dir / f'P{pid}' / 'baseline_auc_folds.csv')

    # Full-grid features and matched-rest labels.
    data = {}
    for run in runs:
        series   = load_series(pdir, pid, run)
        duration = min(len(series['eeg']['sig']) / series['eeg']['fs'],
                       len(series['emg']['sig']) / series['emg']['fs'])
        ev       = events[events['run'] == run]
        t, label = label_times(duration, ev, 'matched')
        feats    = series_features(series, t)
        data[run] = {'t': t, 'label': label, 'X': feats, 'duration': duration,
                     'prep': prep_windows(ev, t, *WINDOWS[MAIN_WINDOW])}
    cols = {'EMG': [c for c in feats.columns if c.startswith('emg_')],
            'EEG': [c for c in feats.columns if c.startswith('eeg_')]}
    cols['EEG+EMG'] = cols['EEG'] + cols['EMG']

    # Nested leave-one-series-out scores.
    arrays, rows = {}, []
    for held in runs:
        train_runs = [r for r in runs if r != held]
        for fset in FEATURE_SETS:
            c = float(c_table.loc[(c_table['held_series'] == held) &
                                  (c_table['features'] == fset), 'c_chosen'].iloc[0])
            arrays[f'{fset}|{held}'] = _fit_score(data, train_runs, held, cols[fset], c)

            # Out-of-fold training scores for the threshold.
            oof = {j: _fit_score(data, [r for r in train_runs if r != j], j, cols[fset], c)
                   for j in train_runs}
            thr, fp, det = _choose_threshold(oof, data, train_runs)
            rows.append({'pid': pid, 'held': held, 'features': fset, 'c': c,
                         'threshold': thr, 'train_fp_min': fp, 'train_detect': det})
        print(f'  P{pid} S{held} scored', flush=True)
    for run in runs:
        arrays[f't|{run}'] = data[run]['t']
    np.savez_compressed(out_dir / f'scores_P{pid}.npz', **arrays)
    pd.DataFrame(rows).to_csv(out_dir / f'thresholds_P{pid}.csv', index=False)
    events[events['run'].isin(runs)].to_csv(out_dir / f'lifts_P{pid}.csv', index=False)
    print(f'P{pid}: scored series {runs}, test series {test_runs} untouched', flush=True)


def evaluate_participant(pid, out_dir):
    if (out_dir / f'simulation_P{pid}.csv').exists():
        print(f'P{pid}: evaluation cached')
        return
    rng    = np.random.default_rng(SEED + pid)
    scores = np.load(out_dir / f'scores_P{pid}.npz')
    thr    = pd.read_csv(out_dir / f'thresholds_P{pid}.csv')
    events = pd.read_csv(out_dir / f'lifts_P{pid}.csv')
    runs   = sorted(int(k.split('|')[1]) for k in scores.files if k.startswith('t|'))

    ev_rows, trial_rows = [], []
    for run in runs:
        t     = scores[f't|{run}']
        ev    = events[events['run'] == run]
        preps = {name: prep_windows(ev, t, *w) for name, w in WINDOWS.items()}
        for fset in FEATURE_SETS:
            th   = float(thr.loc[(thr['held'] == run) & (thr['features'] == fset),
                                 'threshold'].iloc[0])
            trig = detect_events(t, scores[f'{fset}|{run}'], th)
            ev_rows += _score_all(pid, run, fset, trig, preps)
            trial_rows += first_triggers(pid, run, fset, trig, ev)

            # Rate-matched random detector.
            rate = len(trig) / ((t[-1] - t[0]) / 60)
            for rep in range(N_RANDOM):
                rand = random_events(rng, rate, t[0], t[-1])
                ev_rows += _score_all(pid, run, f'Random ({fset} rate)', rand, preps, rep)

        # Cue timer, delay chosen on the other dev series.
        train = events[events['run'] != run]
        for name, (pre, post) in WINDOWS.items():
            d    = timer_delay(train, pre, post)
            trig = np.sort(ev['led_on'].dropna().to_numpy() + d)
            ev_rows += [r for r in _score_all(pid, run, 'Cue timer', trig, preps)
                        if r['window'] == name]
    pd.DataFrame(ev_rows).to_csv(out_dir / f'events_P{pid}.csv', index=False)

    # Slope estimates with trial bootstrap.
    tr = pd.DataFrame(trial_rows)
    tr.to_csv(out_dir / f'slope_trials_P{pid}.csv', index=False)
    slope_rows = []
    for fset, g in tr.groupby('features'):
        pre_pct = 100 * g['pre_cue'].mean()
        g       = g[~g['pre_cue']]
        x, y, c = g['rt'].to_numpy(), g['y'].to_numpy(), g['censored'].to_numpy()
        est     = tobit(x, y, c)
        boot    = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(g), len(g))
            boot.append(tobit(x[idx], y[idx], c[idx])[1])
        naive = g[g['hit_B']]
        slope_rows.append({'pid': pid, 'features': fset, 'n': len(g), 'alpha': est[0],
                           'beta': est[1], 'beta_lo': np.nanpercentile(boot, 2.5),
                           'beta_hi': np.nanpercentile(boot, 97.5),
                           'censored_pct': 100 * g['censored'].mean(), 'pre_cue_pct': pre_pct,
                           'beta_naive_hits': ols_slope(naive['rt'], naive['hit_time_rel']),
                           'rt_iqr': g['rt'].quantile(0.75) - g['rt'].quantile(0.25)})
    pd.DataFrame(slope_rows).to_csv(out_dir / f'slope_P{pid}.csv', index=False)

    # Estimator check on simulated streams.
    sim_rows = []
    for kind in ('cue-locked', 'movement-locked', 'random'):
        for rep in range(SIM_REPS):
            rows = []
            for run in runs:
                t  = scores[f't|{run}']
                ev = events[events['run'] == run]
                rows += first_triggers(pid, run, kind, simulate(rng, kind, ev, t), ev)
            g     = pd.DataFrame(rows)
            g     = g[~g['pre_cue']]
            est   = tobit(g['rt'].to_numpy(), g['y'].to_numpy(), g['censored'].to_numpy())
            naive = g[g['hit_B']]
            sim_rows.append({'pid': pid, 'kind': kind, 'rep': rep, 'beta': est[1],
                             'beta_naive_hits': ols_slope(naive['rt'], naive['hit_time_rel']),
                             'censored_pct': 100 * g['censored'].mean()})
    pd.DataFrame(sim_rows).to_csv(out_dir / f'simulation_P{pid}.csv', index=False)
    print(f'P{pid}: evaluated', flush=True)


def summarize(pids, out_dir):
    ev = pd.concat([pd.read_csv(out_dir / f'events_P{p}.csv') for p in pids], ignore_index=True)
    ev['leads'] = ev['leads'].fillna('')

    # Per-subject operating points, random reps averaged.
    keys = ['pid', 'detector', 'window', 'rep']
    agg  = (ev.groupby(keys)
              .agg(n_trials=('n_trials', 'sum'), n_hit=('n_hit', 'sum'), n_fp=('n_fp', 'sum'),
                   free_min=('free_min', 'sum'), n_motion=('n_motion', 'sum'),
                   leads=('leads', lambda s: ';'.join(x for x in s if x)))
              .reset_index())
    agg['lead_med_ms'] = agg['leads'].map(
        lambda s: 1000 * np.median(np.array(s.split(';'), float)) if s else np.nan)
    agg['detect_pct']  = 100 * agg['n_hit'] / agg['n_trials']
    agg['fp_per_min']  = agg['n_fp'] / agg['free_min']
    agg['motion_per_trial'] = agg['n_motion'] / agg['n_trials']
    cols    = ['detect_pct', 'lead_med_ms', 'fp_per_min', 'motion_per_trial']
    per_sub = agg.groupby(['pid', 'detector', 'window'])[cols].mean().reset_index()
    per_sub.to_csv(out_dir / 'operating_points.csv', index=False)
    group = per_sub.groupby(['window', 'detector'])[cols].agg(['mean', 'min', 'max'])
    group.to_csv(out_dir / 'operating_points_group.csv')

    # Slopes and simulation.
    sl  = pd.concat([pd.read_csv(out_dir / f'slope_P{p}.csv') for p in pids], ignore_index=True)
    sim = pd.concat([pd.read_csv(out_dir / f'simulation_P{p}.csv') for p in pids],
                    ignore_index=True)
    sl['beta_used'] = sl['beta'].where(sl['censored_pct'] <= MAX_CENSORED)
    sl.to_csv(out_dir / 'slope.csv', index=False)
    sim['beta'] = sim['beta'].where(sim['censored_pct'] <= MAX_CENSORED)
    sim_sum = (sim.groupby(['pid', 'kind'])[['beta', 'beta_naive_hits', 'censored_pct']]
                  .mean().reset_index())
    sim_sum.to_csv(out_dir / 'simulation.csv', index=False)

    # Paired sign-flip tests across subjects.
    tests = []
    for fset, g in sl.groupby('features'):
        for null in (0.0, 1.0):
            d = g['beta_used'].to_numpy() - null
            d = d[~np.isnan(d)]
            tests.append({'test': f'{fset} slope vs {null:g}',
                          'mean': d.mean() + null if len(d) else np.nan,
                          'n_above': int((d > 0).sum()), 'n': len(d),
                          'p_two_sided': sign_flip_p(d)})
    main = per_sub[per_sub['window'] == MAIN_WINDOW]
    for a, b in (('EEG+EMG', 'EMG'), ('EEG', 'EMG'), ('Cue timer', 'EEG+EMG'),
                 ('EEG', 'Random (EEG rate)'), ('EEG+EMG', 'Random (EEG+EMG rate)')):
        for col in ('lead_med_ms', 'detect_pct'):
            x = main.pivot(index='pid', columns='detector', values=col)
            d = (x[a] - x[b]).to_numpy()
            tests.append({'test': f'{a} - {b}, {col}', 'mean': np.nanmean(d),
                          'n_above': int((d > 0).sum()), 'n': len(d),
                          'p_two_sided': sign_flip_p(d)})
    tests = pd.DataFrame(tests)
    tests.to_csv(out_dir / 'tests.csv', index=False)

    with pd.option_context('display.width', 250, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(group.to_string())
        print(sl[['pid', 'features', 'n', 'beta', 'beta_used', 'beta_lo', 'beta_hi', 'censored_pct',
                  'pre_cue_pct', 'beta_naive_hits', 'rt_iqr']].to_string())
        print(sim_sum.groupby('kind')[['beta', 'beta_naive_hits', 'censored_pct']]
                     .agg(['mean', 'min', 'max']).to_string())
        print(tests.to_string())
    _plot(per_sub, sl, sim_sum, out_dir)


def _fit_score(data, train_runs, score_run, cols, c):
    X = pd.concat([data[r]['X'][data[r]['label'] >= 0] for r in train_runs])
    y = np.concatenate([data[r]['label'][data[r]['label'] >= 0] for r in train_runs])
    model = make_model(c).fit(X[cols], y)
    return model.predict_proba(data[score_run]['X'][cols])[:, 1]


def _choose_threshold(oof, data, train_runs):
    allscores = np.concatenate(list(oof.values()))
    cands     = np.unique(np.quantile(allscores, THR_QUANTILES))[::-1]
    best, over = (-1.0, np.inf, cands[0]), 0
    for th in cands:
        n_hit = n_trial = n_fp = 0
        free  = 0.0
        for j in train_runs:
            r = score_events(detect_events(data[j]['t'], oof[j], th), data[j]['prep'])
            n_hit += r['n_hit']; n_trial += r['n_trials']; n_fp += r['n_fp']; free += r['free_min']
        fp, det = n_fp / free, n_hit / max(n_trial, 1)
        if fp <= TARGET_FP and (det > best[0] or (det == best[0] and fp < best[1])):
            best = (det, fp, th)
        over = over + 1 if fp > STOP_FP * TARGET_FP else 0
        if over >= STOP_RUN:
            break
    return best[2], best[1], best[0]


def _score_all(pid, run, detector, trig, preps, rep=0):
    rows = []
    for name, p in preps.items():
        r = score_events(trig, p)
        rows.append({'pid': pid, 'run': run, 'detector': detector, 'window': name, 'rep': rep,
                     'n_trials': r['n_trials'], 'n_hit': r['n_hit'], 'n_fp': r['n_fp'],
                     'n_motion': r['n_motion'], 'free_min': r['free_min'],
                     'leads': ';'.join(f'{x:.4f}' for x in r['leads'])})
    return rows


def _plot(per_sub, sl, sim_sum, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # Operating points, main window.
    ax  = axes[0]
    ops = per_sub[per_sub['window'] == MAIN_WINDOW]
    for det, color in (('EMG', 'C0'), ('EEG', 'C2'), ('EEG+EMG', 'C1'), ('Cue timer', 'C3'),
                       ('Random (EEG+EMG rate)', '0.5')):
        g = ops[ops['detector'] == det]
        ax.scatter(g['lead_med_ms'], g['detect_pct'], color=color, s=18, label=det)
    ax.set_xlabel('Median lead before hand start (ms)')
    ax.set_ylabel('Detection rate (%)')
    ax.set_title('Window [HS-1.0, HS+0.15] s, threshold for 1 FP/min', fontsize=9)
    ax.legend(fontsize=7)

    # Slopes per subject.
    ax = axes[1]
    for k, (fset, color) in enumerate((('EMG', 'C0'), ('EEG', 'C2'), ('EEG+EMG', 'C1'))):
        g = sl[(sl['features'] == fset) & sl['beta_used'].notna()].sort_values('pid')
        x = g['pid'] + (k - 1) * 0.22
        ax.errorbar(x, g['beta'], yerr=[(g['beta'] - g['beta_lo']).clip(lower=0),
                                        (g['beta_hi'] - g['beta']).clip(lower=0)],
                    fmt='o', color=color, ms=4, lw=1, label=fset)
    ax.axhline(0, color='C3', lw=1, ls='--')
    ax.axhline(1, color='0.4', lw=1, ls='--')
    ax.set_ylim(-1.0, 2.0)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel('Participant')
    ax.set_ylabel('Slope of trigger latency on RT')
    ax.set_title('Slope test (0 = follows LED, 1 = follows hand)', fontsize=9)
    ax.legend(fontsize=7)

    # Estimator check.
    ax = axes[2]
    for k, (kind, color) in enumerate((('cue-locked', 'C3'), ('movement-locked', '0.3'),
                                       ('random', '0.6'))):
        g = sim_sum[sim_sum['kind'] == kind].sort_values('pid')
        ax.scatter(g['pid'] + (k - 1) * 0.22, g['beta'], color=color, s=18,
                   label=f'{kind}, censored regression')
        ax.scatter(g['pid'] + (k - 1) * 0.22, g['beta_naive_hits'], color=color, s=22, marker='x',
                   label=f'{kind}, hits only')
    ax.axhline(0, color='C3', lw=1, ls='--')
    ax.axhline(1, color='0.4', lw=1, ls='--')
    ax.set_ylim(-1.0, 2.0)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel('Participant')
    ax.set_title('Simulated streams: does the estimator recover 0 and 1?', fontsize=9)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / 'pseudo_online.png', dpi=150)


if __name__ == '__main__':
    main()
