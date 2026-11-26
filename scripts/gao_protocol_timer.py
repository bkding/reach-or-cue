import argparse
import io
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy.stats import spearmanr

from reach_or_cue.data import load_lifts, load_series, select_test_runs

# Gao 2026 valid interval.
CUE_GUARD = 0.1 # s, earliest valid response after LED
PRE_HS    = 0.2 # s, interval start before hand start
POST_HS   = 0.1 # s, interval end after hand start

# Timer delay grid.
D_MIN  = 0.1   # s
D_MAX  = 1.5   # s
D_STEP = 0.005 # s

# Gao 2026 within-subject group means, nominal 1 FP/min.
GAO_GROUP  = {'ordinary': (86.52, 65.36), 'event': (86.38, 81.70)} # detect %, median lead ms
GAO_FP_MIN = (1.334, 1.387)                                       # FP/min
GAO_LEAD   = 0.0817                                               # s, event-guided median lead

# Gao 2026 Table 3, nominal 1 FP/min: detect %, median lead ms, FP/min.
GAO_SUBJECT = {
    1:  {'ordinary': (90.1, 44.4, 1.819), 'event': (83.8, 68.4,  0.737)},
    2:  {'ordinary': (81.5, 97.8, 0.533), 'event': (83.7, 100.8, 1.082)},
    4:  {'ordinary': (97.0, 49.1, 0.686), 'event': (97.0, 72.9,  0.315)},
    6:  {'ordinary': (92.8, 53.9, 0.451), 'event': (92.5, 74.6,  0.718)},
    9:  {'ordinary': (86.3, 57.5, 0.175), 'event': (85.0, 72.1,  0.496)},
    10: {'ordinary': (81.6, 88.9, 1.635), 'event': (88.2, 99.0,  1.838)},
    11: {'ordinary': (86.0, 51.3, 3.602), 'event': (78.0, 71.4,  4.146)},
    12: {'ordinary': (76.7, 80.1, 2.195), 'event': (83.0, 94.4,  1.340)},
}

# Delay selection rules.
RULES      = ('gao_rule', 'lead_group', 'lead_subject')
EARLY_LEAD = 0.1 # s, early-detection threshold


def main():
    parser = argparse.ArgumentParser(description='Cue-timer detector under the Gao 2026 '
                                                 'event protocol on WAY-EEG-GAL dev series.')
    parser.add_argument('data_root', help='folder holding P1, P2, ... subfolders')
    parser.add_argument('zip_root', help='folder holding the original P#.zip files')
    parser.add_argument('--participants', default='1,2,3,4,5,6,7,8,9,10,11,12')
    parser.add_argument('--out', default='gao_timer_out')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pids = [int(p) for p in args.participants.split(',')]
    if not args.summarize_only:
        for pid in pids:
            run_participant(Path(args.data_root) / f'P{pid}', Path(args.zip_root) / f'P{pid}.zip',
                            pid, out_dir)
    else:
        summarize(pids, out_dir)


def run_participant(pdir, zpath, pid, out_dir):
    if (out_dir / f'trials_P{pid}.csv').exists():
        print(f'P{pid}: cached')
        return
    events    = load_lifts(pdir, pid)
    test_runs = select_test_runs(events)
    dev       = events[~events['run'].isin(test_runs)].copy()
    dev_runs  = sorted(int(r) for r in dev['run'].unique())
    print(f'P{pid}: test series {test_runs}, dev series {dev_runs}', flush=True)

    # Valid intervals.
    dev['v_start'] = np.maximum(dev['led_on'] + CUE_GUARD, dev['hand_start'] - PRE_HS)
    dev['v_end']   = dev['hand_start'] + POST_HS
    dev['scored']  = dev['hand_start'].notna()

    # Background minutes.
    minutes = pd.DataFrame([_run_minutes(pdir, zpath, pid, run, dev[dev['run'] == run])
                            for run in dev_runs])
    minutes.to_csv(out_dir / f'minutes_P{pid}.csv', index=False)

    # Leave-one-series-out delay.
    grid   = np.round(np.arange(D_MIN, D_MAX + D_STEP / 2, D_STEP), 4)
    target = GAO_SUBJECT[pid]['event'][1] / 1000 if pid in GAO_SUBJECT else np.nan
    parts  = []
    for run in dev_runs:
        train = dev[(dev['run'] != run) & dev['scored']]
        for rule, d in _select_delays(train, grid, target).items():
            held = dev[dev['run'] == run].copy()
            fire = held['led_on'] + d
            held['rule'] = rule
            held['d']    = d
            held['hit']  = (fire >= held['v_start']) & (fire <= held['v_end']) & held['scored']
            held['lead'] = held['hand_start'] - fire
            parts.append(held)
    trials = pd.concat(parts, ignore_index=True)
    trials.insert(0, 'pid', pid)
    trials.to_csv(out_dir / f'trials_P{pid}.csv', index=False)

    # Delay sweep on all dev trials.
    scored = dev[dev['scored']]
    hits   = _hits(scored, grid)
    lead   = (scored['hand_start'].to_numpy()[:, None]
              - (scored['led_on'].to_numpy()[:, None] + grid))
    sweep  = pd.DataFrame({
        'd':       grid,
        'detect':  100 * hits.mean(axis=0),
        'lead_ms': [1000 * np.median(lead[hits[:, j], j]) if hits[:, j].any() else np.nan
                    for j in range(len(grid))],
        'fp_min':  (~hits).sum(axis=0) / minutes['ws_bg_min'].sum(),
    })
    sweep.insert(0, 'pid', pid)
    sweep.to_csv(out_dir / f'sweep_P{pid}.csv', index=False)


def summarize(pids, out_dir):
    # Per-subject operating points.
    rows = []
    for pid in pids:
        trials  = pd.read_csv(out_dir / f'trials_P{pid}.csv')
        minutes = pd.read_csv(out_dir / f'minutes_P{pid}.csv')
        rt      = trials.loc[(trials['rule'] == RULES[0]) & trials['scored'], 'rt']
        for rule, t in trials.groupby('rule', sort=False):
            scored = t[t['scored']]
            hits   = scored[scored['hit']]
            n_fp   = int((~scored['hit']).sum())
            rows.append({
                'rule':          rule,
                'pid':           pid,
                'confirmation':  pid in GAO_SUBJECT,
                'n_trials':      len(scored),
                'n_unscored':    int((~t['scored']).sum()),
                'n_rt_lt_0.1':   int((scored['rt'] < 0.1).sum()),
                'rt_median_s':   rt.median(),
                'rt_iqr_s':      rt.quantile(0.75) - rt.quantile(0.25),
                'd_min_s':       t['d'].min(),
                'd_max_s':       t['d'].max(),
                'detect_pct':    100 * len(hits) / len(scored),
                'lead_med_ms':   1000 * hits['lead'].median(),
                'early_pct':     100 * (hits['lead'] >= EARLY_LEAD).sum() / len(scored),
                'n_fp':          n_fp,
                'fp_min_ws':     n_fp / minutes['ws_bg_min'].sum(),
                'fp_min_series': n_fp / minutes['series_bg_min'].sum(),
                'ws_min':        minutes['ws_min'].sum(),
                'series_min':    minutes['series_min'].sum(),
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / 'summary.csv', index=False)

    # Group means.
    cols  = ['detect_pct', 'lead_med_ms', 'early_pct', 'fp_min_ws', 'fp_min_series']
    means = []
    for (rule, conf), g in summary.groupby(['rule', 'confirmation'], sort=False):
        means.append({'rule': rule, 'subjects': 'confirmation' if conf else 'development',
                      'n': len(g), **g[cols].mean().to_dict()})
    for rule, g in summary.groupby('rule', sort=False):
        means.append({'rule': rule, 'subjects': 'all', 'n': len(g), **g[cols].mean().to_dict()})
    means = pd.DataFrame(means).sort_values(['rule', 'subjects'])
    means.to_csv(out_dir / 'group_means.csv', index=False)

    # Head-to-head with Gao Table 3.
    comp = []
    for pid in [p for p in pids if p in GAO_SUBJECT]:
        row = {'pid': pid}
        for name in ('ordinary', 'event'):
            (row[f'gao_{name}_detect'], row[f'gao_{name}_lead'],
             row[f'gao_{name}_fp']) = GAO_SUBJECT[pid][name]
        for rule in RULES:
            r = summary[(summary['pid'] == pid) & (summary['rule'] == rule)].iloc[0]
            row[f'{rule}_detect'] = r['detect_pct']
            row[f'{rule}_lead']   = r['lead_med_ms']
            row[f'{rule}_fp']     = r['fp_min_ws']
        r = summary[(summary['pid'] == pid) & (summary['rule'] == RULES[0])].iloc[0]
        row['rt_median_s'], row['rt_iqr_s'] = r['rt_median_s'], r['rt_iqr_s']
        row['beats_event_all3'] = (row['lead_subject_detect'] >= row['gao_event_detect']
                                   and row['lead_subject_lead'] >= row['gao_event_lead']
                                   and row['lead_subject_fp'] <= row['gao_event_fp'])
        comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(out_dir / 'gao_comparison.csv', index=False)

    # RT spread vs timer detection.
    lg  = summary[summary['rule'] == 'lead_group']
    rho = spearmanr(lg['rt_iqr_s'], lg['detect_pct'])
    with pd.option_context('display.width', 250, 'display.max_columns', 30,
                           'display.float_format', '{:.3f}'.format):
        print(summary[['rule', 'pid', 'confirmation', 'n_trials', 'rt_median_s', 'rt_iqr_s',
                       'detect_pct', 'lead_med_ms', 'fp_min_ws', 'fp_min_series']].to_string())
        print(means.to_string())
        print(comp.to_string())
    print(f'Spearman RT IQR vs lead_group detection: rho = {rho.statistic:.3f}, '
          f'p = {rho.pvalue:.4f}, n = {len(lg)}')
    _plot_subjects(pids, summary, out_dir)
    _plot_spread(summary, comp, out_dir)


def _select_delays(train, grid, target):
    hits   = _hits(train, grid)
    lead   = train['hand_start'].to_numpy()[:, None] - (train['led_on'].to_numpy()[:, None] + grid)
    detect = hits.mean(axis=0)

    # Gao rule: max detection, then max mean lead with misses as zero.
    mean_lead = np.where(hits, lead, 0).mean(axis=0)
    best      = np.flatnonzero(detect == detect.max())
    delays    = {'gao_rule': grid[best[np.argmax(mean_lead[best])]]}

    # Lead-constrained rules: max detection with median lead at least the target.
    med_lead = np.array([np.median(lead[hits[:, j], j]) if hits[:, j].any() else -np.inf
                         for j in range(len(grid))])
    for rule, floor in (('lead_group', GAO_LEAD), ('lead_subject', target)):
        if np.isnan(floor):
            continue
        ok = np.flatnonzero(med_lead >= floor)
        delays[rule] = grid[ok[np.argmax(detect[ok])]] if ok.size else np.nan
    return delays


def _hits(ev, grid):
    fire = ev['led_on'].to_numpy()[:, None] + grid
    return (fire >= ev['v_start'].to_numpy()[:, None]) & (fire <= ev['v_end'].to_numpy()[:, None])


def _run_minutes(pdir, zpath, pid, run, ev):
    # WS trial windows.
    with zipfile.ZipFile(zpath) as z:
        ws = scipy.io.loadmat(io.BytesIO(z.read(f'WS_P{pid}_S{run}.mat')),
                              squeeze_me=True, struct_as_record=False)['ws']
    wins   = np.atleast_1d(ws.win)
    ws_sec = sum(float(w.eeg_t[-1] - w.eeg_t[0]) for w in wins)

    # Continuous series.
    eeg        = load_series(pdir, pid, run)['eeg']
    series_sec = len(eeg['sig']) / eeg['fs']

    # Valid-interval time.
    scored    = ev[ev['hand_start'].notna()]
    valid_sec = float(np.clip(scored['v_end'] - scored['v_start'], 0, None).sum())
    if len(wins) != len(ev):
        print(f'  P{pid} S{run}: {len(wins)} WS windows vs {len(ev)} lifts', flush=True)
    return {'run': run, 'block': int(ev['block'].iloc[0]), 'n_lifts': len(ev), 'n_ws': len(wins),
            'ws_min': ws_sec / 60, 'ws_bg_min': (ws_sec - valid_sec) / 60,
            'series_min': series_sec / 60, 'series_bg_min': (series_sec - valid_sec) / 60}


def _plot_subjects(pids, summary, out_dir):
    ncol = 4
    nrow = int(np.ceil(len(pids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), sharex=True, sharey=True)
    markers = {'gao_rule': 'o', 'lead_group': 's', 'lead_subject': 'D'}
    for ax, pid in zip(axes.flat, pids):
        sweep = pd.read_csv(out_dir / f'sweep_P{pid}.csv')
        ax.plot(sweep['lead_ms'], sweep['detect'], color='0.65', lw=1, label='timer, delay sweep')

        # Timer operating points.
        for _, r in summary[summary['pid'] == pid].iterrows():
            ax.plot(r['lead_med_ms'], r['detect_pct'], markers[r['rule']], color='C0', ms=6,
                    label=f'timer, {r["rule"]}')
        r = summary[(summary['pid'] == pid) & (summary['rule'] == 'lead_subject')]
        if len(r):
            r = r.iloc[0]
            ax.annotate(f'{r["fp_min_ws"]:.2f}/min', (r['lead_med_ms'], r['detect_pct']),
                        textcoords='offset points', xytext=(5, 4), fontsize=7, color='C0')

        # Gao operating points.
        for name, marker in (('ordinary', '^'), ('event', '*')):
            if pid in GAO_SUBJECT:
                det, lead, fp = GAO_SUBJECT[pid][name]
                ax.plot(lead, det, marker, color='C3', ms=8, label=f'Gao {name}')
                if name == 'event':
                    ax.annotate(f'{fp:.2f}/min', (lead, det), textcoords='offset points',
                                xytext=(5, -10), fontsize=7, color='C3')
        rt = summary[summary['pid'] == pid].iloc[0]
        tag = '' if pid in GAO_SUBJECT else ', Gao dev subject'
        ax.set_title(f'P{pid} (RT IQR {1000 * rt["rt_iqr_s"]:.0f} ms{tag})', fontsize=8)
        ax.set_xlim(-50, 250)
        ax.set_ylim(0, 100)
    for ax in axes.flat[len(pids):]:
        ax.axis('off')
    for ax in axes[-1]:
        ax.set_xlabel('Median lead before hand start (ms)', fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel('Valid detection rate (%)', fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=7, frameon=False)
    fig.suptitle('Cue timer (no physiological input) vs Gao 2026 under the Gao protocol, '
                 'dev series',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out_dir / 'gao_timer_subjects.png', dpi=150)


def _plot_spread(summary, comp, out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.8))
    lg = summary[summary['rule'] == 'lead_group']
    for conf, color in ((True, 'C3'), (False, '0.5')):
        g = lg[lg['confirmation'] == conf]
        a.scatter(1000 * g['rt_iqr_s'], g['detect_pct'], color=color,
                  label='Gao confirmation' if conf else 'Gao development')
    for _, r in lg.iterrows():
        a.annotate(f'P{r["pid"]}', (1000 * r['rt_iqr_s'], r['detect_pct']),
                   textcoords='offset points', xytext=(4, 3), fontsize=7)
    a.set_xlabel('Reaction-time IQR (ms)')
    a.set_ylabel('Timer detection rate (%), lead >= 81.7 ms')
    a.legend(fontsize=7)

    diff = comp['lead_subject_detect'] - comp['gao_event_detect']
    b.axhline(0, color='0.6', lw=1)
    b.scatter(1000 * comp['rt_iqr_s'], diff, color='C3')
    for x, y, pid in zip(1000 * comp['rt_iqr_s'], diff, comp['pid']):
        b.annotate(f'P{pid}', (x, y), textcoords='offset points', xytext=(4, 3), fontsize=7)
    b.set_xlabel('Reaction-time IQR (ms)')
    b.set_ylabel('Timer minus Gao detection (pp)\nat Gao\'s own median lead')
    fig.tight_layout()
    fig.savefig(out_dir / 'rt_spread.png', dpi=150)


if __name__ == '__main__':
    main()
