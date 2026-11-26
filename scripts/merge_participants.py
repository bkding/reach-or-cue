import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reach_or_cue.data import _get, label_times, load_lifts, load_mat, select_test_runs
from reach_or_cue.evaluation import report

# Cue-only reference.
CUE_FEATURE = 'Cue (LED on)'
DUR_PAD     = 1.0 # s, labeled windows end before the last hand stop

# Paired comparisons.
PAIRS = (('EEG+EMG', 'EMG'), ('EEG', 'EMG'), ('Cue (LED on)', 'EEG'),
         ('Cue (LED on)', 'EMG'), ('Cue (LED on)', 'EEG+EMG'))


def main():
    parser = argparse.ArgumentParser(description='Merge per-participant baseline folds, add '
                                                 'cue-only references and paired sign-flip tests.')
    parser.add_argument('data_root', help='folder holding P1, P2, ... subfolders')
    parser.add_argument('run_dir', help='folder holding P#/baseline_auc_folds.csv')
    parser.add_argument('--rest', choices=('matched', 'all'), default='matched')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    folds   = pd.concat([pd.read_csv(f) for f in sorted(run_dir.glob('P*/baseline_auc_folds.csv'))],
                        ignore_index=True)
    pids    = sorted(folds['participant'].unique())
    report(folds, run_dir)

    # Cue-only folds.
    cue = pd.concat([cue_folds(Path(args.data_root) / f'P{pid}', pid, args.rest) for pid in pids],
                    ignore_index=True)
    check = cue.merge(folds[folds['features'] == 'EMG'], on=['participant', 'held_series'],
                      suffixes=('', '_emg'))
    assert ((check['n_pos'] == check['n_pos_emg']).all()
            and (check['n_rest'] == check['n_rest_emg']).all())
    cue.to_csv(run_dir / 'cue_only_folds.csv', index=False)

    # Subject means.
    allf    = pd.concat([folds[['participant', 'held_series', 'features', 'auc']],
                         cue[['participant', 'held_series', 'features', 'auc']]])
    subject = allf.groupby(['participant', 'features'])['auc'].mean().unstack()
    subject.to_csv(run_dir / 'subject_auc.csv')

    # Paired sign-flip tests.
    rows = []
    for a, b in PAIRS:
        diff = (subject[a] - subject[b]).to_numpy()
        rows.append({'comparison': f'{a} - {b}', 'n': len(diff), 'mean_diff': diff.mean(),
                     'n_positive': int((diff > 0).sum()), 'p_two_sided': sign_flip_p(diff)})
    tests = pd.DataFrame(rows)
    tests.to_csv(run_dir / 'paired_tests.csv', index=False)
    with pd.option_context('display.width', 200, 'display.float_format', '{:.3f}'.format):
        print(subject.to_string())
        print(subject.mean().to_string())
        print(tests.to_string())
    _plot(subject, run_dir)


def cue_folds(pdir, pid, rest_mode):
    events    = load_lifts(pdir, pid)
    test_runs = select_test_runs(events)
    led       = _led_intervals(pdir, pid)
    rows      = []
    for run in sorted(int(r) for r in events['run'].unique() if r not in test_runs):
        ev           = events[events['run'] == run]
        t_dec, label = label_times(ev['hand_stop'].max() + DUR_PAD, ev, rest_mode)
        t, y         = t_dec[label >= 0], label[label >= 0]

        # Current LED state.
        on  = led.loc[led['run'] == run, ['led_on', 'led_off']].to_numpy()
        lit = ((t[:, None] >= on[:, 0]) & (t[:, None] < on[:, 1])).any(axis=1)
        rows.append({'participant': pid, 'held_series': run, 'features': CUE_FEATURE,
                     'auc': roc_auc_score(y, lit.astype(float)), 'n_pos': int(y.sum()),
                     'n_rest': int((y == 0).sum()), 'pos_lit': float(lit[y == 1].mean()),
                     'rest_lit': float(lit[y == 0].mean())})
    return pd.DataFrame(rows)


def sign_flip_p(diff):
    obs   = abs(diff.mean())
    signs = np.array(list(itertools.product((-1, 1), repeat=len(diff))))
    return float(np.mean(np.abs((signs * diff).mean(axis=1)) >= obs - 1e-12))


def _plot(subject, run_dir):
    order  = ['EMG', 'EEG', 'EEG+EMG', CUE_FEATURE]
    colors = {'EMG': 'C0', 'EEG': 'C2', 'EEG+EMG': 'C1', CUE_FEATURE: 'C3'}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(subject))
    for k, name in enumerate(order):
        ax.plot(x + (k - 1.5) * 0.15, subject[name], 'o', color=colors[name], ms=5,
                label=f'{name} (mean {subject[name].mean():.3f})')
    ax.axhline(0.5, color='0.7', lw=1)
    ax.set_xticks(x, [f'P{p}' for p in subject.index])
    ax.set_ylabel('Window AUC, dev series')
    ax.set_ylim(0.45, 1.01)
    ax.set_title('Reach onset vs matched rest: physiological detectors and the cue-only reference',
                 fontsize=9)
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(run_dir / 'cue_vs_detectors.png', dpi=150)


def _led_intervals(pdir, pid):
    p     = _get(load_mat(pdir / f'P{pid}_AllLifts.mat'), 'P')
    col   = {str(c): i for i, c in enumerate(np.atleast_1d(_get(p, 'ColNames')))}
    lifts = np.asarray(_get(p, 'AllLifts'), dtype=float)

    # LED interval on the series clock.
    t0 = lifts[:, col['StartTime']] - lifts[:, col['LEDOn']]
    return pd.DataFrame({'run':     lifts[:, col['Run']].astype(int),
                         'led_on':  t0 + lifts[:, col['LEDOn']],
                         'led_off': t0 + lifts[:, col['LEDOff']]})


if __name__ == '__main__':
    main()
