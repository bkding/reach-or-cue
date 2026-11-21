import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reach_or_cue.config import STEP, PRE_ONSET, MIN_RT, WEIGHT_BLOCK
from reach_or_cue.data import load_lifts, load_series, label_times, select_test_runs
from reach_or_cue.evaluation import cross_validate, report
from reach_or_cue.features import emg_envelope, eeg_mrcp, series_features
from reach_or_cue.qc import epoch_mean, hand_speed, plot_qc


def main():
    parser = argparse.ArgumentParser(description='Window-level EMG / EEG / fusion baseline for '
                                                 'reach-onset detection on WAY-EEG-GAL.')
    parser.add_argument('data_root', help='folder holding P1, P2, ... subfolders')
    parser.add_argument('--participants', default='1,2,3')
    parser.add_argument('--out', default='baseline_out')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--rest', choices=('matched', 'all'), default='matched')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = []
    for pid in [int(p) for p in args.participants.split(',')]:
        result = run_participant(Path(args.data_root) / f'P{pid}', pid, out_dir,
                                 args.check_only, args.rest)
        if result is not None:
            folds.append(result)
    if folds:
        report(pd.concat(folds, ignore_index=True), out_dir)
    print(f'\nOutputs in {out_dir.resolve()}')


def run_participant(pdir, pid, out_dir, check_only, rest_mode):
    events    = load_lifts(pdir, pid)
    runs      = sorted(int(r) for r in events['run'].unique())
    test_runs = select_test_runs(events)
    windows, shapes = [], {}
    rest_min = {'dev': 0.0, 'test': 0.0, 'dev_weight': 0.0, 'dev_long': 0.0}
    qc       = {'emg': [], 'mrcp': [], 'speed': {}, 'led_rel': []}

    for run in runs:
        # Series signals and durations.
        ev = events[events['run'] == run]
        series = load_series(pdir, pid, run)
        eeg, emg = series['eeg'], series['emg']
        dur_eeg = len(eeg['sig']) / eeg['fs']
        dur_emg = len(emg['sig']) / emg['fs']
        shapes[run] = (f'EEG {eeg["sig"].shape} @ {eeg["fs"]:g} Hz, EMG {emg["sig"].shape} @ '
                       f'{emg["fs"]:g} Hz, dur EEG/EMG {dur_eeg:.1f}/{dur_emg:.1f} s, '
                       f'last hand stop {ev["hand_stop"].max():.1f} s')

        # Labels and rest minutes.
        t_dec, label = label_times(min(dur_eeg, dur_emg), ev, rest_mode)
        split = 'test' if run in test_runs else 'dev'
        block = int(ev['block'].iloc[0])
        rest_min[split] += np.sum(label == 0) * STEP / 60
        if split == 'dev':
            rest_min['dev_weight' if block == WEIGHT_BLOCK else 'dev_long'] += (
                np.sum(label == 0) * STEP / 60)

        # QC epochs around hand start, dev series only.
        if split == 'dev':
            onsets = ev['hand_start'].dropna().to_numpy()
            _, emg_avg = epoch_mean(np.log(emg_envelope(emg['sig'], emg['fs'])), emg['fs'],
                                    onsets)
            mrcp, mrcp_names, _ = eeg_mrcp(eeg['sig'], eeg['fs'], eeg['names'])
            _, mrcp_avg = epoch_mean(mrcp, eeg['fs'], onsets)
            qc['emg'].append(emg_avg)
            qc['mrcp'].append(mrcp_avg)
            qc['led_rel'].extend((ev['led_on'] - ev['hand_start']).dropna())
            if 'kin' in series:
                for name, speed in hand_speed(series['kin']).items():
                    _, speed_avg = epoch_mean(speed[:, None], series['kin']['fs'], onsets)
                    qc['speed'].setdefault(name, []).append(speed_avg)

        # Features at labeled decision times.
        if not check_only:
            feats = series_features(series, t_dec[label >= 0])
            feats['label'] = label[label >= 0]
            feats['run']   = run
            feats['block'] = block
            feats['split'] = split
            windows.append(feats)
        emg_names = emg['names']

    summarize(pid, events, shapes, rest_min, test_runs)
    plot_qc(pid, qc, emg_names, mrcp_names, out_dir)
    if check_only:
        return None
    return cross_validate(pid, pd.concat(windows, ignore_index=True), test_runs)


def summarize(pid, events, shapes, rest_min, test_runs):
    rt   = events['rt']
    runs = sorted(int(r) for r in events['run'].unique())
    print(f'P{pid}: {len(events)} trials, series {runs}')
    print(f'  RT median {np.nanmedian(rt):.3f} s (IQR {np.nanpercentile(rt, 25):.3f}-'
          f'{np.nanpercentile(rt, 75):.3f}), RT < {PRE_ONSET} s: {np.mean(rt < PRE_ONSET):.0%}')
    print(f'  false starts (RT < {MIN_RT} s) excluded: {int(np.sum(rt < MIN_RT))}')
    print(f'  test series {test_runs}')
    print(f'  rest: dev {rest_min["dev"]:.1f} min (weight {rest_min["dev_weight"]:.1f}, '
          f'surface/mixed {rest_min["dev_long"]:.1f}), test {rest_min["test"]:.1f} min')
    for run, desc in shapes.items():
        print(f'  S{run}: {desc}')


if __name__ == '__main__':
    main()
