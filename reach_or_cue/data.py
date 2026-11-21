import numpy as np
import pandas as pd
import scipy.io

from .config import STEP, PRE_ONSET, MIN_RT, WEIGHT_BLOCK

# Rest windows.
WARMUP                  = 10.0 # s, filter warm-up at series start
REST_AFTER_STOP         = 1.0  # s, all inter-trial rest
REST_PRE_CUE            = 1.0  # s, matched rest
REST_AFTER_STOP_MATCHED = 0.5  # s, matched rest


def load_mat(path):
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        import mat73
        return mat73.loadmat(path)


def load_lifts(pdir, pid):
    p     = _get(load_mat(pdir / f'P{pid}_AllLifts.mat'), 'P')
    col   = {str(c): i for i, c in enumerate(np.atleast_1d(_get(p, 'ColNames')))}
    lifts = np.asarray(_get(p, 'AllLifts'), dtype=float)

    # Event times on the series clock.
    t0 = lifts[:, col['StartTime']] - lifts[:, col['LEDOn']]
    return (pd.DataFrame({
                'run':        lifts[:, col['Run']].astype(int),
                'led_on':     t0 + lifts[:, col['LEDOn']],
                'hand_start': t0 + lifts[:, col['tHandStart']],
                'hand_stop':  t0 + lifts[:, col['tHandStop']],
                'block':      lifts[:, col['BlockType']].astype(int),
            })
            .assign(rt=lambda d: d['hand_start'] - d['led_on'])
            .sort_values(['run', 'led_on'])
            .reset_index(drop=True))


def load_series(pdir, pid, run):
    hs  = _get(load_mat(pdir / f'HS_P{pid}_S{run}.mat'), 'hs')
    out = {}
    for name in ('eeg', 'emg', 'kin'):
        block = _get(hs, name)
        if block is None:
            continue
        out[name] = {
            'sig':   np.asarray(_get(block, 'sig'), dtype=float),
            'fs':    float(_get(block, 'samplingrate', np.nan)),
            'names': _as_names(block),
        }
    return out


def label_times(duration, events, rest_mode):
    t_dec = np.arange(WARMUP, duration - STEP, STEP)
    label = np.full(len(t_dec), -1)
    valid = events['rt'] >= MIN_RT

    # Pre-onset windows.
    for hs in events.loc[valid, 'hand_start'].dropna():
        label[(t_dec >= hs - PRE_ONSET) & (t_dec < hs)] = 1

    # Rest windows.
    prev_stop = None
    for ok, (_, ev) in zip(valid, events.iterrows()):
        if rest_mode == 'matched':
            guard = WARMUP if prev_stop is None else prev_stop + REST_AFTER_STOP_MATCHED
            start = max(ev['led_on'] - REST_PRE_CUE, guard)
        else:
            start = WARMUP if prev_stop is None else prev_stop + REST_AFTER_STOP
        end = min(ev['led_on'], ev['hand_start'] - PRE_ONSET)
        if ok and np.isfinite(start) and np.isfinite(end) and end > start:
            label[(t_dec >= start) & (t_dec < end) & (label == -1)] = 0
        prev_stop = ev['hand_stop']
    return t_dec, label


def select_test_runs(events):
    blocks = events.groupby('run')['block'].first()
    return [int(blocks[blocks == WEIGHT_BLOCK].index.max()),
            int(blocks[blocks != WEIGHT_BLOCK].index.max())]


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_names(block):
    return [str(n) for n in np.atleast_1d(_get(block, 'names', []))]
