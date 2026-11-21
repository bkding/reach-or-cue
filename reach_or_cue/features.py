import numpy as np
import pandas as pd
import scipy.signal

from .config import EPS

# EMG envelope.
F_EMG_HIGHPASS = 20.0       # Hz
F_EMG_ENVELOPE = 5.0        # Hz
EMG_LAGS       = (0.0, 0.1) # s

# Movement-related cortical potential.
F_MRCP        = (0.05, 5.0)               # Hz
MRCP_LAGS     = (0.0, 0.1, 0.2, 0.3, 0.4) # s
MRCP_CHANNELS = ('FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2')

# Event-related desynchronization.
F_ERD        = {'mu': (8.0, 12.0), 'beta': (13.0, 30.0)} # Hz
F_ERD_SMOOTH = 2.0                                       # Hz
ERD_CHANNELS = ('C3', 'Cz', 'C4')


def emg_envelope(emg, fs):
    sos_hp = scipy.signal.butter(4, F_EMG_HIGHPASS, btype='highpass', fs=fs, output='sos')
    sos_lp = scipy.signal.butter(2, F_EMG_ENVELOPE, btype='lowpass', fs=fs, output='sos')
    return np.clip(_causal_filter(sos_lp, np.abs(_causal_filter(sos_hp, emg))), EPS, None)


def eeg_mrcp(eeg, fs, names):
    eeg_car = eeg - eeg.mean(axis=1, keepdims=True)
    sos = scipy.signal.butter(2, F_MRCP, btype='bandpass', fs=fs, output='sos')
    idx = _channel_index(names, MRCP_CHANNELS)
    return _causal_filter(sos, eeg_car[:, idx]), [names[i] for i in idx], eeg_car


def eeg_band_power(eeg_car, fs, names):
    idx = _channel_index(names, ERD_CHANNELS)
    sos_smooth = scipy.signal.butter(2, F_ERD_SMOOTH, btype='lowpass', fs=fs, output='sos')
    powers, cols = [], []
    for band, (lo, hi) in F_ERD.items():
        sos_band = scipy.signal.butter(4, (lo, hi), btype='bandpass', fs=fs, output='sos')
        x_band = _causal_filter(sos_band, eeg_car[:, idx])
        powers.append(np.log(np.clip(_causal_filter(sos_smooth, x_band ** 2), EPS, None)))
        cols += [f'eeg_{band}_{names[i]}' for i in idx]
    return np.hstack(powers), cols


def series_features(series, t_dec):
    eeg, emg = series['eeg'], series['emg']
    mrcp, mrcp_names, eeg_car = eeg_mrcp(eeg['sig'], eeg['fs'], eeg['names'])
    power, power_cols = eeg_band_power(eeg_car, eeg['fs'], eeg['names'])
    env_log = np.log(emg_envelope(emg['sig'], emg['fs']))

    # MRCP amplitude at lags.
    blocks, cols = [], []
    for lag in MRCP_LAGS:
        blocks.append(sample_at(mrcp, eeg['fs'], t_dec - lag))
        cols += [f'eeg_mrcp_{n}_lag{int(lag * 1000)}' for n in mrcp_names]

    # ERD band power.
    blocks.append(sample_at(power, eeg['fs'], t_dec))
    cols += power_cols

    # EMG envelope at lags.
    for lag in EMG_LAGS:
        blocks.append(sample_at(env_log, emg['fs'], t_dec - lag))
        cols += [f'emg_{n}_lag{int(lag * 1000)}' for n in emg['names']]
    return pd.DataFrame(np.hstack(blocks), columns=cols)


def sample_at(x, fs, t):
    idx = np.clip(np.round(t * fs).astype(int), 0, len(x) - 1)
    return x[idx]


def _causal_filter(sos, x):
    zi = scipy.signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    y, _ = scipy.signal.sosfilt(sos, x, axis=0, zi=zi)
    return y


def _channel_index(names, wanted):
    lookup  = {n.lower(): i for i, n in enumerate(names)}
    missing = [w for w in wanted if w.lower() not in lookup]
    if missing:
        print(f'    missing channels: {missing}')
    return [lookup[w.lower()] for w in wanted if w.lower() in lookup]
