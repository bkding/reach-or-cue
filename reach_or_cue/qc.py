import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .config import PRE_ONSET, EPS
from .features import sample_at

# Epoch grid around hand start.
QC_WINDOW = (-1.5, 1.0) # s
FS_QC     = 500.0       # Hz


def epoch_mean(x, fs, onsets):
    t_rel  = np.arange(QC_WINDOW[0], QC_WINDOW[1], 1.0 / FS_QC)
    epochs = [sample_at(x, fs, onset + t_rel) for onset in onsets
              if onset + QC_WINDOW[0] >= 0 and (onset + QC_WINDOW[1]) * fs < len(x)]
    return t_rel, np.nanmean(np.stack(epochs), axis=0) if epochs else None


def hand_speed(kin):
    # Position channels per Polhemus sensor.
    sensors = {}
    for i, n in enumerate(kin['names']):
        m = re.match(r'P([xyz])(\d+)\b', n)
        if m:
            sensors.setdefault(m.group(2), {})[m.group(1)] = i

    # Speed per complete sensor.
    speeds = {}
    for sid, axes in sensors.items():
        if len(axes) == 3:
            pos = kin['sig'][:, [axes['x'], axes['y'], axes['z']]]
            speeds[f'P{sid}'] = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * kin['fs']
    return speeds


def plot_qc(pid, qc, emg_names, mrcp_names, out_dir):
    t_rel = np.arange(QC_WINDOW[0], QC_WINDOW[1], 1.0 / FS_QC)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # Hand speed and LED onsets.
    for name, curves in qc['speed'].items():
        curves = [c for c in curves if c is not None]
        if curves:
            axes[0].plot(t_rel, np.nanmean(curves, axis=0).ravel(), label=name)
    axes[0].set_ylabel('speed (kin units/s)')
    axes[0].legend(fontsize=8)
    ax_led = axes[0].twinx()
    ax_led.hist(qc['led_rel'], bins=30, range=QC_WINDOW, alpha=0.3, color='gray')
    ax_led.set_ylabel('LED on count')

    # EMG log envelope, z-scored to the first 0.5 s.
    emg_avg = np.nanmean([c for c in qc['emg'] if c is not None], axis=0)
    emg_z   = (emg_avg - emg_avg[:250].mean(axis=0)) / (emg_avg[:250].std(axis=0) + EPS)
    for i, n in enumerate(emg_names):
        axes[1].plot(t_rel, emg_z[:, i], label=n)
    axes[1].set_ylabel('EMG log env (z)')
    axes[1].legend(fontsize=8)

    # MRCP on C3, Cz, C4.
    mrcp_avg = np.nanmean([c for c in qc['mrcp'] if c is not None], axis=0)
    for i, n in enumerate(mrcp_names):
        if n in ('C3', 'Cz', 'C4'):
            axes[2].plot(t_rel, mrcp_avg[:, i], label=n)
    axes[2].set_ylabel('MRCP (causal, CAR)')
    axes[2].set_xlabel('time from hand start (s)')
    axes[2].legend(fontsize=8)

    # Hand start and pre-onset window.
    for ax in axes:
        ax.axvline(0, color='k', lw=0.8)
        ax.axvline(-PRE_ONSET, color='k', lw=0.8, ls='--')
    fig.suptitle(f'P{pid} QC: averages aligned to hand start')
    fig.tight_layout()
    fig.savefig(out_dir / f'qc_P{pid}.png', dpi=120)
    plt.close(fig)
