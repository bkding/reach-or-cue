import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Feature sets: key, label, color, marker, x offset.
SERIES = (('EMG', 'EMG-only (baseline)', '#2a78d6', 'o', -0.2),
          ('EEG', 'EEG-only', '#eb6834', 's', 0.0),
          ('EEG+EMG', 'Fusion', '#1baf7a', '^', 0.2))

# Colors.
INK     = '#0b0b0b'
INK_2   = '#52514e'
GRID    = '#e4e3df'
SURFACE = '#fcfcfb'


def main():
    err = pd.read_csv(sys.argv[1])
    out = Path(sys.argv[2])
    groups = ['1', '2', '3', 'all']
    labels = ['P1', 'P2', 'P3', 'Mean of P1–P3']
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    # Means with SE or bootstrap intervals.
    for key, name, color, marker, dx in SERIES:
        sub = err[err['quantity'] == key].set_index('participant')
        for i, g in enumerate(groups):
            row = sub.loc[g]
            if g == 'all':
                lo, hi = row['mean'] - row['ci_low'], row['ci_high'] - row['mean']
            else:
                lo = hi = row['se']
            ax.errorbar(i + dx, row['mean'], yerr=[[lo], [hi]], fmt=marker, ms=7, color=color,
                        ecolor=color, elinewidth=1.6, capsize=3, zorder=3,
                        label=name if i == 0 else None)
            if g == 'all':
                ax.text(i + dx, row['ci_low'] - 0.012, f'{row["mean"]:.3f}', color=INK_2,
                        fontsize=8, va='top', ha='center')

    # Axes.
    ax.axvline(2.5, color=GRID, lw=1)
    ax.axhline(0.5, color=INK_2, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.text(-0.45, 0.505, 'chance', color=INK_2, fontsize=8, va='bottom')
    ax.set_xticks(range(len(groups)), labels, color=INK)
    ax.set_xlim(-0.5, len(groups) - 0.3)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel('Window AUC (0.5 s pre-onset vs matched rest)', color=INK_2, fontsize=9)
    ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)

    # Legend and note.
    ax.legend(loc='upper left', frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3)
    fig.text(0.5, -0.03, 'Nested leave-one-series-out CV on 7 dev series per participant. '
             'P1–P3: mean ± SE over folds; mean: two-level bootstrap 95% CI (2,000 resamples).',
             ha='center', color=INK_2, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=SURFACE)


if __name__ == '__main__':
    main()
