import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Feature sets.
FEATURES = ('EMG', 'EEG', 'EEG+EMG')
LABELS   = {'EMG': 'EMG', 'EEG': 'EEG', 'EEG+EMG': 'Fusion'}

# Series types: label, color, marker, x offset.
GROUPS = {'weight': ('Weight series (1–3 s gaps)', '#2a78d6', 'o', -0.12),
          'surface/mixed': ('Surface / mixed series (~10 s gaps)', '#eb6834', 's', 0.12)}

# Rest definitions, one per row.
ROWS = (('all', 'All inter-trial rest'), ('matched', 'Matched rest: 1 s before the LED'))

# Colors.
INK     = '#0b0b0b'
INK_2   = '#52514e'
GRID    = '#e4e3df'
SURFACE = '#fcfcfb'


def main():
    folds = {'all': pd.read_csv(sys.argv[1]), 'matched': pd.read_csv(sys.argv[2])}
    out   = Path(sys.argv[3])
    pids  = sorted(folds['matched']['participant'].unique())
    fig, axes = plt.subplots(len(ROWS), len(pids), figsize=(10, 7), sharey=True,
                             facecolor=SURFACE)
    for r, (key, row_title) in enumerate(ROWS):
        for c, pid in enumerate(pids):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            sub = folds[key][folds[key]['participant'] == pid]

            # Fold AUCs and group means.
            for stype, (name, color, marker, dx) in GROUPS.items():
                g = sub[sub['series_type'] == stype]
                for i, fset in enumerate(FEATURES):
                    y = g.loc[g['features'] == fset, 'auc'].to_numpy()
                    ax.scatter(np.full(len(y), i + dx), y, s=40, color=color, marker=marker,
                               edgecolor='none', alpha=0.85, zorder=3,
                               label=name if (r, c, i) == (0, 0, 0) else None)
                    ax.plot([i + dx - 0.1, i + dx + 0.1], [y.mean()] * 2, color=color, lw=2,
                            zorder=2)

            # Reference line and axes.
            ax.axhline(0.5, color=INK_2, lw=1, ls=(0, (3, 3)), zorder=1)
            ax.set_xticks(range(len(FEATURES)), [LABELS[f] for f in FEATURES], color=INK)
            ax.set_xlim(-0.5, len(FEATURES) - 0.5)
            ax.set_ylim(0.38, 1.0)
            ax.set_title(f'P{pid} · {row_title}', color=INK, fontsize=10, loc='left')
            ax.grid(axis='y', color=GRID, lw=0.8, zorder=0)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, length=0)
        axes[r, 0].set_ylabel('Window AUC', color=INK_2)

    # Legend and note.
    axes[0, 0].text(2.45, 0.505, 'chance', color=INK_2, fontsize=8, ha='right', va='bottom')
    fig.legend(loc='upper center', ncol=2, frameon=False, fontsize=9, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, -0.01, 'Each dot is one held-out dev series (leave-one-series-out); '
             'bar = group mean. Positive class: 0.5 s before hand start.', ha='center',
             color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=SURFACE)


if __name__ == '__main__':
    main()
