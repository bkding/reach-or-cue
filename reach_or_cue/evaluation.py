import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import WEIGHT_BLOCK, FEATURE_SETS

# Regularization grid.
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)

# Bootstrap.
N_BOOT = 2000
SEED   = 0


def make_model(c):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, class_weight='balanced', max_iter=2000),
    )


def select_c(train, cols):
    # Mean inner leave-one-series-out AUC per C.
    inner = []
    for c in C_GRID:
        aucs = []
        for held in sorted(train['run'].unique()):
            fit, val = train[train['run'] != held], train[train['run'] == held]
            if val['label'].nunique() < 2:
                continue
            model = make_model(c).fit(fit[cols], fit['label'])
            aucs.append(roc_auc_score(val['label'], model.predict_proba(val[cols])[:, 1]))
        inner.append(np.mean(aucs))
    return C_GRID[int(np.argmax(inner))], float(np.max(inner))


def cross_validate(pid, windows, test_runs):
    # Feature columns per set.
    dev  = windows[windows['split'] == 'dev']
    cols = {
        'EMG': [c for c in dev.columns if c.startswith('emg_')],
        'EEG': [c for c in dev.columns if c.startswith('eeg_')],
    }
    cols['EEG+EMG'] = cols['EEG'] + cols['EMG']

    # Outer leave-one-series-out folds.
    rows = []
    for held in sorted(dev['run'].unique()):
        train, val = dev[dev['run'] != held], dev[dev['run'] == held]
        if val['label'].nunique() < 2:
            continue
        for fset in FEATURE_SETS:
            c_best, inner_auc = select_c(train, cols[fset])
            model = make_model(c_best).fit(train[cols[fset]], train['label'])
            score = model.predict_proba(val[cols[fset]])[:, 1]
            rows.append({
                'participant': pid, 'held_series': held, 'features': fset,
                'series_type': ('weight' if val['block'].iloc[0] == WEIGHT_BLOCK
                                else 'surface/mixed'),
                'auc': roc_auc_score(val['label'], score),
                'c_chosen': c_best, 'inner_auc': inner_auc,
                'n_pos': int(val['label'].sum()), 'n_rest': int((val['label'] == 0).sum()),
            })
    print(f'  test series {test_runs} untouched')
    return pd.DataFrame(rows)


def two_level_bootstrap(values, rng):
    pids  = list(values)
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        picked   = rng.choice(pids, size=len(pids), replace=True)
        stats[b] = np.mean([rng.choice(values[p], size=len(values[p]), replace=True).mean()
                            for p in picked])
    return np.percentile(stats, [2.5, 97.5])


def report(folds, out_dir):
    # Fold table and per-participant summary.
    folds.to_csv(out_dir / 'baseline_auc_folds.csv', index=False)
    table = folds.groupby(['participant', 'features'])['auc'].agg(['mean', 'std', 'count'])
    table = table.unstack('features')
    table.to_csv(out_dir / 'baseline_auc_summary.csv')
    print('\nWindow-level AUC, leave-one-series-out on dev series (mean / sd over folds)')
    print(table.round(3).to_string())

    # Fusion minus EMG-only.
    subj = folds.groupby(['participant', 'features'])['auc'].mean().unstack('features')
    diff = subj['EEG+EMG'] - subj['EMG']
    print('\nFusion minus EMG-only, per participant:')
    print(diff.round(3).to_string())
    print(f'Across participants: mean {diff.mean():.3f}, min {diff.min():.3f}, '
          f'max {diff.max():.3f}')

    # Standard errors and two-level bootstrap intervals.
    rng  = np.random.default_rng(SEED)
    wide = folds.pivot_table(index=['participant', 'held_series'], columns='features',
                             values='auc')
    wide['diff'] = wide['EEG+EMG'] - wide['EMG']
    err_rows = []
    for col in list(FEATURE_SETS) + ['diff']:
        values = {pid: g[col].to_numpy() for pid, g in wide.groupby(level='participant')}
        lo, hi = two_level_bootstrap(values, rng)
        for pid, v in values.items():
            err_rows.append({'quantity': col, 'participant': pid, 'mean': v.mean(),
                             'se': v.std(ddof=1) / np.sqrt(len(v)), 'n_folds': len(v)})
        err_rows.append({'quantity': col, 'participant': 'all',
                         'mean': np.mean([v.mean() for v in values.values()]),
                         'ci_low': lo, 'ci_high': hi})
    err = pd.DataFrame(err_rows)
    err.to_csv(out_dir / 'baseline_auc_errorbars.csv', index=False)
    print(f"\nMean +/- SE over folds; 'all' = participant mean with two-level bootstrap 95% CI "
          f'({N_BOOT} resamples, seed {SEED})')
    print(err.round(3).to_string(index=False))
    print('\nC chosen by inner CV (counts over outer folds)')
    print(folds.groupby(['features', 'c_chosen']).size().unstack(fill_value=0).to_string())

    # AUC by held-out series type.
    by_type = (folds.groupby(['series_type', 'participant', 'features'])['auc'].mean()
               .unstack('features'))
    by_type.to_csv(out_dir / 'baseline_auc_by_series_type.csv')
    print('\nAUC by held-out series type (mean over folds)')
    print(by_type.round(3).to_string())

    # Per-participant AUC figure.
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(FEATURE_SETS))
    for pid, row in subj.iterrows():
        ax.plot(x, row[list(FEATURE_SETS)], marker='o', alpha=0.7, label=f'P{pid}')
    ax.axhline(0.5, color='gray', ls='--', lw=0.8)
    ax.set_xticks(x, FEATURE_SETS)
    ax.set_ylabel('window AUC (pre-onset 0.5 s vs rest)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / 'baseline_auc_by_participant.png', dpi=120)
    plt.close(fig)
