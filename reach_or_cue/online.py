import itertools

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from .config import (MIN_RT, PRE_CUE, REFRACTORY, SEARCH, SIM_CUE_LAG, SIM_DETECT, SIM_JITTER,
                     SIM_MOV_LAG, STEP, STOP_GUARD, TARGET_FP, WINDOWS)

# Timer delay grid.
D_GRID = np.round(np.arange(0.0, 1.5001, 0.005), 4) # s


def detect_events(t, score, thr):
    above = score >= thr
    edges = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    return refractory(t[edges])


def refractory(times):
    keep, last = [], -np.inf
    for x in times:
        if x - last >= REFRACTORY:
            keep.append(x)
            last = x
    return np.array(keep)


def random_events(rng, rate, t0, t1):
    n = rng.poisson(rate * (t1 - t0) / 60)
    return refractory(np.sort(rng.uniform(t0, t1, n)))


def prep_windows(ev, t, pre, post):
    valid = ev[ev['hand_start'].notna()].sort_values('hand_start')
    use   = valid[valid['rt'] >= MIN_RT]

    # Merged motion masks.
    lo = (valid['hand_start'] - pre).to_numpy()
    hi = (valid['hand_stop'] + STOP_GUARD).to_numpy()
    m_lo, m_hi = [], []
    for a, b in zip(lo, hi):
        if m_lo and a <= m_hi[-1]:
            m_hi[-1] = max(m_hi[-1], b)
        else:
            m_lo.append(a)
            m_hi.append(b)
    m_lo, m_hi = np.array(m_lo), np.array(m_hi)
    hs = use['hand_start'].to_numpy()
    return {'m_lo': m_lo, 'm_hi': m_hi, 'hs': hs, 'w_lo': hs - pre, 'w_hi': hs + post,
            'free_min': (~_inside(t, m_lo, m_hi)).sum() * STEP / 60}


def score_events(trig, p):
    trig = np.asarray(trig, float)

    # First trigger inside each hit window.
    i    = np.searchsorted(trig, p['w_lo'], side='left')
    ok   = i < len(trig)
    hitv = np.zeros(len(p['hs']), bool)
    hitv[ok] = trig[i[ok]] <= p['w_hi'][ok]
    leads = p['hs'][hitv] - trig[i[hitv]]

    # Triggers in masks, hit windows and free time.
    in_mask = _inside(trig, p['m_lo'], p['m_hi'])
    in_hit  = _inside(trig, p['w_lo'], p['w_hi'])
    return {'n_trials': len(p['hs']), 'n_hit': int(hitv.sum()), 'leads': leads,
            'n_fp': int((~in_mask).sum()), 'n_motion': int((in_mask & ~in_hit).sum()),
            'free_min': p['free_min']}


def first_triggers(pid, run, fset, trig, ev):
    trig = np.asarray(trig, float)
    e    = ev[ev['hand_start'].notna() & (ev['rt'] >= MIN_RT)]
    led, hs, rt = e['led_on'].to_numpy(), e['hand_start'].to_numpy(), e['rt'].to_numpy()
    pre, post   = WINDOWS['B']

    # Pre-cue triggers and first trigger after LED.
    i_pre   = np.searchsorted(trig, led - PRE_CUE, side='left')
    pre_cue = (i_pre < len(trig)) & (trig[np.clip(i_pre, 0, max(len(trig) - 1, 0))] < led) \
              if len(trig) else np.zeros(len(led), bool)
    i_led   = np.searchsorted(trig, led, side='left')
    has     = (i_led < len(trig))
    first   = np.full(len(led), np.inf)
    first[has] = trig[i_led[has]] - led[has]
    cens    = first > SEARCH

    # First trigger inside window B.
    i_b   = np.searchsorted(trig, hs - pre, side='left')
    has_b = i_b < len(trig)
    hit_t = np.full(len(led), np.nan)
    hit_t[has_b] = trig[i_b[has_b]]
    hit_b = has_b & (hit_t <= hs + post)
    return [{'pid': pid, 'run': run, 'features': fset, 'rt': rt[k], 'pre_cue': bool(pre_cue[k]),
             'censored': bool(cens[k]), 'y': SEARCH if cens[k] else first[k],
             'hit_B': bool(hit_b[k]), 'hit_time_rel': hit_t[k] - led[k] if hit_b[k] else np.nan}
            for k in range(len(led))]


def tobit(x, y, cens):
    if len(x) < 5 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    obs = ~cens
    if obs.sum() >= 3:
        b0, a0 = np.polyfit(x[obs], y[obs], 1)
        s0 = max(np.std(y[obs] - (a0 + b0 * x[obs])), 1e-2)
    else:
        a0, b0, s0 = SEARCH, 0.0, 0.5

    def nll(p):
        a, b, ls = p
        s  = np.exp(ls)
        mu = a + b * x
        return -(norm.logpdf(y[obs], mu[obs], s).sum() + norm.logsf(SEARCH, mu[~obs], s).sum())

    res = minimize(nll, [a0, b0, np.log(s0)], method='L-BFGS-B')
    return res.x[0], res.x[1], np.exp(res.x[2])


def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.polyfit(x, y, 1)[0] if len(x) >= 5 and np.ptp(x) > 0 else np.nan


def timer_delay(train, pre, post):
    tr   = train[train['hand_start'].notna() & (train['rt'] >= MIN_RT)]
    lead = tr['rt'].to_numpy()[:, None] - D_GRID
    hit  = (lead >= -post) & (lead <= pre)
    det  = hit.mean(axis=0)
    best = np.flatnonzero(det == det.max())
    mean_lead = np.where(hit, lead, 0).mean(axis=0)
    return D_GRID[best[np.argmax(mean_lead[best])]]


def simulate(rng, kind, ev, t):
    bg = random_events(rng, TARGET_FP, t[0], t[-1])
    e  = ev[ev['hand_start'].notna()]
    if kind == 'cue-locked':
        true = e['led_on'].to_numpy() + SIM_CUE_LAG + rng.normal(0, SIM_JITTER, len(e))
    elif kind == 'movement-locked':
        true = e['hand_start'].to_numpy() - SIM_MOV_LAG + rng.normal(0, SIM_JITTER, len(e))
    else:
        true = np.array([])
    true = true[rng.random(len(true)) < SIM_DETECT]
    return refractory(np.sort(np.concatenate([bg, true])))


def sign_flip_p(d):
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan
    obs   = abs(d.mean())
    signs = np.array(list(itertools.product((-1, 1), repeat=len(d))))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12))


def _inside(x, lo, hi):
    if len(lo) == 0 or len(x) == 0:
        return np.zeros(len(x), bool)
    j = np.searchsorted(lo, x, side='right') - 1
    return (j >= 0) & (x < hi[np.clip(j, 0, None)])
