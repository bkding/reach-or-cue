# Reach or Cue?

*Causal EEG / EMG reach-onset detection*

### ESE 5420 - Statistics for Data Science &nbsp;·&nbsp; University of Pennsylvania

**Bokang Ding** &nbsp;·&nbsp; Team Dyn (Solo)

Does EEG detect the reach, or the cue? This project asks whether EEG lets a detector recognise an upcoming reach earlier than EMG alone when every detector is strictly causal and held to the same false-alarm rate, and whether any extra lead follows the movement or the visual go cue. It uses the public WAY-EEG-GAL grasp-and-lift dataset, with hand start from 3-D hand kinematics as an answer key independent of both signals.

> **Status: in progress.** The numbers below are preliminary (3 of 12 participants, development data only). The pseudo-online detector, the matched-false-alarm comparison and the cue/movement slope test are still to come.

<p align="center">
  <img src="assets/baseline.png" width="100%">
  <br>
  <b>Fig. 1.</b> Window AUC (0.5 s before hand start vs. matched rest) on held-out development series. P1–P3: mean ± SE over 7 folds; mean: two-level bootstrap 95% CI.
</p>

## Results (preliminary)

Nested leave-one-series-out cross-validation on 7 development series per participant.

| | EMG-only (baseline) | EEG-only | Fusion | Fusion − EMG |
| :-- | :-: | :-: | :-: | :-: |
| **P1** | 0.615 ± 0.005 | 0.584 ± 0.009 | 0.625 ± 0.007 | +0.011 |
| **P2** | 0.859 ± 0.037 | 0.729 ± 0.039 | 0.888 ± 0.025 | +0.028 |
| **P3** | 0.633 ± 0.010 | 0.649 ± 0.023 | 0.712 ± 0.020 | +0.078 |
| **Mean (95% CI)** | 0.702 [0.615, 0.849] | 0.654 [0.586, 0.730] | 0.742 [0.628, 0.883] | +0.039 [0.009, 0.075] |

The choice of rest windows matters. When rest is taken from all inter-trial time, most of it comes from the long waits of the surface and mixed series, and EEG partly learns "long wait vs. near a reach" rather than movement preparation. Rest is therefore taken from the 1 s before each LED onset in every series.

## Pipeline

The data come from [WAY-EEG-GAL](https://doi.org/10.6084/m9.figshare.988376) (Luciw et al., 2014): 12 participants, 3,936 grasp-and-lift trials, `32`-channel EEG and hand kinematics at `500 Hz`, and `5`-channel EMG at `4 kHz`.

- **Alignment** ([`data.py`](reach_or_cue/data.py)). Event times are relative to a window that starts `2 s` before LED onset, so series time = `StartTime − LEDOn + event`, checked against the recorded LED channel (within `2 ms`).
- **Labels** ([`data.py`](reach_or_cue/data.py)). A decision every `50 ms`. Positive: the `0.5 s` before hand start. Rest: the `1 s` before each LED onset, at least `0.5 s` after the previous hand stop. Lifts with reaction time below `0.1 s` are excluded.
- **Features** ([`features.py`](reach_or_cue/features.py)). All filters are causal. EMG: `20 Hz` high-pass, rectification, `5 Hz` envelope, log, at lags `0` and `100 ms`. EEG: common average reference, `0.05–5 Hz` slow potentials on seven central channels at lags `0–400 ms`, and μ/β log power on C3, Cz, C4. Fusion concatenates both.
- **Model** ([`evaluation.py`](reach_or_cue/evaluation.py)). Standardization and L2 logistic regression with balanced class weights; `C` chosen by inner leave-one-series-out CV within each training fold.
- **Split** ([`data.py`](reach_or_cue/data.py)). Per participant, the last weight series and the last surface/mixed series are held out as the test set and used once. The other 7 series are cross-validated one series at a time.
- **QC** ([`qc.py`](reach_or_cue/qc.py)). Hand speed, EMG envelopes and slow potentials averaged around hand start, one figure per participant.

## Quickstart

Requires **Python 3.12+**:

```bash
git clone https://github.com/bkding/reach-or-cue.git
cd reach-or-cue
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Download the participant archives (`P1.zip`, ...) from [figshare](https://doi.org/10.6084/m9.figshare.988376) and extract `HS_P#_S1.mat` to `HS_P#_S9.mat` and `P#_AllLifts.mat` into `data/P#/`.

**Data check** (alignment, reaction times, rest minutes, QC figures):

```bash
python scripts/run_baseline.py data --participants 1,2,3 --check-only --out results/check
```

**Baseline** (nested CV for EMG, EEG and fusion; about 1.5 min for three participants on a CPU):

```bash
python scripts/run_baseline.py data --participants 1,2,3 --out results/baseline
python scripts/plot_baseline.py results/baseline/baseline_auc_errorbars.csv results/baseline/baseline.png
```

Add `--rest all` to use all inter-trial time as rest instead of matched rest.

## Roadmap

- [x] Alignment, causal features, matched rest, nested-CV baseline (P1–P3)
- [ ] All 12 participants
- [ ] Pseudo-online detector: lead time and true-positive rate at 0.5, 1 and 2 false alarms per minute
- [ ] Slope test: do detections follow the hand or the LED?
- [ ] Test series, scored once, with participant-level bootstrap intervals

## Acknowledgements

Data: M. D. Luciw, E. Jarocka and B. B. Edin, "Multi-channel EEG recordings during 3,936 grasp and lift trials with varying weight and friction," *Scientific Data* 1, 140047 (2014), CC BY 4.0.
