# Step 4: QGAN Model Pass — Changes and Ablation Plan

All changes are flag-controlled; v1/log_minmax reproduces impg@735bdfc exactly.

## Circuit (circuit.py, `--circuit v1|v2`)

| issue in v1 (verified) | fix in v2 | evidence |
|---|---|---|
| 14 dead params: final-layer RZ on numeric wires commutes with PauliZ measurement | per-layer gate order RZ→RY (non-diagonal gate always last) | gradient check: v1 = 14/96 dead, v2 = 0/144 dead |
| entanglers parameter-free (CZ/CNOT): correlations not learnable | CZ ring → CRY ring with per-pair angles; cross-register CNOTs → CRYs | by construction; corr_dist in validation metric tracks it |
| discrete-wire phase `RZ(inputs·π)`, inputs already in [0,π] → range π² > 2π, phase wrap | `RZ(inputs)` | arithmetic |

v2 weights: (3, 3, 16) = 144 params (v1: 96, of which 82 effective).

## Preprocessing (preprocessing.py, `--preproc log_minmax|quantile`)

`quantile`: per-feature QuantileTransformer to uniform·π. The smooth
expectation-value generator only has to produce roughly-uniform marginals;
the inverse recovers heavy tails and multimodality by construction.
**Evaluation consequence**: marginal metrics (JS, per-feature W1) become
trivially good under `quantile` — quality claims for such runs must rest on
joint structure (correlation distance, C2ST) and downstream utility.

## Training (train.py)

- 15% per-class GAN-validation holdout (never seen by the critic).
- Every 5 epochs: critic-independent score = standardized per-feature
  Wasserstein-1 + mean |Δcorrelation matrix|; **best checkpoint kept**
  (weights_best.pth) alongside weights_last.pth. Fixed-last-epoch saving
  is gone.
- training_history.csv: epoch-mean D/G losses + validation scores.
- model_manifest.json: circuit version, preproc, seed, best epoch/score,
  feature order.

## Generation (generate.py)

- Manifest-driven: rebuilds the exact trained circuit and inverse transform;
  train/generate divergence is structurally impossible.
- Loads validation-best checkpoint.
- Postprocessing ON by default (`--no-postprocess` for ablation):
  clip ≥ 0; round `syn_flag_cnt`/`fin_flag_cnt` to integers within training
  support; repair ordering chains (fwd min≤mean≤max, pkt min≤mean,
  bwd min≤seg_avg) by row-wise sorting.

## Deliberately unchanged

`N_CRITIC=2` (WGAN-GP convention is critic:gen 5:1 — worth a later
experiment, config knob exists), `LR_GENERATOR=0.015`, `TOTAL_EPOCHS=50`,
noise prior Uniform[0,π]^16. One variable at a time.

## Ablation grid for the rerun

Minimum: {v1, v2} × {log_minmax, quantile} on 2–3 representative classes
(e.g. 4 = smallest, 8 = mid, 1 = largest minority), 3 seeds each; compare
validation score curves and downstream macro-recall contribution. If v2 +
quantile dominates, run all 7 minority classes with it for the main results;
report the grid in the appendix.

## Known remaining weaknesses (documented, not hidden)

- Expectation-value outputs remain deterministic functions of the latent;
  stochasticity is bounded by the noise pushforward. Shot-based sampling or
  a Born-machine head is the next architectural step if joint-structure
  metrics stay poor.
- Validation score weights W1 and corr_dist equally (both dimensionless but
  not calibrated against each other); checkpoint selection is robust to
  this, exact score values are not comparable across preprocs.
- Smallest classes (~500 training rows) give the CRY-ring 144-param model
  little signal; watch the val curves for overfitting to the critic.
