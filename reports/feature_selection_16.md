# 16-Dimensional Feature Set: Selection Analysis

Date: 2026-07-04 · Data: deduped, session-split training set (25,185 rows, 62 candidate features) · Decision recorded in `src/config.py` (`NUMERIC_FEATURES` + `DISCRETE_FEATURES`)

## Method

1. **Redundancy pruning**: greedy keep-best by preliminary RF rank, dropping any
   feature with |Spearman ρ| > 0.90 against an already-kept one. Spearman (not
   Pearson) so monotone nonlinear pairs are caught — this is what finally
   removes the `pkt_len_var` / `pkt_len_std` duplication (ρ = 1.0).
   28 of 62 features pruned, 34 survive.
2. **Ranking**: permutation importance (scoring = macro-F1, RF, 2 repeats)
   aggregated over **GroupKFold(5) by `session_id`** on the training split
   only. Grouped validation means a feature scores highly only if it helps
   classify flows from *unseen sessions* — session-fingerprint features are
   structurally penalized.
3. **Assembly under the architecture constraint** (14 continuous + 2 discrete
   wires): continuous register = ranks 1–14; discrete register = the two
   best *genuinely* discrete features (`syn_flag_cnt`, 8 distinct values;
   `fin_flag_cnt`, 12 distinct values).

## Selected set

Continuous (wires 0–13, in wire order):
`bwd_seg_size_avg, pkt_len_min, fwd_pkt_len_min, totlen_bwd_pkts,
fwd_pkt_len_max, bwd_pkt_len_min, pkt_len_mean, fwd_pkt_len_std,
bwd_pkt_len_std, flow_byts_s, fwd_pkt_len_mean, psh_flag_cnt,
down_up_ratio, fwd_act_data_pkts`

Discrete (wires 14–15): `syn_flag_cnt, fin_flag_cnt`

## Validation (held-out session-protocol test set, RF, 3 seeds)

| feature set | macro-F1 |
|---|---|
| previous manual 16 | 0.4087 ± 0.0024 |
| **selected 16** | **0.4153 ± 0.0007** |
| all 62 features | 0.3962 |

Both 16-dim sets beat the full 62 — under session shift, the pruned noise
features actively hurt generalization. The selected set beats the previous
one consistently across seeds. Sanity checks: max pairwise |Spearman| within
the set = 0.899 (< 0.90); all minima ≥ 0, so the frozen log1p → MinMax
preprocessing applies without modification.

## Changes vs the previous manual 16

Kept (7): `pkt_len_min, fwd_pkt_len_min, fwd_pkt_len_max, pkt_len_mean,
flow_byts_s, fwd_pkt_len_mean, psh_flag_cnt`.

Removed and why:
- `init_fwd_win_byts`, `init_bwd_win_byts` — grouped ranks 18/15 with high
  variance; TCP initial windows are OS/path properties, i.e. session
  fingerprints. Their apparent value on the old leaky split does not survive
  cross-session evaluation.
- `pkt_len_var` — deterministic square of `pkt_len_std` (and neither made
  the final 14; the directional stds `fwd_/bwd_pkt_len_std` rank higher).
- `flow_pkts_s`, `bwd_pkt_len_mean`, `bwd_pkt_len_max` — Spearman-redundant
  with retained features (`flow_byts_s`, `bwd_seg_size_avg`).
- `fwd_psh_flags` — near-duplicate of `psh_flag_cnt`.
- `fwd_byts_b_avg` — below rank 20 under grouped evaluation.

Added: `bwd_seg_size_avg, totlen_bwd_pkts, bwd_pkt_len_min, fwd_pkt_len_std,
bwd_pkt_len_std, down_up_ratio, fwd_act_data_pkts, syn_flag_cnt, fin_flag_cnt`.

## Side benefit for the QGAN

The previous "discrete register" features (`fwd_psh_flags`, `psh_flag_cnt`)
take >1,200 distinct values up to ~3,200 — they were counts, not flags, and
the circuit's discrete-boundary narrative did not apply to them.
`syn_flag_cnt` / `fin_flag_cnt` have 8/12 values with 76%/91% zeros: the
discrete register now models genuinely discrete quantities, and the
round-to-integer postprocessing (to be re-enabled in the model pass) becomes
meaningful. `psh_flag_cnt` moves to the continuous register where it belongs.

## Caveats

- Permutation-importance std is large relative to means (few sessions per
  class drive fold heterogeneity); exact ranks 10–20 are not stable, the
  top-half membership is. The head-to-head on held-out sessions is the
  binding evidence, not the rank order.
- Validation classifier is RF; the MLP experiments should confirm the gap.
  Both feature sets were chosen without touching the test split.
- Switching feature sets invalidates all previous QGAN/CTGAN/MLP artifacts:
  retrain everything after adopting this set.
