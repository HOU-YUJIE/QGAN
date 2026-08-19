# QGAN Minority-Class Augmentation for Encrypted Traffic Classification

## Objective

This project designs a quantum generative model to improve
minority-class augmentation for encrypted traffic classification. The study
compares a 16-qubit QGAN against CTGAN, a parameter-matched classical
generator, and non-generative rebalancing methods on the MalayaNetwork_GT
dataset.

## Dataset

The benchmark is MalayaNetwork_GT, containing encrypted traffic flows from
10 application classes. The preprocessing pipeline removes identifier
features, drops invalid values, removes zero-variance columns, and eliminates
feature-identical rows with conflicting labels and exact duplicate feature
rows.

The final feature representation is reduced from 62 raw variables to 16
selected features. The selection pipeline first ranks the top 25 features by
train-only random-forest importance and then retains the final 16-feature
set in fixed order, which is used as the qubit wiring order.

## Split protocols

Two evaluation protocols are used:

- Session protocol: flows from the same capture session are kept in the same
  split. This is the primary evaluation protocol and is closer to deployment
  conditions.
- Random protocol: a stratified flow-level split. This is a secondary,
  same-environment reference protocol.

The class counts are balanced in downstream training by applying the selected
rebalancing strategy to each dataset variant.

## Experimental conditions

The following training-set variants are compared:

- baseline (imbalanced)
- undersample
- oversample
- SMOTE
- QGAN
- CTGAN
- classical generator

The same downstream classifier is used across all conditions, and all results
are reported with paired comparisons across multiple random seeds.

## Model setup

### QGAN

The QGAN uses a 16-qubit generator with a 3-layer circuit and WGAN-GP
training. The main configuration uses the live v2 circuit and feature-wise
quantile preprocessing, which maps each feature to an approximately uniform
distribution before the quantum circuit is applied.

### Baselines

- CTGAN: trained with its original architecture and training budget
- Classical generator: matched to the QGAN in parameter count for a fair
  capacity comparison
- Rebalancing-only baselines: undersampling, oversampling, and SMOTE

### Downstream classifier

The final classifier is a multilayer perceptron trained on the augmented data.
Metrics are macro-F1 and macro-recall, with accuracy excluded from the main
comparison protocol.

## Evaluation metrics

The evaluation focuses on downstream utility and synthetic-data quality:

- macro-F1
- macro-recall
- paired statistical tests against undersampling and baseline conditions
- synthetic-data checks including detectability and memorization indicators

The main interpretation is based on downstream classification performance,
while synthetic-quality metrics are used to explain the observed behavior.

## Results summary

### Session protocol

Generative augmentation improves over undersampling in the realistic session
setting. The QGAN achieves a mean macro-F1 gain of about +0.0140 against the
undersampled baseline, while CTGAN shows a similar gain of +0.0142.

### Random protocol

Under the same-environment random split, no generator outperforms the
undersampling baseline. Oversampling is the strongest condition in this
setting, and the QGAN is not competitive with the rebalancing-only baseline.

### Protocol interaction

The effect of the QGAN relative to undersampling flips sign between the two
protocols. This indicates that the synthetic samples provide regularization
that is useful under distribution shift but not under the simpler,
within-environment random split.

### Parameter parity

The QGAN and CTGAN have very different model sizes, yet their downstream
performance is statistically indistinguishable under both protocols. This
suggests that the quantum circuit is not providing a clear per-parameter
advantage in this setting.

### Main comparison table

| condition | session F1 | ΔF1 | p | random F1 | ΔF1 | p |
|---|---:|---:|---:|---:|---:|---:|
| baseline (imbalanced) | 0.2764 | +0.0099 | 0.064 | 0.4269 | -0.0037 | 0.689 |
| undersample | 0.2665 | — | — | 0.4306 | — | — |
| oversample | 0.2836 | +0.0172 | 0.006 | 0.4489 | +0.0183 | 0.021 |
| smote | 0.2862 | +0.0198 | <0.001 | 0.4458 | +0.0152 | 0.071 |
| qgan | 0.2804 | +0.0140 | <0.001 | 0.4161 | -0.0145 | 0.106 |
| ctgan | 0.2806 | +0.0142 | <0.001 | 0.4236 | -0.0070 | 0.333 |
| classical | 0.2753 | +0.0089 | 0.033 | 0.4230 | -0.0076 | 0.379 |

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the data pipeline:

```bash
python src/data/clean_data.py
python src/data/split_dataset.py --protocol session
python src/data/feature_selection.py --top-k 25
python src/data/select_manual_features.py
```

Run the main experiment pipeline:

```bash
bash scripts/run_main.sh
```

Additional analyses:

```bash
PARALLEL_JOBS=4 THREADS_PER_JOB=4 bash scripts/run_grid.sh
python scripts/summarize_runs.py outputs/ablation
python scripts/sweep_classical_capacity.py --classes 4 9 8
python scripts/data_audit_table.py
python scripts/plot_training_curves.py
python -m src.qgan.circuit
```

## Project structure

```text
src/
  config.py
  data/
  qgan/
  baselines/
  fusion/
  mlp/
  evaluation/

scripts/
  run_main.sh
  run_grid.sh
  summarize_runs.py
  sweep_classical_capacity.py
  data_audit_table.py
  plot_training_curves.py
  verify_version.py

reports/
  feature_selection_16.md
  qgan_step4_design.md

outputs/
outputs_session/

data/processed/
```

## Notes

- The primary conclusion is that QGAN-based augmentation is useful under the
  session-based protocol but not under the random split that reflects a more
  optimistic same-environment setting.
- The experimental findings support the interpretation that generator
  quality is task- and distribution-dependent rather than universally
  superior to classical rebalancing methods.
- The key comparison metric is macro-F1. Macro-recall is treated as an
  additional robustness check and is reported alongside the main results.

License: [LICENSE](LICENSE).
