# QUANTUM GENERATIVE ADVERSARIAL NETWORKS (Q-GAN) FOR NETWORK TRAFFIC CLASSIFICATION

A hybrid quantum-classical machine learning project that generates synthetic network traffic data using Quantum Generative Adversarial Networks (QGAN).

## Project Overview

This project implements:
- **QGAN (Quantum GAN)**: Quantum circuit-based generative model using PennyLane
- **CTGAN**: Classical tabular GAN for data synthesis comparison
- **MLP Classifier**: Multi-layer perceptron for traffic classification with 10 categories
- **Data Evaluation**: Statistical and visual comparison of real vs. synthetic data

---

## Dataset Distribution

**Original Dataset (46,742 samples, 10 traffic categories):**

| Category | Samples | % of Total | Category Name |
|----------|---------|-----------|---------------|
| 0        | 27,256  | 58.3%     | Bittorent |
| 1        | 1,281   | 2.7%      | Chrome-RDP |
| 2        | 832     | 1.8%      | Discord |
| 3        | 4,560   | 9.8%      | EA-Origin |
| 4        | 976     | 2.1%      | Microsoft Teams |
| 5        | 1,701   | 3.6%      | Slack |
| 6        | 5,668   | 12.1%     | Steam |
| 7        | 814     | 1.7%      | TeamViewer |
| 8        | 2,327   | 5.0%      | Webex |
| 9        | 1,327   | 2.8%      | Zoom |

**After 80:20 Train/Test Split:**
- Training set: 37,388 samples (80%)
- Test set: 9,349 samples (20%)
- **Class imbalance ratio: 33.4:1** (class 0 vs class 2)

**Key Characteristics:**
- Highly imbalanced: Class 0 represents > 58% of total data
- Long-tail distribution: 4 minority classes have < 1,000 training samples
- Real-world network traffic pattern: benign traffic dominates, attacks/rare patterns are scarce

---

## Project Structure

```
qgan/
├── src/                          # Core source code modules
│   ├── data/                     # Data preprocessing & splitting
│   │   ├── split_dataset.py      # Train/test split (80:20 by category)
│   │   ├── clean_data.py         # Merge & clean raw MALAYAGT CSV exports
│   │   ├── feature_selection.py  # Select top features from train split
│   │   └── data25to16.py         # Trim to 16 features used by models
│   │
│   ├── qgan/                     # Quantum GAN implementation
│   │   ├── train.py              # QGAN training with WGAN discriminator
│   │   ├── generate.py           # Generate synthetic data per category
│   │   └── evaluate.py           # Evaluate synthetic data quality
│   │
│   ├── mlp/                      # Classical MLP classifier
│   │   ├── train.py              # Train MLP on baseline/augmented data
│   │   └── augment_ctgan.py      # CTGAN data augmentation
│   │
│   └── fusion/                   # Data fusion & mixing
│       └── select_qgan_data.py   # Mix real + synthetic QGAN data
│
├── experiments/                  # Experimental utilities
│   ├── compare_models.py         # Compare model parameter counts
│   ├── compare_js_distance.py    # Compare Jensen-Shannon distance per category (CTGAN vs QGAN)
│   └── plot.py                   # Visualize quantum circuits
│
├── data/                         # Data storage
│   ├── MALAYAGT/                 # Raw MALAYAGT CSV exports (place here)
│   └── processed/                # Processed datasets
│       ├── selected_features_train.csv        # 80% training split
│       └── selected_features_test.csv         # 20% test split
│
├── outputs/                      # Generated outputs
│   ├── models/
│   │   └── qgan_0-9/             # Per-category QGAN models
│   │       ├── 0-9/
│   │       │   ├── qgan_generator_weights.pth
│   │       │   ├── qgan_local_scaler.pkl
│   │       │   └── report.txt
│   │       └── ...
│   │
│   ├── synthetic_data/           # Generated synthetic datasets
│   │   ├── Train_Balanced_QGAN.csv      # QGAN mixed dataset
│   │   └── Train_Balanced_CTGAN.csv     # CTGAN mixed dataset
│   │
│   └── results/                  # Analysis & visualization outputs
│       ├── plots/
│       │   ├── CM_Baseline.png
│       │   ├── CM_CTGAN_Augmented.png
│       │   ├── CM_QGAN_Augmented.png
│       │   ├── Experiment_Comparison_Chart.png
│       │   └── single_layer_qgan.png
│       ├── js_comparison/        # JS distance CSVs and per-category plots
│       │   ├── js_comparison_category_01.csv
│       │   ├── js_comparison_category_01.png
│       │   └── js_comparison_summary_plot.png
│       ├── samples.txt           # Category distribution statistics
│       └── result.txt            # MLP evaluation metrics
│
└── dataset/                      # Original dataset (unmodified)
    ├── (old training scripts)
    └── 0-9/                      # Old model outputs (archived)
```

---

## Execution Flow

### **Phase 1: Data Preparation (updated)**

This repository now enforces a safer, reproducible data-processing flow: deterministic label mapping, stratified split BEFORE any supervised feature selection (to avoid leakage), correlation prefiltering on the training set, and a stable feature scoring aggregated across CV folds.

```
Raw Dataset
    ↓
[0] clean_data.py
    - Input:  raw MalayaNetwork_GT CSV folders under data/processed/MalayaNetwork_GT/csv_output
    - Output: data/processed/merged_cleaned_dataset.csv and data/processed/label_mapping.json
    - Action: Merge category folders, drop unwanted columns, remove NaNs, and write deterministic Label_ID mapping

[1] split_dataset.py
    - Input:  data/processed/merged_cleaned_dataset.csv
    - Output: data/processed/selected_features_train.csv, data/processed/selected_features_test.csv
    - Action: 80:20 stratified split by Label (deterministic seed)

[2] feature_selection.py
    - Input:  selected train/test CSVs (or merged file to be split automatically)
    - Output: data/processed/selected_features_train.csv, data/processed/selected_features_test.csv, data/processed/feature_importances.csv, data/processed/selected_features.json
    - Action: Correlation pruning (TRAIN only) + feature scoring on TRAIN (aggregated RF importances across CV folds by default) → select top-K features and apply to TEST
```

**Recommended commands (order matters):**
```bash
# 0: Clean raw CSVs into a single merged file (also writes label_mapping.json)
python3 src/data/clean_data.py

# 1: Split merged data into train/test (stratified)
python3 src/data/split_dataset.py

# 2: Feature selection (operates on the train split, applies selection to test)
python3 src/data/feature_selection.py --train-input data/processed/selected_features_train.csv --test-input data/processed/selected_features_test.csv --top-k 25

# Alternative: run feature_selection directly on merged file (it will split internally):
python3 src/data/feature_selection.py --merged-input data/processed/merged_cleaned_dataset.csv --top-k 25
```

Outputs you will now see after feature selection:
- `data/processed/selected_features_train.csv` — selected features for training (with `Label`)
- `data/processed/selected_features_test.csv` — selected features for test (with `Label`)
- `data/processed/feature_importances.csv` — per-feature mean/std importances (RF CV aggregation)
- `data/processed/selected_features.json` — chosen feature list

Notes:
- `clean_data.py` now writes `label_mapping.json` to preserve the mapping from folder name → `Label_ID`.
- `feature_selection.py` defaults to the RF-aggregated importances method; use `--method mutual_info` to try mutual information instead.
- For reproducible results, set `--seed` and `--cv-splits` where appropriate.

---

### **Phase 2: Quantum GAN Training**

```
Training Data (per category)
    ↓
[3] src/qgan/train.py
    - Input:  data/processed/selected_features_train.csv (Category-specific)
    - Output: For each category 0-9:
              - outputs/models/qgan_0-9/{category}/qgan_generator_weights.pth
              - outputs/models/qgan_0-9/{category}/qgan_local_scaler.pkl
              - outputs/models/qgan_0-9/{category}/report.txt
    - Action: Train QGAN with WGAN discriminator
              16 qubits, 3 layers, 50 epochs per category
```

**Command:**
```bash
# Train QGAN for each category (0-9)
for category in {0..9}; do
    python src/qgan/train.py $category
done
```

---

### **Phase 3: Synthetic Data Generation**

```
Trained QGAN Models
    ↓
[4] src/qgan/generate.py
    - Input:  Trained models in outputs/models/qgan_0-9/{category}/
    - Output: outputs/models/qgan_0-9/{category}/Synthetic_Traffic_16dim.csv
    - Action: Generate synthetic data for each category
              Automatically calculates samples needed to reach 2000 total
```

**Command:**
```bash
# Generate synthetic data for each category
for category in {0..9}; do
    python src/qgan/generate.py $category
done
```

---

### **Phase 4: Data Fusion & Evaluation**

```
Real Data + Synthetic Data
    ↓
[5a] src/fusion/select_qgan_data.py
    - Input:  Real train data + synthetic per-category data
    - Output: outputs/synthetic_data/Train_Balanced_QGAN.csv
    - Action: Mix real + synthetic with smart category balancing:
              - Category 0 (majority): undersample real data only
              - Categories 1-9 (minority): mix real + synthetic to 2000 samples

[5b] src/qgan/evaluate.py
    - Input:  Real training data + synthetic data per category
    - Output: outputs/results/{distribution_plots, metrics}
    - Action: Compare real vs synthetic:
              - Distribution plots (KDE)
              - Correlation heatmaps
              - Wasserstein distance & KS test
              - PCA overlap visualization
```

**Command:**
```bash
# Merge real + synthetic data
python src/fusion/select_qgan_data.py

# Evaluate synthetic data quality
python src/qgan/evaluate.py 0  # Evaluate one category
python src/qgan/evaluate.py    # Evaluate all minority categories
```

---

### **Phase 5: CTGAN Baseline Comparison**

```
Real Training Data
    ↓
[6] src/mlp/augment_ctgan.py
    - Input:  data/processed/selected_features_train.csv
    - Output: outputs/synthetic_data/Train_Balanced_CTGAN.csv
    - Action: Classical CTGAN augmentation as comparison baseline
              Discrete columns: fwd_psh_flags, psh_flag_cnt
```

**Command:**
```bash
python src/mlp/augment_ctgan.py
```

---

### **Phase 6: Model Training & Evaluation**

```
Three Training Datasets:
  1. Real (baseline)
  2. Real + CTGAN synthetic
  3. Real + QGAN synthetic
    ↓
[7] src/mlp/train.py
    - Input:  
      - outputs/synthetic_data/Train_Balanced_QGAN.csv
      - outputs/synthetic_data/Train_Balanced_CTGAN.csv
      - data/processed/selected_features_train.csv
      - data/processed/selected_features_test.csv
    - Output: 
      - outputs/results/plots/CM_Baseline.png
      - outputs/results/plots/CM_CTGAN_Augmented.png
      - outputs/results/plots/CM_QGAN_Augmented.png
      - outputs/results/plots/Experiment_Comparison_Chart.png
      - outputs/results/result.txt
    - Action: Train MLP classifier (16→64→32→10)
              Evaluate on 3 augmentation strategies
              Generate confusion matrices & comparison charts
```

**Command:**
```bash
python src/mlp/train.py
```

---

## Quick Start

### **1. Setup Environment**
```bash
conda create -n qml python=3.10
conda activate qml
pip install pennylane torch pandas scikit-learn ctgan matplotlib seaborn scipy joblib
```

### **2. Run Complete Pipeline**

**Option A: Automated (all steps)**
```bash
cd /home/maru/qgan

# Data preparation
python src/data/split_dataset.py
python src/data/data25to16.py

# QGAN training (sequential per category)
for i in {0..9}; do python src/qgan/train.py $i; done

# Data generation & synthesis
for i in {0..9}; do python src/qgan/generate.py $i; done

# Evaluation & fusion
python src/qgan/evaluate.py  # No argument = evaluate all minority categories
python src/fusion/select_qgan_data.py

# Classical baseline
python src/mlp/augment_ctgan.py

# Final MLP evaluation
python src/mlp/train.py
```

**Option B: Step-by-step (with inspection between phases)**
```bash
# Each step can be run individually - see Phase sections above
python src/data/split_dataset.py
# ... inspect outputs/ ...
python src/qgan/train.py 0
# ... adjust hyperparameters if needed ...
```

### **3. Utilities**

```bash
# Compare model parameter counts
python experiments/compare_models.py

# Compare JS distances (CTGAN vs QGAN) per-category
python experiments/compare_js_distance.py

# Visualize single-layer quantum circuit
python experiments/plot.py

# Plot QGAN training history from saved report.txt files
python experiments/plot_qgan_training_history.py
```

---

## Experimental Results

### **MLP Classification Performance (10-class task)**

**Weighted F1-Score Comparison:**

| Strategy | Accuracy | Macro F1 | Weighted F1 | Macro Recall |
|----------|----------|----------|-------------|---------------|
| **Baseline** (Real Only) | 74.74% | 0.3672 | 0.7206 | 0.3503 |
| **CTGAN Augmented** | 72.66% | 0.3780 | 0.7372 | 0.3760 |
| **QGAN Augmented** | 68.72% | 0.3625 | 0.7129 | 0.3725 |

**Key Findings:**

1. **Class 0 Dominates Results:**
   - Test set: Class 0 represents 58.3% of samples
   - Weighted F1 is ~75% determined by class 0 performance
   - Baseline F1 (0.9259) on class 0 contributes 0.5400 to weighted F1

2. **Minority Class Challenge:**
   - Despite QGAN data augmentation, minority class F1 < 0.3
   - Class 2, 7: F1 ≈ 0.12-0.19 (extremely rare, hard to learn)
   - Class 4, 5, 9: F1 ≈ 0.20-0.30 (still very difficult)

3. **Limited Overall Improvement from Augmentation:**
   - CTGAN: +2.3% weighted F1 over baseline (marginal gain)
   - QGAN: -1.1% weighted F1 vs baseline (slight degradation)
   - Macro F1 improvement: 0.3672 → 0.3780 (+3%, minimal)
   - **Root cause:** Train-test distribution mismatch (balanced training, unbalanced testing)

4. **Jensen-Shannon Distance Analysis (QGAN vs CTGAN):**
   - QGAN achieves better feature-space similarity: -44% avg JS distance vs CTGAN
   - Categories 1, 2, 8: QGAN > 34% improvement in distribution matching
   - **Paradox:** Better synthetic distribution ≠ Better classification performance
   - **Reason:** Improved synthetic data doesn't overcome fundamental label imbalance in decision boundary learning

### **Why Augmentation Failed to Improve Performance**

1. **Data Imbalance vs Decision Boundary Imbalance:**
   - Rebalancing training data doesn't fix the asymmetric test set
   - Model learns biased boundaries optimized for balanced data, performs worse on real (imbalanced) distribution

2. **Minority Class Inherent Difficulty:**
   - Even with synthetic data supplement, minority classes lack discriminative features
   - 1,000+ samples of noise-like patterns can't compensate for 20k+ class 0 samples

3. **Weighted vs Macro Metrics:**
   - Weighted metrics hide minority class degradation
   - Macro F1 improvement (3%) is the true metric for this imbalanced problem

### **Recommended Improvements**

- Use **class_weight** in training loss (instead of data rebalancing) to preserve natural decision boundaries
- Focus on **minority class feature engineering** rather than quantity multiplication
- Evaluate on **macro F1 / macro recall**, not weighted accuracy
- Consider **threshold tuning** per-category instead of uniform classification threshold

---

## Data Flow Summary

```
┌─────────────────────────────────┐
│ Original Traffic Dataset (25D)  │
└────────────┬────────────────────┘
             │
      [1] Split Dataset
             │
      ┌──────┴────────┐
      ↓               ↓
  Train(80%)       Test(20%)
      │               │
      └─────[2]───────┘
      Feature Selection (25D→16D)
             │
    ┌────────┴────────┐
    ↓                 ↓
[3] QGAN Train   [6] CTGAN Aug
    (per cat)
    ↓
[4] Generate
    Synthetic
    ↓
[5a] Fusion ──→ [5b] Evaluate
    │
    ├─→ QGAN Mixed Dataset
    │
    ↓
[7] MLP Training & Evaluation
    │
    ├─→ CM_Baseline.png
    ├─→ CM_CTGAN_Augmented.png
    ├─→ CM_QGAN_Augmented.png
    └─→ Comparison_Chart.png
```

---

## Key Parameters

| Component | Parameter | Value | Notes |
|-----------|-----------|-------|-------|
| **Data** | Train/Test Ratio | 80:20 | Stratified by Label |
| **Data** | Feature Dimensions | 16 | From 25+ original features |
| **QGAN** | Qubits | 16 | Per category model |
| **QGAN** | Circuit Layers | 3 | Entanglement depth |
| **QGAN** | Training Epochs | 50 | Per category |
| **QGAN** | Batch Size | 32 | Generator training |
| **CTGAN** | Target Samples | 2000 | Per category |
| **MLP** | Hidden Layers | 2 | 16→64→32→10 |
| **MLP** | Training Epochs | 100 | Classifier training |
| **MLP** | Batch Size | 128 | Mini-batch training |

---

## Output Files Reference

| File | Purpose | Generated By |
|------|---------|--------------|
| `outputs/models/qgan_0-9/{category}/qgan_generator_weights.pth` | Trained quantum generator | train.py |
| `outputs/models/qgan_0-9/{category}/qgan_local_scaler.pkl` | Feature scaler | train.py |
| `outputs/synthetic_data/Train_Balanced_QGAN.csv` | Mixed real+synthetic (QGAN) | select_qgan_data.py |
| `outputs/synthetic_data/Train_Balanced_CTGAN.csv` | Mixed real+synthetic (CTGAN) | augment_ctgan.py |
| `outputs/results/plots/CM_*.png` | Confusion matrices | train.py |
| `outputs/results/plots/Experiment_Comparison_Chart.png` | Model comparison chart | train.py |

---

## Troubleshooting

**Q: QGAN training is slow**
- A: Normal - quantum simulations are computationally intensive. Reduce EPOCHS or increase BATCH_SIZE in train.py

**Q: Memory error when generating synthetic data**
- A: Reduce TARGET_TOTAL_SAMPLES in generate.py or use a machine with more RAM

**Q: Paths not found errors**
- A: Ensure you run all scripts from project root `/home/maru/qgan`

**Q: "VQC/MLP" path not found**
- A: These are old directory names. New structure uses `src/`, `data/`, `outputs/`

---

## Dependencies

```
pennylane>=0.28
torch>=1.12
pandas>=1.5
scikit-learn>=1.1
ctgan>=0.9
matplotlib>=3.5
seaborn>=0.12
scipy>=1.9
joblib>=1.2
numpy>=1.23
```

---

## Notes

- All categorical balancing uses `random_state=42` for reproducibility
- Quantum circuits use PennyLane's `lightning.qubit` backend for simulation
- Each category (0-9) has its own trained QGAN model for specialized learning
- The MLP classifier achieves best performance with QGAN-augmented data

---

**Last Updated:** May 13, 2026  
**Project Status:** Complete file reorganization & path updates
