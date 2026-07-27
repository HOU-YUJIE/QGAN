"""Check that this repo copy contains the latest delivered code (steps 0-7).
Run from repo root: python scripts/verify_version.py"""
import os, sys
CHECKS = [
    ("src/qgan/circuit.py", ["def build_generator", "class ClassicalGenerator", "circuit_v2"]),
    ("src/qgan/train.py", ["generator_model_dir", "--patience", "weights_best"]),
    ("src/qgan/generate.py", ["--generator", "def postprocess", "model_manifest"]),
    ("src/qgan/preprocessing.py", ["quantile", "fit_preproc"]),
    ("src/config.py", ["generator_model_dir", "condition_file", "SESSION_COLUMN", "syn_flag_cnt"]),
    ("src/data/clean_data.py", ["label-conflict", "session_id"]),
    ("src/data/split_dataset.py", ["session_split", "choose_test_sessions"]),
    ("src/data/feature_selection.py", ["importance-guided", "corr_method"]),
    ("src/data/select_manual_features.py", ["MANUAL_FEATURES_16"]),
    ("src/baselines/train_ctgan.py", ["get_majority_labels"]),
    ("src/fusion/build_datasets.py", ["minority_topup", "build_smote"]),
    ("src/mlp/train.py", ["paired_tests", "dF1_vs_undersample"]),
    ("src/evaluation/synth_quality.py", ["c2st_auc", "dcr_ratio"]),
    ("scripts/run_main.sh", ["SKIP_QGAN_TRAIN", "select_best"]),
    ("src/config.py", ["TOTAL_EPOCHS = 100"]),
]
bad = 0
for path, markers in CHECKS:
    if not os.path.exists(path):
        print(f"[MISSING] {path}"); bad += 1; continue
    text = open(path, encoding="utf-8").read()
    missing = [m for m in markers if m not in text]
    if missing:
        print(f"[STALE  ] {path} - lacks {missing}"); bad += 1
    else:
        print(f"[ok     ] {path}")
sys.exit(1 if bad else print("\nAll files up to date.") or 0)
