"""Multi-seed MLP evaluation over all training-set conditions.

Step-6 rewrite of src/mlp/train.py. The CLASSIFIER is frozen from
impg@735bdfc (64-LayerNorm-32 MLP, Adam lr=1e-3, batch 128, CrossEntropy;
StandardScaler fitted on the baseline training set and shared across all
conditions; fixed feature order). What changed is the EVALUATION PROTOCOL:

  * every condition is trained with N seeds (default 5: 42..46) - with
    condition gaps around 0.002 macro-F1, single-seed numbers are noise
  * 10% stratified validation split with early stopping on val macro-F1
    (patience 10), best-epoch weights restored - replaces fixed 100 epochs
  * test metrics: accuracy, macro P/R/F1, per-class F1
  * paired significance tests (paired t-test + Wilcoxon, paired by seed)
    of every condition against BOTH reference points:
      - baseline    ("does anything help at all?")
      - undersample ("does it help beyond mere rebalancing?")
    The second comparison is the one that can credit the generators.

Outputs (all under outputs/results/):
  mlp_runs.csv      one row per (condition, seed)
  mlp_summary.csv   mean +/- std per condition + significance columns
  mlp_per_class.csv per-class F1 mean per condition
  result.txt        human-readable report

Usage:
    python src/mlp/train.py                          # all built conditions
    python src/mlp/train.py --conditions baseline undersample qgan --seeds 5
"""

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    FEATURES16_TEST, LABEL_COLUMN, MANUAL_FEATURES_16, NUM_CLASSES,
    RESULTS_DIR, SEED, TRAIN_CONDITIONS, condition_file, set_seed,
)

EPOCHS_MAX = 100
BATCH_SIZE = 128
LR = 0.001
VAL_FRACTION = 0.10
PATIENCE = 10  # epochs without val macro-F1 improvement


class TrafficClassifierMLP(nn.Module):
    """Frozen architecture from impg@735bdfc."""

    def __init__(self, input_dim=16, num_classes=NUM_CLASSES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def run_once(condition: str, seed: int, scaler: StandardScaler,
             X_test: np.ndarray, y_test: np.ndarray, device) -> dict:
    set_seed(seed)
    df = pd.read_csv(condition_file(condition))
    X = scaler.transform(df[MANUAL_FEATURES_16].values)
    y = df[LABEL_COLUMN].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VAL_FRACTION, random_state=seed, stratify=y)

    tr_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                           batch_size=BATCH_SIZE, shuffle=True)
    X_val_t = torch.FloatTensor(X_val).to(device)

    model = TrafficClassifierMLP().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_state, best_epoch, since = -1.0, None, 0, 0
    for epoch in range(1, EPOCHS_MAX + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).argmax(1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro")
        if val_f1 > best_f1:
            best_f1, best_state, best_epoch, since = val_f1, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since += 1
            if since >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_test).to(device)).argmax(1).cpu().numpy()

    per_class = f1_score(y_test, pred, average=None, labels=range(NUM_CLASSES))
    return {
        "condition": condition, "seed": seed,
        "accuracy": accuracy_score(y_test, pred),
        "macro_precision": precision_score(y_test, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, pred, average="macro"),
        "best_epoch": best_epoch, "stopped_epoch": epoch,
        **{f"f1_class_{c}": float(per_class[c]) for c in range(NUM_CLASSES)},
    }


def paired_tests(runs: pd.DataFrame, cond: str, ref: str) -> dict:
    a = runs[runs.condition == cond].sort_values("seed")["macro_f1"].values
    b = runs[runs.condition == ref].sort_values("seed")["macro_f1"].values
    if len(a) != len(b) or len(a) < 2:
        return {"delta": np.nan, "t_p": np.nan, "wilcoxon_p": np.nan}
    t_p = stats.ttest_rel(a, b).pvalue
    try:
        w_p = stats.wilcoxon(a, b).pvalue
    except ValueError:  # all differences zero
        w_p = 1.0
    return {"delta": float(np.mean(a - b)), "t_p": float(t_p), "wilcoxon_p": float(w_p)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", default=None,
                   help=f"subset of {TRAIN_CONDITIONS}; default = all with a built file")
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (42..42+n-1)")
    a = p.parse_args()

    conditions = a.conditions or [c for c in TRAIN_CONDITIONS
                                  if os.path.exists(condition_file(c))]
    missing = [c for c in conditions if not os.path.exists(condition_file(c))]
    if missing:
        p.error(f"condition files missing (run fusion/build_datasets.py): {missing}")
    seeds = list(range(SEED, SEED + a.seeds))
    device = torch.device("cpu")
    print(f"conditions: {conditions} | seeds: {seeds}")

    # shared scaler: fitted ONCE on the baseline (real) training distribution
    base = pd.read_csv(condition_file("baseline")) if os.path.exists(condition_file("baseline")) \
        else pd.read_csv(condition_file(conditions[0]))
    scaler = StandardScaler().fit(base[MANUAL_FEATURES_16].values)

    test = pd.read_csv(FEATURES16_TEST)
    X_test = scaler.transform(test[MANUAL_FEATURES_16].values)
    y_test = test[LABEL_COLUMN].values

    rows = []
    for cond in conditions:
        for seed in seeds:
            r = run_once(cond, seed, scaler, X_test, y_test, device)
            rows.append(r)
            print(f"[{cond:12s} seed {seed}] macro-F1 {r['macro_f1']:.4f} "
                  f"acc {r['accuracy']:.4f} (best epoch {r['best_epoch']})")
    runs = pd.DataFrame(rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    runs.to_csv(os.path.join(RESULTS_DIR, "mlp_runs.csv"), index=False)

    metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    summary = runs.groupby("condition")[metrics].agg(["mean", "std"]).round(4)
    summary = summary.reindex([c for c in conditions])

    sig_rows = []
    for cond in conditions:
        row = {"condition": cond}
        for ref in ("baseline", "undersample"):
            if ref in conditions and cond != ref:
                t = paired_tests(runs, cond, ref)
                row[f"dF1_vs_{ref}"] = round(t["delta"], 4)
                row[f"p_t_vs_{ref}"] = round(t["t_p"], 4)
                row[f"p_wilcoxon_vs_{ref}"] = round(t["wilcoxon_p"], 4)
        sig_rows.append(row)
    sig = pd.DataFrame(sig_rows).set_index("condition")

    per_class = (runs.groupby("condition")[[f"f1_class_{c}" for c in range(NUM_CLASSES)]]
                 .mean().round(4).reindex(conditions))
    per_class.to_csv(os.path.join(RESULTS_DIR, "mlp_per_class.csv"))
    summary.to_csv(os.path.join(RESULTS_DIR, "mlp_summary.csv"))

    report = ["=" * 72, f"MLP evaluation | {len(seeds)} seeds {seeds} | test rows: {len(y_test)}",
              "=" * 72, "", summary.to_string(), "",
              "significance (paired by seed, macro-F1):", sig.to_string(), "",
              "per-class F1 (mean over seeds):", per_class.to_string(), "",
              "reading guide: dF1_vs_undersample > 0 with small p is the only",
              "result that can credit a generator; dF1_vs_baseline conflates",
              "rebalancing with synthesis quality."]
    text = "\n".join(report)
    with open(os.path.join(RESULTS_DIR, "result.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
