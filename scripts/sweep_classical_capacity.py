"""Classical-generator capacity sweep: the parameter-efficiency curve.

Trains the classical control generator at several hidden widths (params =
33*h + 16) under the SAME harness/protocol as the QGAN (quantile preproc,
validation selection, early stopping), then assembles a params-vs-quality
table so you can read off: "how many classical parameters does it take to
match the 144-parameter QGAN?"

Quality = the critic-independent validation metrics already produced by
train.py (std_w1, corr_dist at the best checkpoint). For the paper's final
version, also run synth_quality.py on the winning widths for C2ST/DCR.

Outputs:
  outputs/results/classical_capacity_sweep.csv
  outputs/results/figures/classical_capacity_curve.{png,pdf}

Usage:
    python scripts/sweep_classical_capacity.py --classes 4 9 8
    python scripts/sweep_classical_capacity.py --dry-run     # list planned runs
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

import src.qgan.circuit as C
import src.qgan.train as T
from src.config import RESULTS_DIR, SEED

HIDDEN_WIDTHS = [4, 8, 16, 32, 64]          # params: 148, 280, 544, 1072, 2128
SWEEP_ROOT = "outputs/models/capacity_sweep"


def run_dir(label: int, hidden: int, seed: int) -> str:
    return os.path.join(SWEEP_ROOT, f"c{label}_h{hidden}_s{seed}")


def train_width(label: int, hidden: int, seed: int, epochs: int) -> None:
    out = run_dir(label, hidden, seed)
    if os.path.exists(os.path.join(out, "model_manifest.json")):
        print(f"[skip] c{label} h{hidden} s{seed}")
        return
    # The sweep owns the generator width: train.py imported build_generator
    # by name, so we point that name at a width-specific constructor. The
    # rest of the harness (preproc, validation, early stop) is untouched.
    T.build_generator = lambda version, h=hidden: C.ClassicalGenerator(hidden=h)
    print(f"[train] c{label} h{hidden} ({33*hidden+16} params) s{seed}")
    T.train_one_class(label, seed, circuit="classical", preproc="quantile",
                      epochs=epochs, out_dir=out)


def collect() -> pd.DataFrame:
    rows = []
    for mpath in sorted(glob.glob(os.path.join(SWEEP_ROOT, "*", "model_manifest.json"))):
        d = os.path.dirname(mpath)
        with open(mpath, encoding="utf-8") as f:
            m = json.load(f)
        tag = os.path.basename(d)                      # cL_hH_sS
        _, h_part, _ = tag.split("_")
        hidden = int(h_part[1:])
        hist = pd.read_csv(os.path.join(d, "training_history.csv")).dropna(subset=["score"])
        best = hist.loc[hist["score"].idxmin()]
        rows.append({"label": m["label"], "hidden": hidden,
                     "params": 33 * hidden + 16, "seed": m["seed"],
                     "best_epoch": int(best["epoch"]),
                     "std_w1": round(float(best["std_w1"]), 4),
                     "corr_dist": round(float(best["corr_dist"]), 4),
                     "score": round(float(best["score"]), 4)})
    return pd.DataFrame(rows).sort_values(["label", "params"])


def plot(df: pd.DataFrame, out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for metric, ax in zip(("corr_dist", "std_w1"), axes):
        for label, grp in df.groupby("label"):
            g = grp.groupby("params")[metric].mean()
            ax.plot(g.index, g.values, marker="o", markersize=4,
                    linewidth=1.6, label=f"class {label}")
        ax.axvline(144, color="#D85A30", linewidth=1.2, linestyle="--")
        ax.text(144, ax.get_ylim()[1], " QGAN v2 (144)", fontsize=8,
                color="#993C1D", va="top")
        ax.set_xscale("log")
        ax.set_xlabel("classical generator parameters (log)", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=9)
    axes[0].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"classical_capacity_curve.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[saved] {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classes", nargs="+", type=int, default=[4, 9, 8])
    p.add_argument("--hidden", nargs="+", type=int, default=HIDDEN_WIDTHS)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None, help="default: config TOTAL_EPOCHS")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    epochs = a.epochs or T.TOTAL_EPOCHS
    seeds = list(range(SEED, SEED + a.seeds))

    plan = [(c, h, s) for c in a.classes for h in a.hidden for s in seeds]
    print(f"{len(plan)} runs planned "
          f"(classes {a.classes} x hidden {a.hidden} x seeds {seeds}, {epochs} epochs)")
    if a.dry_run:
        for c, h, s in plan:
            print(f"  c{c} h{h} ({33*h+16} params) s{s} -> {run_dir(c, h, s)}")
        return

    for c, h, s in plan:
        train_width(c, h, s, epochs)

    df = collect()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "classical_capacity_sweep.csv"), index=False)
    print(df.to_string(index=False))
    plot(df, os.path.join(RESULTS_DIR, "figures"))


if __name__ == "__main__":
    main()
