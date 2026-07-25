"""Plot training curves from ablation/main-run histories, PPT-ready.

Default: corr_dist (the RQ2 evidence - v1 flat vs v2 descending). One
subplot per class; line color encodes circuit+preproc config, one line per
seed. Saves PNG (300 dpi) and PDF.

Usage:
    python scripts/plot_training_curves.py                        # corr_dist
    python scripts/plot_training_curves.py --metric std_w1
    python scripts/plot_training_curves.py --root outputs/ablation --out outputs/results/figures
"""

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CONFIG_STYLE = {  # color, label
    "v1_log_minmax": ("#888780", "v1 + log-minmax"),
    "v2_log_minmax": ("#378ADD", "v2 + log-minmax"),
    "v2_quantile":   ("#1D9E75", "v2 + quantile"),
    "v1_quantile":   ("#D85A30", "v1 + quantile"),
}
RUN_RE = re.compile(r"c(?P<cls>\d+)_(?P<config>v\d_\w+?)_s(?P<seed>\d+)$")


def label_names() -> dict:
    try:
        from src.config import LABEL_MAPPING_FILE
        with open(LABEL_MAPPING_FILE, encoding="utf-8") as f:
            return {v: k for k, v in json.load(f).items()}
    except Exception:
        return {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs/ablation")
    p.add_argument("--metric", default="corr_dist",
                   choices=["corr_dist", "std_w1", "score", "d_loss", "g_loss"])
    p.add_argument("--out", default="outputs/results/figures")
    a = p.parse_args()

    runs = []
    for path in sorted(glob.glob(os.path.join(a.root, "*", "training_history.csv"))):
        m = RUN_RE.search(os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        h = pd.read_csv(path)
        if a.metric in ("d_loss", "g_loss"):
            h = h.dropna(subset=[a.metric])
        else:
            h = h.dropna(subset=["score"])  # validation rows only
        runs.append({**m.groupdict(), "history": h})
    if not runs:
        raise SystemExit(f"no parsable training_history.csv under {a.root}")

    names = label_names()
    classes = sorted({r["cls"] for r in runs}, key=int)
    fig, axes = plt.subplots(1, len(classes), figsize=(4.2 * len(classes), 3.6),
                             sharey=True, squeeze=False)

    seen = {}
    for ax, cls in zip(axes[0], classes):
        for r in (r for r in runs if r["cls"] == cls):
            color, label = CONFIG_STYLE.get(r["config"], ("#B4B2A9", r["config"]))
            key = r["config"]
            ax.plot(r["history"]["epoch"], r["history"][a.metric],
                    color=color, alpha=0.85, linewidth=1.6,
                    label=label if key not in seen else None)
            seen[key] = True
        title = f"Class {cls}" + (f" ({names[int(cls)]})" if int(cls) in names else "")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch", fontsize=10)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=9)
    axes[0][0].set_ylabel(a.metric, fontsize=10)

    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()

    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(a.out, f"training_{a.metric}_by_class.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
