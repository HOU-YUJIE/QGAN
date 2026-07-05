"""Summarize QGAN training runs from their model_manifest.json files.

Usage:
    python scripts/summarize_runs.py outputs/ablation
    python scripts/summarize_runs.py outputs/models/qgan_0-9
"""

import glob
import json
import os
import sys

import pandas as pd


def main(root: str) -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "**", "model_manifest.json"),
                                 recursive=True)):
        with open(path, encoding="utf-8") as f:
            m = json.load(f)
        hist_path = os.path.join(os.path.dirname(path), "training_history.csv")
        final_score = None
        if os.path.exists(hist_path):
            h = pd.read_csv(hist_path)
            if "score" in h.columns and h["score"].notna().any():
                final_score = float(h["score"].dropna().iloc[-1])
        rows.append({
            "run": os.path.relpath(os.path.dirname(path), root),
            "label": m.get("label"), "circuit": m.get("circuit_version"),
            "preproc": m.get("preproc"), "seed": m.get("seed"),
            "best_epoch": m.get("best_epoch"), "stopped": m.get("stopped_epoch"),
            "early": m.get("early_stopped"), "best_score": round(m.get("best_score", float("nan")), 4),
            "final_score": round(final_score, 4) if final_score is not None else None,
        })
    if not rows:
        print(f"no model_manifest.json found under {root}")
        return
    df = pd.DataFrame(rows).sort_values(["label", "circuit", "preproc", "seed"])
    print(df.to_string(index=False))

    if df["circuit"].nunique() > 1 or df["preproc"].nunique() > 1:
        print("\nmean best_score by config (lower is better):")
        print(df.groupby(["circuit", "preproc"])["best_score"]
                .agg(["mean", "std", "count"]).round(4).to_string())
    hitting = df[df["best_epoch"] >= df["stopped"] - 5]
    if len(hitting):
        print(f"\n[note] {len(hitting)} run(s) peaked within 5 epochs of the end - "
              f"budget may be truncating them:")
        print(hitting[["run", "best_epoch", "stopped"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/ablation")
