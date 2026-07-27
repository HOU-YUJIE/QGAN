"""Train CTGAN per minority class and export synthetic samples.

Replaces src/mlp/augment_ctgan.py (delete it). Fixes and changes:
  * `for category in range(3, 7)` covered only classes 3-6, so the committed
    script could never have produced the committed 10-class results. Minority
    classes are now DERIVED from training counts (config.get_majority_labels),
    identical to the QGAN pipeline.
  * Output layout mirrors the QGAN one: one directory per class with
    synthetic.csv + model_manifest.json, consumed by fusion/build_datasets.py.
    Dataset assembly is NOT done here - one constructor for all conditions.
  * Discrete flag features are declared to CTGAN as discrete_columns.
  * Seeded per class.

CTGAN hyperparameters are FROZEN at the original values (epochs=300,
(256,256,256) nets). The training-budget asymmetry vs the ~150-parameter
QGAN/classical generators is a known property of the comparison - report it,
do not hide it (manifest records epochs and parameter counts).

Usage:
    python src/baselines/train_ctgan.py <category>          # one class
    python src/baselines/train_ctgan.py --all               # all minority classes
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from ctgan import CTGAN

from src.config import (
    DISCRETE_FEATURES, FEATURES16_TRAIN, LABEL_COLUMN, MANUAL_FEATURES_16,
    SEED, TARGET_TOTAL_SAMPLES, generator_model_dir, get_majority_labels,
    set_seed, synthetic_file,
)

CTGAN_EPOCHS = 300  # frozen original budget


def train_one(label: int, epochs: int, seed: int) -> None:
    set_seed(seed + label)
    df = pd.read_csv(FEATURES16_TRAIN)
    real = df.loc[df[LABEL_COLUMN] == label, MANUAL_FEATURES_16]
    n_needed = max(0, TARGET_TOTAL_SAMPLES - len(real))
    if n_needed == 0:
        print(f"class {label}: already >= {TARGET_TOTAL_SAMPLES} rows, skipping")
        return
    print(f"class {label}: {len(real)} real rows -> training CTGAN "
          f"({epochs} epochs), will sample {n_needed}")

    model = CTGAN(epochs=epochs, verbose=False, cuda=False)
    model.fit(real, discrete_columns=[c for c in DISCRETE_FEATURES if c in real.columns])
    syn = model.sample(n_needed)[MANUAL_FEATURES_16]
    syn[LABEL_COLUMN] = label

    out_dir = generator_model_dir("ctgan", label)
    os.makedirs(out_dir, exist_ok=True)
    syn.to_csv(synthetic_file("ctgan", label), index=False)
    n_params = sum(p.numel() for net in (model._generator,) if net is not None
                   for p in net.parameters())
    with open(os.path.join(out_dir, "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"label": int(label), "generator": "ctgan", "epochs": int(epochs),
                   "seed": int(seed + label), "n_real": int(len(real)),
                   "n_synthetic": int(n_needed), "generator_params": int(n_params),
                   "feature_order": MANUAL_FEATURES_16}, f, indent=2)
    print(f"[saved] {synthetic_file('ctgan', label)} ({n_needed} rows, "
          f"generator params: {n_params})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("category", type=int, nargs="?", default=None)
    p.add_argument("--all", action="store_true", help="train all minority classes")
    p.add_argument("--epochs", type=int, default=CTGAN_EPOCHS)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()

    if a.all:
        majority = get_majority_labels(FEATURES16_TRAIN)
        labels = sorted(set(pd.read_csv(FEATURES16_TRAIN, usecols=[LABEL_COLUMN])
                            [LABEL_COLUMN].unique()) - majority)
        print(f"minority classes (derived): {labels}")
    elif a.category is not None:
        labels = [a.category]
    else:
        p.error("give a category or --all")
    for lb in labels:
        train_one(lb, a.epochs, a.seed)


if __name__ == "__main__":
    main()
