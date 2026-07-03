"""Generate synthetic samples for one class from a trained QGAN.

Engineering pass only - generation behavior identical to impg@735bdfc:
same circuit (now imported, cannot diverge from training), same inverse
transform (scaler inverse -> expm1), postprocessing constraints still OFF.

What changed:
  * Circuit/generator imported from circuit.py - the (L,16) vs (L,2,16)
    weight-shape divergence that broke load_state_dict cannot recur.
  * Majority classes derived from actual training counts
    (config.get_majority_labels) instead of hardcoded {0, 3, 6}.
  * Seeded for reproducible sampling.

Usage:
    python src/qgan/generate.py <category 0-9> [--seed 42]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import (
    FEATURES16_TRAIN,
    LABEL_COLUMN,
    MANUAL_FEATURES_16,
    SEED,
    TARGET_TOTAL_SAMPLES,
    get_majority_labels,
    qgan_model_dir,
    qgan_synthetic_file,
    set_seed,
)
from src.qgan.circuit import TabularQuantumGenerator, sample_noise


def samples_needed(category: int) -> int:
    majority = get_majority_labels(FEATURES16_TRAIN)
    if category in majority:
        print(f"Category {category} is a majority class (derived, not hardcoded); nothing to generate.")
        return 0
    df = pd.read_csv(FEATURES16_TRAIN, usecols=[LABEL_COLUMN])
    real_count = int((df[LABEL_COLUMN] == category).sum())
    n = max(0, TARGET_TOTAL_SAMPLES - real_count)
    print(f"Category {category}: real={real_count}, target={TARGET_TOTAL_SAMPLES}, to generate={n}")
    return n

def generate(category: int, num_samples: int) -> None:
    if num_samples <= 0:
        return

    model_dir = qgan_model_dir(category)
    weights_path = os.path.join(model_dir, "qgan_generator_weights.pth")
    scaler_path = os.path.join(model_dir, "qgan_local_scaler.pkl")
    if not (os.path.exists(weights_path) and os.path.exists(scaler_path)):
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}; run train.py {category} first.")

    gen = TabularQuantumGenerator()
    gen.load_state_dict(torch.load(weights_path, map_location="cpu"))
    gen.eval()
    scaler = joblib.load(scaler_path)

    print(f"Generating {num_samples} samples for category {category}...")
    with torch.no_grad():
        synthetic_pi = gen(sample_noise(num_samples)).numpy()

    # Inverse transform: [0, pi] -> log space -> original scale
    synthetic_log = scaler.inverse_transform(synthetic_pi)
    synthetic = np.expm1(synthetic_log)

    df_syn = pd.DataFrame(synthetic, columns=MANUAL_FEATURES_16)
    df_syn[LABEL_COLUMN] = category

    # NOTE: physical/protocol postprocessing (integer flags, MTU bounds,
    # min<=mean<=max, ...) is intentionally still DISABLED to keep this an
    # engineering-only change. Re-enabling it is a Step-1 experiment; when
    # you do, put it in a postprocess() function here so it is testable.

    out_path = qgan_synthetic_file(category)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_syn.to_csv(out_path, index=False)
    print(f"[OK] synthetic data saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic samples for one category")
    parser.add_argument("category", type=int, nargs="?", default=0, help="Target label (0-9)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    set_seed(args.seed + args.category)
    generate(args.category, samples_needed(args.category))


if __name__ == "__main__":
    main()