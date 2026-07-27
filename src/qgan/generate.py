"""Generate synthetic samples for one class from a trained QGAN.

Step-4 version. Reads model_manifest.json for circuit version, preprocessing
kind and feature order - the generator is reconstructed exactly as trained,
so a train/generate mismatch is structurally impossible. Loads the BEST
checkpoint (validation-selected), falling back to the last.

Postprocessing (on by default, --no-postprocess to disable for ablation):
  1. clip all features at >= 0 (traffic statistics are non-negative)
  2. round discrete flag features to integers, clip to the training support
  3. repair ordering constraints (min <= mean <= max chains) by sorting

Usage:
    python src/qgan/generate.py <category> [--no-postprocess] [--seed 42]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import (
    DISCRETE_FEATURES, generator_model_dir, synthetic_file, FEATURES16_TRAIN, LABEL_COLUMN, MANUAL_FEATURES_16,
    SEED, TARGET_TOTAL_SAMPLES, get_majority_labels, qgan_model_dir,
    qgan_synthetic_file, set_seed,
)
from src.qgan.circuit import build_generator, sample_noise
from src.qgan.preprocessing import inverse_preproc

# ordering constraints that must hold for physically consistent flows;
# repaired by sorting the group values row-wise (ascending)
ORDER_CHAINS = [
    ("fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_max"),
    ("pkt_len_min", "pkt_len_mean"),
    ("bwd_pkt_len_min", "bwd_seg_size_avg"),
]


def postprocess(df_syn: pd.DataFrame, df_real_class: pd.DataFrame) -> pd.DataFrame:
    df = df_syn.copy()
    feat = [c for c in df.columns if c != LABEL_COLUMN]

    # 1. non-negativity
    df[feat] = df[feat].clip(lower=0)

    # 2. discrete flags: integers within the training support
    for c in DISCRETE_FEATURES:
        if c in df.columns:
            hi = int(df_real_class[c].max())
            df[c] = df[c].round().clip(0, hi).astype(int)

    # 3. ordering repair
    for chain in ORDER_CHAINS:
        cols = [c for c in chain if c in df.columns]
        if len(cols) >= 2:
            df[cols] = np.sort(df[cols].values, axis=1)
    return df


def samples_needed(category: int) -> int:
    majority = get_majority_labels(FEATURES16_TRAIN)
    if category in majority:
        print(f"Category {category} is a majority class; nothing to generate.")
        return 0
    df = pd.read_csv(FEATURES16_TRAIN, usecols=[LABEL_COLUMN])
    real_count = int((df[LABEL_COLUMN] == category).sum())
    n = max(0, TARGET_TOTAL_SAMPLES - real_count)
    print(f"Category {category}: real={real_count}, target={TARGET_TOTAL_SAMPLES}, to generate={n}")
    return n


def generate(category: int, num_samples: int, do_postprocess: bool = True,
             generator: str = "qgan") -> None:
    if num_samples <= 0:
        return
    model_dir = generator_model_dir(generator, category)
    manifest_path = os.path.join(model_dir, "model_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"{manifest_path} missing - run train.py {category} first.")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["feature_order"] == MANUAL_FEATURES_16, \
        "feature order in manifest differs from config - retrain before generating"

    weights = os.path.join(model_dir, "weights_best.pth")
    if not os.path.exists(weights):
        weights = os.path.join(model_dir, "weights_last.pth")
    gen = build_generator(manifest["circuit_version"])
    gen.load_state_dict(torch.load(weights, map_location="cpu"))
    gen.eval()
    preproc_state = joblib.load(os.path.join(model_dir, "preproc.pkl"))
    print(f"Loaded {os.path.basename(weights)} (circuit {manifest['circuit_version']}, "
          f"preproc {manifest['preproc']}, best epoch {manifest['best_epoch']})")

    with torch.no_grad():
        fake_pi = gen(sample_noise(num_samples)).numpy()
    synthetic = inverse_preproc(preproc_state, fake_pi)
    df_syn = pd.DataFrame(synthetic, columns=MANUAL_FEATURES_16)

    if do_postprocess:
        real = pd.read_csv(FEATURES16_TRAIN)
        df_syn = postprocess(df_syn, real[real[LABEL_COLUMN] == category])
    df_syn[LABEL_COLUMN] = category

    out_path = synthetic_file(generator, category)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_syn.to_csv(out_path, index=False)
    print(f"[OK] {len(df_syn)} synthetic rows -> {out_path}"
          + ("" if do_postprocess else "  (postprocess DISABLED)"))


def main():
    p = argparse.ArgumentParser(description="Generate synthetic samples for one category")
    p.add_argument("category", type=int, nargs="?", default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-postprocess", action="store_true")
    p.add_argument("--generator", choices=["qgan", "classical"], default="qgan",
                   help="which trained generator's model dir to load and where to write synthetic.csv")
    a = p.parse_args()
    set_seed(a.seed + a.category)
    generate(a.category, samples_needed(a.category), not a.no_postprocess, a.generator)


if __name__ == "__main__":
    main()
