"""Build ALL training-set conditions with ONE constructor.

Replaces src/fusion/select_qgan_data.py and the assembly half of the old
augment_ctgan.py. Every condition follows the SAME rule; the only degree of
freedom is where minority top-up rows come from:

  baseline     - real imbalanced training set, untouched (copied for uniformity)
  undersample  - majority classes capped at TARGET; minority untouched
  oversample   - undersample + minority duplicated (with replacement) to TARGET
  smote        - undersample + minority SMOTE-interpolated to TARGET
  qgan / ctgan / classical
               - undersample + real minority + generator synthetic to TARGET

Because majority handling, target counts and real/synthetic mixing are shared
code, the conditions differ in exactly one factor - the previous situation
(QGAN and CTGAN sets assembled by different scripts with different class
coverage) cannot recur. Every output ships with a composition manifest and
hard assertions (feature order, no NaN, expected per-class counts).

Usage:
    python src/fusion/build_datasets.py                       # all conditions
    python src/fusion/build_datasets.py baseline smote qgan   # a subset
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from src.config import (
    CONDITIONS_DIR, DISCRETE_FEATURES, FEATURES16_TRAIN, LABEL_COLUMN,
    MANUAL_FEATURES_16, SEED, TARGET_TOTAL_SAMPLES, TRAIN_CONDITIONS,
    assert_feature_frame, condition_file, get_majority_labels, synthetic_file,
)

GENERATOR_CONDITIONS = {"qgan", "ctgan", "classical"}


def minority_topup(condition: str, label: int, real_grp: pd.DataFrame,
                   rng: np.random.RandomState) -> pd.DataFrame:
    """Rows added on top of the real minority rows (may be empty)."""
    n_needed = TARGET_TOTAL_SAMPLES - len(real_grp)
    if n_needed <= 0 or condition == "undersample":
        return real_grp.iloc[0:0]

    if condition == "oversample":
        return real_grp.sample(n_needed, replace=True, random_state=rng)

    if condition in GENERATOR_CONDITIONS:
        path = synthetic_file(condition, label)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{condition}] missing synthetic data for class {label}: {path} - "
                f"run the corresponding generator first.")
        syn = pd.read_csv(path)
        missing = [c for c in MANUAL_FEATURES_16 if c not in syn.columns]
        if missing:
            raise ValueError(f"{path} lacks features {missing}")
        if len(syn) < n_needed:
            raise ValueError(f"{path}: {len(syn)} rows < needed {n_needed} - regenerate.")
        out = syn[MANUAL_FEATURES_16].head(n_needed).copy()
        out[LABEL_COLUMN] = label
        return out

    raise ValueError(f"unknown condition {condition!r}")


def build(condition: str, df: pd.DataFrame, majority: set) -> dict:
    rng = np.random.RandomState(SEED)
    if condition == "baseline":
        out = df[MANUAL_FEATURES_16 + [LABEL_COLUMN]].copy()
        composition = {int(l): {"real": int(n), "synthetic": 0}
                       for l, n in out[LABEL_COLUMN].value_counts().items()}
    elif condition == "smote":
        out, composition = build_smote(df, majority, rng)
    else:
        parts, composition = [], {}
        for label, grp in df.groupby(LABEL_COLUMN):
            grp = grp[MANUAL_FEATURES_16 + [LABEL_COLUMN]]
            if len(grp) >= TARGET_TOTAL_SAMPLES:            # majority (or big minority)
                real = grp.sample(TARGET_TOTAL_SAMPLES, random_state=rng)
                extra = real.iloc[0:0]
            else:
                real, extra = grp, minority_topup(condition, label, grp, rng)
            parts += [real, extra]
            composition[int(label)] = {"real": int(len(real)), "synthetic": int(len(extra))}
        out = pd.concat(parts, ignore_index=True)

    out = out.sample(frac=1.0, random_state=rng).reset_index(drop=True)  # shuffle

    # hard checks
    assert_feature_frame(out, where=f"condition {condition}")
    assert out.isna().sum().sum() == 0, f"{condition}: NaNs in output"
    if condition not in ("baseline",):
        counts = out[LABEL_COLUMN].value_counts()
        bad = {int(l): int(c) for l, c in counts.items()
               if c > TARGET_TOTAL_SAMPLES or (condition != "undersample" and c < TARGET_TOTAL_SAMPLES)}
        assert not bad, f"{condition}: unexpected class counts {bad}"

    os.makedirs(CONDITIONS_DIR, exist_ok=True)
    out.to_csv(condition_file(condition), index=False)
    print(f"[{condition}] {len(out)} rows -> {condition_file(condition)}")
    return composition


def build_smote(df: pd.DataFrame, majority: set, rng) -> tuple:
    from imblearn.over_sampling import SMOTE

    parts = []
    for label, grp in df.groupby(LABEL_COLUMN):
        grp = grp[MANUAL_FEATURES_16 + [LABEL_COLUMN]]
        parts.append(grp.sample(TARGET_TOTAL_SAMPLES, random_state=rng)
                     if len(grp) >= TARGET_TOTAL_SAMPLES else grp)
    inter = pd.concat(parts, ignore_index=True)

    counts = inter[LABEL_COLUMN].value_counts()
    strategy = {int(l): TARGET_TOTAL_SAMPLES for l, c in counts.items()
                if c < TARGET_TOTAL_SAMPLES}
    k = max(1, min(5, int(counts.min()) - 1))
    sm = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=SEED)
    X, y = sm.fit_resample(inter[MANUAL_FEATURES_16], inter[LABEL_COLUMN])
    out = pd.DataFrame(X, columns=MANUAL_FEATURES_16)
    # SMOTE interpolates -> repair discrete flags back to integers
    for c in DISCRETE_FEATURES:
        out[c] = out[c].round().clip(lower=0).astype(int)
    out[LABEL_COLUMN] = y
    composition = {int(l): {"real": int(counts.get(l, 0)),
                            "synthetic": int(TARGET_TOTAL_SAMPLES - counts.get(l, 0))
                            if l in strategy else 0}
                   for l in out[LABEL_COLUMN].unique()}
    return out, composition


def main():
    p = argparse.ArgumentParser()
    p.add_argument("conditions", nargs="*", default=None,
                   help=f"subset of {TRAIN_CONDITIONS}; default = all")
    a = p.parse_args()
    conditions = a.conditions or TRAIN_CONDITIONS
    unknown = set(conditions) - set(TRAIN_CONDITIONS)
    if unknown:
        p.error(f"unknown conditions {unknown}")

    df = pd.read_csv(FEATURES16_TRAIN)
    assert_feature_frame(df, where=FEATURES16_TRAIN)
    majority = get_majority_labels(FEATURES16_TRAIN)
    print(f"majority classes (derived): {sorted(majority)} | target {TARGET_TOTAL_SAMPLES}")

    manifest = {}
    for cond in conditions:
        manifest[cond] = build(cond, df, majority)
    with open(os.path.join(CONDITIONS_DIR, "build_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "target": TARGET_TOTAL_SAMPLES,
                   "majority": sorted(int(m) for m in majority),
                   "composition": manifest}, f, indent=2)
    print(f"[saved] {os.path.join(CONDITIONS_DIR, 'build_manifest.json')}")


if __name__ == "__main__":
    main()
