"""Split the cleaned dataset into train/test under two protocols.

Step-2 rewrite. Protocols:

  session (PRIMARY)  - whole capture sessions are held out for test. Flows
      from one session share the network environment (RTT scale, TCP window
      defaults, throughput regime), so a random flow-level split lets the
      classifier key on session fingerprints. Holding out sessions measures
      generalization to unseen environments. Per class, we pick the subset
      of sessions whose total size is closest to TEST_RATIO (brute force
      over proper subsets; every class keeps >= 1 training session).

  random (SECONDARY) - stratified random flow-level split on the deduped
      data. Optimistic upper bound (same-environment generalization).
      Report BOTH protocols; the gap between them is itself a finding.

The chosen protocol writes the same canonical outputs
(split62_train/test.csv), so downstream stages are protocol-agnostic; the
manifest (split_manifest.json) records which protocol, seed, and sessions
produced the current files. Run the full pipeline once per protocol.

Guardrails (hard assertions, not warnings):
  * zero feature-identical rows across train/test (both protocols)
  * every class present on both sides
  * session protocol: train/test session sets are disjoint

Usage:
    python src/data/split_dataset.py --protocol session
    python src/data/split_dataset.py --protocol random
"""

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABEL_COLUMN,
    MERGED_FILE,
    METADATA_COLUMNS,
    SEED,
    SESSION_COLUMN,
    SPLIT62_TEST,
    SPLIT62_TRAIN,
    SPLIT_MANIFEST_FILE,
    TEST_RATIO,
)


def choose_test_sessions(session_sizes: pd.Series, target: float) -> tuple:
    """Pick the proper subset of sessions whose size fraction is closest to
    `target`. Classes have 2-5 sessions, so brute force is exact and cheap."""
    names = list(session_sizes.index)
    total = session_sizes.sum()
    best = None
    for r in range(1, len(names)):  # proper subset: >=1 session stays in train
        for combo in itertools.combinations(names, r):
            frac = session_sizes[list(combo)].sum() / total
            score = abs(frac - target)
            if best is None or score < best[0]:
                best = (score, combo, frac)
    return best[1], best[2]


def session_split(df: pd.DataFrame) -> tuple:
    test_mask = pd.Series(False, index=df.index)
    chosen = {}
    for label, grp in df.groupby(LABEL_COLUMN):
        sizes = grp.groupby(SESSION_COLUMN).size()
        if len(sizes) < 2:
            raise ValueError(f"Class {label} has only {len(sizes)} session(s); "
                             "session protocol impossible.")
        combo, frac = choose_test_sessions(sizes, TEST_RATIO)
        chosen[int(label)] = {"test_sessions": list(combo), "test_fraction": round(float(frac), 4)}
        test_mask |= df[SESSION_COLUMN].isin(combo) & (df[LABEL_COLUMN] == label)
    return df[~test_mask], df[test_mask], chosen


def random_split(df: pd.DataFrame) -> tuple:
    train_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, random_state=SEED, shuffle=True,
        stratify=df[LABEL_COLUMN],
    )
    return train_df, test_df, {}


def verify(train_df: pd.DataFrame, test_df: pd.DataFrame, protocol: str) -> None:
    feature_cols = [c for c in train_df.columns if c not in [LABEL_COLUMN] + METADATA_COLUMNS]

    train_keys = set(map(tuple, train_df[feature_cols].values))
    n_overlap = sum(tuple(r) in train_keys for r in test_df[feature_cols].values)
    assert n_overlap == 0, f"{n_overlap} test rows are feature-identical to training rows"

    tr_classes = set(train_df[LABEL_COLUMN].unique())
    te_classes = set(test_df[LABEL_COLUMN].unique())
    assert tr_classes == te_classes, f"class mismatch: train {tr_classes} vs test {te_classes}"

    if protocol == "session":
        overlap = set(train_df[SESSION_COLUMN]) & set(test_df[SESSION_COLUMN])
        assert not overlap, f"sessions leak across the split: {overlap}"

    print("[verify] zero cross-split duplicates, all classes on both sides"
          + (", sessions disjoint" if protocol == "session" else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test split with leakage guardrails")
    parser.add_argument("--protocol", choices=["session", "random"], default="session")
    parser.add_argument("--input", default=MERGED_FILE)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for col in [LABEL_COLUMN, SESSION_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"{args.input} lacks '{col}' - re-run clean_data.py (step 1) first.")

    if args.protocol == "session":
        train_df, test_df, chosen = session_split(df)
    else:
        train_df, test_df, chosen = random_split(df)

    verify(train_df, test_df, args.protocol)

    train_df.to_csv(SPLIT62_TRAIN, index=False)
    test_df.to_csv(SPLIT62_TEST, index=False)

    per_class = pd.DataFrame({
        "train": train_df[LABEL_COLUMN].value_counts().sort_index(),
        "test": test_df[LABEL_COLUMN].value_counts().sort_index(),
    })
    per_class["test_frac"] = (per_class["test"] / per_class.sum(axis=1)).round(3)
    print(f"\n[{args.protocol}] per-class counts:\n{per_class.to_string()}")

    manifest = {
        "protocol": args.protocol,
        "seed": SEED,
        "test_ratio_target": TEST_RATIO,
        "input": os.path.abspath(args.input),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "per_class": {str(k): {"train": int(per_class.loc[k, "train"]),
                               "test": int(per_class.loc[k, "test"]),
                               "test_frac": float(per_class.loc[k, "test_frac"])}
                      for k in per_class.index},
        "session_choices": chosen,
    }
    with open(SPLIT_MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[saved] {SPLIT62_TRAIN} ({len(train_df)} rows)")
    print(f"[saved] {SPLIT62_TEST} ({len(test_df)} rows)")
    print(f"[saved] {SPLIT_MANIFEST_FILE}")


if __name__ == "__main__":
    main()
