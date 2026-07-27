"""Merge raw per-session CSVs into one cleaned dataset.

Step-1 rewrite. Kept from the previous version: sorted category order,
persisted label_mapping.json, identifier-column dropping, inf->NaN->dropna,
zero-variance column removal.

New in this version:
  1. session_id column (raw filename stem, e.g. "Bittorent_5G_Nightime1").
     Carried as METADATA - never a model feature - so the split stage can
     group by capture session instead of leaking sessions across train/test.
  2. Label-conflict resolution: feature-identical rows that carry DIFFERENT
     labels are irreducible noise; the entire conflicting group is dropped.
  3. Exact-duplicate removal on the feature columns (keep first occurrence).
     Previously ~32% of a random test split had bit-identical twins in the
     training split; this is the fix at the source.
  4. Per-class accounting for every destructive operation, written to
     data/processed/cleaning_report.txt.

Determinism: categories sorted, files-within-category sorted, dedup keeps
the first occurrence under that fixed order.

Usage:
    python src/data/clean_data.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    LABEL_COLUMN,
    LABEL_MAPPING_FILE,
    MERGED_FILE,
    METADATA_COLUMNS,
    SESSION_COLUMN,
    CLEANING_REPORT_FILE,
)

# Identifier / shortcut columns: must never reach the model.
COLUMNS_TO_DROP = ["src_ip", "dst_ip", "timestamp", "src_port", "dst_port", "protocol"]


def load_raw() -> tuple[pd.DataFrame, dict]:
    if not os.path.exists(DATA_RAW_DIR):
        raise FileNotFoundError(f"Expected raw data at {DATA_RAW_DIR}")

    categories = sorted(d for d in os.listdir(DATA_RAW_DIR)
                        if os.path.isdir(os.path.join(DATA_RAW_DIR, d)))
    label_mapping = {name: idx for idx, name in enumerate(categories)}
    print(f"{len(categories)} categories: {label_mapping}")

    frames = []
    for name in categories:
        files = sorted(glob.glob(os.path.join(DATA_RAW_DIR, name, "*.csv")))
        print(f"  {name} (Label {label_mapping[name]}): {len(files)} session files")
        for path in files:
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
            df[SESSION_COLUMN] = os.path.splitext(os.path.basename(path))[0]
            df[LABEL_COLUMN] = label_mapping[name]
            df["Label_Name"] = name
            frames.append(df)
    return pd.concat(frames, ignore_index=True), label_mapping


def per_class(df: pd.DataFrame) -> pd.Series:
    return df["Label_Name"].value_counts().sort_index()


def main() -> None:
    report_lines = []

    def log(msg: str) -> None:
        print(msg)
        report_lines.append(msg)

    merged, label_mapping = load_raw()
    non_feature = [LABEL_COLUMN] + METADATA_COLUMNS
    log(f"\n[load] merged shape: {merged.shape}")

    # ---- 1. inf -> NaN -> dropna, accounted per class -------------------
    merged = merged.replace([np.inf, -np.inf], np.nan)
    before = per_class(merged)
    merged = merged.dropna()
    dropped = (before - per_class(merged)).fillna(before).astype(int)
    log(f"[dropna] removed {int(dropped.sum())} rows; per class:\n{dropped.to_string()}")

    # ---- 2. zero-variance feature columns --------------------------------
    feature_cols = [c for c in merged.columns if c not in non_feature]
    numeric_cols = merged[feature_cols].select_dtypes(include=[np.number]).columns
    zero_var = [c for c in numeric_cols if merged[c].std() == 0]
    if zero_var:
        merged = merged.drop(columns=zero_var)
        feature_cols = [c for c in feature_cols if c not in zero_var]
        log(f"[zero-variance] dropped {len(zero_var)} cols: {zero_var}")

    # ---- 3. label conflicts: identical features, different labels --------
    # These rows are indistinguishable by construction; drop whole groups.
    label_per_group = merged.groupby(feature_cols, sort=False)[LABEL_COLUMN].transform("nunique")
    conflict_mask = label_per_group > 1
    n_groups = merged.loc[conflict_mask, feature_cols].drop_duplicates().shape[0]
    before = per_class(merged)
    merged = merged.loc[~conflict_mask]
    dropped = (before - per_class(merged)).fillna(before).astype(int)
    log(f"[label-conflict] dropped {int(conflict_mask.sum())} rows in {n_groups} "
        f"conflicting groups; per class:\n{dropped.to_string()}")

    # ---- 4. exact duplicates on feature columns --------------------------
    before = per_class(merged)
    merged = merged.drop_duplicates(subset=feature_cols, keep="first")
    dropped = (before - per_class(merged)).fillna(before).astype(int)
    log(f"[dedup] dropped {int(dropped.sum())} duplicate rows; per class:\n{dropped.to_string()}")

    # ---- 5. invariants ----------------------------------------------------
    assert merged[feature_cols].duplicated().sum() == 0, "dedup failed"
    assert merged[feature_cols].isna().sum().sum() == 0, "NaNs survived cleaning"

    # ---- 6. persist --------------------------------------------------------
    # Column order: features..., session_id, Label, Label_Name
    merged = merged[feature_cols + [SESSION_COLUMN, LABEL_COLUMN, "Label_Name"]]
    os.makedirs(os.path.dirname(MERGED_FILE), exist_ok=True)
    merged.to_csv(MERGED_FILE, index=False)
    with open(LABEL_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    log(f"\n[final] shape: {merged.shape} "
        f"({len(feature_cols)} features + {SESSION_COLUMN} + labels)")
    log(f"[final] rows per class:\n{per_class(merged).to_string()}")
    log(f"[final] sessions per class:\n"
        f"{merged.groupby('Label_Name')[SESSION_COLUMN].nunique().to_string()}")
    log(f"[saved] {MERGED_FILE}")
    log(f"[saved] {LABEL_MAPPING_FILE}")

    with open(CLEANING_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"[saved] {CLEANING_REPORT_FILE}")


if __name__ == "__main__":
    main()
