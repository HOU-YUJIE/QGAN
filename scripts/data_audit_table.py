"""Produce the Phase-1 data audit table: every cleaning and leakage number
cited in the paper/slides, regenerated from raw data in one run.

Outputs (to outputs/results/):
  data_audit_table.csv   per-class: raw -> dropna -> conflict -> dedup -> final
  data_audit_summary.txt headline numbers incl. duplicate-leakage before/after

Leakage definition: fraction of test rows whose full feature vector has a
bit-identical copy in the training rows.
  * BEFORE cleaning: measured on the merged raw data (after inf/NaN handling
    only) under an 80/20 stratified random split, seed 42 - reproduces the
    original 31.6% finding.
  * AFTER cleaning: measured on the actual split62_train/test.csv currently
    on disk (protocol recorded in split_manifest.json), and additionally on
    the 16 model features in features16_*.csv.

Usage:
    python scripts/data_audit_table.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURES16_TEST, FEATURES16_TRAIN, LABEL_COLUMN, MANUAL_FEATURES_16,
    METADATA_COLUMNS, RESULTS_DIR, SEED, SPLIT62_TEST, SPLIT62_TRAIN,
    SPLIT_MANIFEST_FILE,
)
from src.data.clean_data import load_raw


def overlap_rate(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list) -> float:
    train_keys = set(map(tuple, train_df[cols].values))
    hits = sum(tuple(r) in train_keys for r in test_df[cols].values)
    return hits / max(len(test_df), 1)


def main() -> None:
    merged, _ = load_raw()
    non_feature = [LABEL_COLUMN] + METADATA_COLUMNS
    per = lambda df: df["Label_Name"].value_counts()

    raw_counts = per(merged)

    # step 1: inf -> NaN -> dropna (mirrors clean_data.py)
    merged = merged.replace([np.inf, -np.inf], np.nan)
    after_dropna = merged.dropna()
    dropna_removed = (raw_counts - per(after_dropna)).fillna(0).astype(int)

    feature_cols = [c for c in after_dropna.columns if c not in non_feature]
    numeric = after_dropna[feature_cols].select_dtypes(include=[np.number]).columns
    zero_var = [c for c in numeric if after_dropna[c].std() == 0]
    work = after_dropna.drop(columns=zero_var)
    feature_cols = [c for c in feature_cols if c not in zero_var]

    # leakage BEFORE cleaning (random 80/20 stratified, seed 42)
    tr0, te0 = train_test_split(work, train_size=0.8, random_state=SEED,
                                shuffle=True, stratify=work[LABEL_COLUMN])
    leak_before = overlap_rate(tr0, te0, feature_cols)

    # step 2: conflict groups
    nunique = work.groupby(feature_cols, sort=False)[LABEL_COLUMN].transform("nunique")
    conflict_mask = nunique > 1
    n_conflict_groups = work.loc[conflict_mask, feature_cols].drop_duplicates().shape[0]
    conflict_removed = per(work[conflict_mask]).reindex(raw_counts.index).fillna(0).astype(int)
    work = work.loc[~conflict_mask]

    # step 3: exact dedup
    before_dedup = per(work)
    work = work.drop_duplicates(subset=feature_cols, keep="first")
    dedup_removed = (before_dedup - per(work)).fillna(0).astype(int)
    final_counts = per(work).reindex(raw_counts.index).fillna(0).astype(int)

    # leakage AFTER cleaning: actual on-disk split
    protocol = "unknown"
    if os.path.exists(SPLIT_MANIFEST_FILE):
        with open(SPLIT_MANIFEST_FILE, encoding="utf-8") as f:
            protocol = json.load(f).get("protocol", "unknown")
    tr62, te62 = pd.read_csv(SPLIT62_TRAIN), pd.read_csv(SPLIT62_TEST)
    cols62 = [c for c in tr62.columns if c not in non_feature]
    leak_after_62 = overlap_rate(tr62, te62, cols62)
    tr16, te16 = pd.read_csv(FEATURES16_TRAIN), pd.read_csv(FEATURES16_TEST)
    leak_after_16 = overlap_rate(tr16, te16, MANUAL_FEATURES_16)

    # ---- assemble ----------------------------------------------------------
    table = pd.DataFrame({
        "raw": raw_counts,
        "dropna_removed": dropna_removed.reindex(raw_counts.index).fillna(0).astype(int),
        "conflict_removed": conflict_removed,
        "dedup_removed": dedup_removed.reindex(raw_counts.index).fillna(0).astype(int),
        "final": final_counts,
    })
    table["retained_pct"] = (table["final"] / table["raw"] * 100).round(1)
    table.loc["TOTAL"] = table.sum(numeric_only=True)
    table.loc["TOTAL", "retained_pct"] = round(
        table.loc["TOTAL", "final"] / table.loc["TOTAL", "raw"] * 100, 1)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(RESULTS_DIR, "data_audit_table.csv"))

    summary = [
        "PHASE-1 DATA AUDIT (regenerated from raw data)",
        "=" * 60,
        f"raw flows:                {int(table.loc['TOTAL','raw'])}",
        f"zero-variance cols:      {len(zero_var)} dropped ({zero_var})",
        f"label-conflict groups:   {n_conflict_groups} groups, "
        f"{int(table.loc['TOTAL','conflict_removed'])} rows "
        f"({table.loc['TOTAL','conflict_removed']/table.loc['TOTAL','raw']:.1%})",
        f"exact duplicates:        {int(table.loc['TOTAL','dedup_removed'])} rows",
        f"final flows:             {int(table.loc['TOTAL','final'])} "
        f"({table.loc['TOTAL','retained_pct']}% retained)",
        "",
        f"duplicate leakage BEFORE cleaning (random 80/20): {leak_before:.1%}",
        f"duplicate leakage AFTER  cleaning ({protocol} split, 62-dim): {leak_after_62:.2%}",
        f"duplicate leakage AFTER  cleaning ({protocol} split, 16-dim): {leak_after_16:.2%}",
        "",
        "per-class table: data_audit_table.csv",
    ]
    text = "\n".join(summary)
    with open(os.path.join(RESULTS_DIR, "data_audit_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print("\n" + table.to_string())


if __name__ == "__main__":
    main()
