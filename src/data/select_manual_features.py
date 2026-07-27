"""Apply the MANUAL 16-feature choice to the automated top-25 files.

Replaces data25to16.py. Differences:
  * Reads features25_train/test.csv and writes NEW files
    (features16_train/test.csv) - never overwrites its input, so every
    pipeline stage stays rerunnable in any order.
  * The 16 names live in src/config.py (MANUAL_FEATURES_16), versioned in
    code, not buried in a script.
  * Validates that every manual feature (a) exists in the input and
    (b) appears in the automated top-25 list - if you hand-picked something
    the importance ranking never surfaced, you get a loud warning instead of
    silent inclusion.
  * Writes columns in MANUAL_FEATURES_16 order, because column order defines
    the qubit wire assignment downstream.

Usage:
    python src/data/select_manual_features.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src.config import (
    FEATURES16_TEST,
    FEATURES16_TRAIN,
    FEATURES25_TEST,
    FEATURES25_TRAIN,
    LABEL_COLUMN,
    MANUAL_FEATURES_16,
    SELECTED25_JSON,
    assert_feature_frame,
    load_selected_25,
)


def filter_file(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path)

    missing = [c for c in MANUAL_FEATURES_16 if c not in df.columns]
    if missing:
        raise ValueError(
            f"{in_path} is missing manual features {missing}. "
            "Did feature_selection.py run with a different top-k or data?"
        )
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"{in_path} has no '{LABEL_COLUMN}' column.")

    out = df[MANUAL_FEATURES_16 + [LABEL_COLUMN]]
    assert_feature_frame(out, where=out_path)
    out.to_csv(out_path, index=False)

    dropped = [c for c in df.columns if c not in MANUAL_FEATURES_16 + [LABEL_COLUMN]]
    print(f"[OK] {os.path.basename(in_path)} -> {os.path.basename(out_path)} "
          f"({len(out)} rows, kept {len(MANUAL_FEATURES_16)}, dropped {len(dropped)}: {dropped})")


def main() -> None:
    # Cross-check the manual choice against the automated ranking.
    if os.path.exists(SELECTED25_JSON):
        top25 = set(load_selected_25())
        outside = [c for c in MANUAL_FEATURES_16 if c not in top25]
        if outside:
            print(f"[WARN] manual features NOT in the automated top-25: {outside}\n"
                  f"       They will still be used, but there is no importance "
                  f"evidence for them - document the rationale.")
    else:
        print(f"[WARN] {SELECTED25_JSON} not found; skipping top-25 cross-check.")

    filter_file(FEATURES25_TRAIN, FEATURES16_TRAIN)
    filter_file(FEATURES25_TEST, FEATURES16_TEST)


if __name__ == "__main__":
    main()
