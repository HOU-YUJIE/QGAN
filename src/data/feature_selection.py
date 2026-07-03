"""Feature selection on the TRAIN split only (no leakage).

Step-2 companion patch. The selection METHOD is unchanged from
impg@735bdfc (Pearson |corr|>0.90 prefilter on train, then RF importance
aggregated over stratified CV folds; mutual_info as alternative).

What changed:
  * Inputs/outputs come from src/config.py: reads split62_train/test.csv,
    writes features25_train/test.csv (+ selected_features_25.json,
    feature_importances.csv). Never overwrites its inputs.
  * Metadata columns (session_id, Label_Name) are excluded from the
    feature matrix - they exist for grouping/auditing only.
  * The internal "split the merged file myself" fallback is REMOVED:
    splitting is split_dataset.py's job (it enforces the session protocol
    and leakage assertions); a second split path here would bypass both.

Known step-3 TODO (deliberately not changed now): the Pearson prefilter
misses nonlinear redundancy such as pkt_len_std = sqrt(pkt_len_var).

Usage:
    python src/data/feature_selection.py [--top-k 25] [--method rf]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold

from src.config import (
    FEATURE_IMPORTANCES_FILE,
    MANUAL_FEATURES_16,
    FEATURES25_TEST,
    FEATURES25_TRAIN,
    LABEL_COLUMN,
    METADATA_COLUMNS,
    SEED,
    SELECTED25_JSON,
    SPLIT62_TEST,
    SPLIT62_TRAIN,
)

CORR_THRESHOLD = 0.90
TOP_K_FEATURES = 25


def aggregate_rf_importances(X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    importances = np.zeros((n_splits, X.shape[1]))
    for i, (tr_idx, _) in enumerate(skf.split(X, y)):
        rf = RandomForestClassifier(n_estimators=200, random_state=seed + i,
                                    n_jobs=-1, class_weight="balanced")
        rf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        importances[i, :] = rf.feature_importances_
    return importances.mean(axis=0), importances.std(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only feature selection")
    parser.add_argument("--top-k", type=int, default=TOP_K_FEATURES)
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD)
    parser.add_argument("--corr-method", choices=["spearman", "pearson"], default="spearman",
                        help="spearman catches monotone nonlinear redundancy (e.g. std vs var)")
    parser.add_argument("--method", choices=["rf", "mutual_info"], default="rf")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    train_df = pd.read_csv(SPLIT62_TRAIN)
    test_df = pd.read_csv(SPLIT62_TEST)

    non_feature = [LABEL_COLUMN] + METADATA_COLUMNS
    X_train = train_df.drop(columns=[c for c in non_feature if c in train_df.columns])
    y_train = train_df[LABEL_COLUMN]

    # --- correlation prefilter, TRAIN only (method frozen: Pearson) -------
    corr = X_train.corr(method=args.corr_method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > args.corr_threshold)]
    if to_drop:
        print(f"[prefilter] dropping {len(to_drop)} correlated features "
              f"(|{args.corr_method}| > {args.corr_threshold}): {to_drop}")
    X_filt = X_train.drop(columns=to_drop)

    # --- importance ranking, TRAIN only ------------------------------------
    if args.method == "rf":
        mean_imp, std_imp = aggregate_rf_importances(X_filt, y_train, args.cv_splits, args.seed)
    else:
        mean_imp = mutual_info_classif(X_filt.fillna(0), y_train, random_state=args.seed)
        std_imp = np.zeros_like(mean_imp)

    fi = (pd.DataFrame({"Feature": X_filt.columns,
                        "ImportanceMean": mean_imp, "ImportanceStd": std_imp})
          .sort_values("ImportanceMean", ascending=False))
    top = fi["Feature"].head(args.top_k).tolist()
    print(f"[selected top-{args.top_k}]\n{fi.head(args.top_k).to_string(index=False)}")

    # Output columns = top-k UNION the manual 16. The manual stage must never
    # find its columns physically missing; picks outside the automated top-k
    # are surfaced as warnings there, and the decision stays with the human.
    extra = [m for m in MANUAL_FEATURES_16 if m not in top and m in train_df.columns]
    if extra:
        ranks = {f: i + 1 for i, f in enumerate(fi["Feature"])}
        print(f"[union] appending manual features outside top-{args.top_k}: "
              f"{[(m, f'rank {ranks.get(m)}') for m in extra]}")
    cols = top + extra

    # --- apply the SAME columns to both splits ------------------------------
    for src_df, out_path in ((train_df, FEATURES25_TRAIN), (test_df, FEATURES25_TEST)):
        out = src_df[cols + [LABEL_COLUMN]]
        out.to_csv(out_path, index=False)
        print(f"[saved] {out_path} ({len(out)} rows, {len(cols)} features)")

    fi.to_csv(FEATURE_IMPORTANCES_FILE, index=False)
    with open(SELECTED25_JSON, "w", encoding="utf-8") as f:
        json.dump(top, f, ensure_ascii=False, indent=2)
    print(f"[saved] {FEATURE_IMPORTANCES_FILE}\n[saved] {SELECTED25_JSON}")


if __name__ == "__main__":
    main()
