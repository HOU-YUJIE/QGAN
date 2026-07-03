import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif


DEFAULT_MERGED = "./data/processed/merged_cleaned_dataset.csv"
DEFAULT_OUT_TRAIN = "./data/processed/selected_features_train.csv"
DEFAULT_OUT_TEST = "./data/processed/selected_features_test.csv"
CORR_THRESHOLD = 0.90
TOP_K_FEATURES = 25


def aggregate_rf_importances(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    importances = np.zeros((n_splits, X.shape[1]), dtype=float)

    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        rf = RandomForestClassifier(n_estimators=200, random_state=seed + i, n_jobs=-1, class_weight='balanced')
        rf.fit(X_tr, y_tr)
        importances[i, :] = rf.feature_importances_

    mean_imp = importances.mean(axis=0)
    std_imp = importances.std(axis=0)
    return mean_imp, std_imp


def run_feature_selection(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_train: str,
    out_test: str,
    top_k: int = TOP_K_FEATURES,
    corr_threshold: float = CORR_THRESHOLD,
    seed: int = 42,
    method: str = 'rf',
    n_splits: int = 5,
):
    label_col = 'Label' if 'Label' in train_df.columns else 'Label_ID'

    X_train = train_df.drop(columns=[c for c in ['Label', 'Label_ID', 'Label_Name'] if c in train_df.columns])
    y_train = train_df[label_col]

    # Correlation-based prefilter on TRAIN only
    corr_matrix = X_train.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]

    if to_drop_corr:
        print(f"Dropping {len(to_drop_corr)} correlated features (threshold={corr_threshold})")

    X_train_filtered = X_train.drop(columns=to_drop_corr)

    if method == 'rf':
        mean_imp, std_imp = aggregate_rf_importances(X_train_filtered, y_train, n_splits=n_splits, seed=seed)
        feature_importance_df = pd.DataFrame({'Feature': X_train_filtered.columns, 'ImportanceMean': mean_imp, 'ImportanceStd': std_imp})
        feature_importance_df = feature_importance_df.sort_values(by='ImportanceMean', ascending=False)
    elif method == 'mutual_info':
        mi = mutual_info_classif(X_train_filtered.fillna(0), y_train, random_state=seed)
        feature_importance_df = pd.DataFrame({'Feature': X_train_filtered.columns, 'ImportanceMean': mi, 'ImportanceStd': np.zeros_like(mi)})
        feature_importance_df = feature_importance_df.sort_values(by='ImportanceMean', ascending=False)
    else:
        raise ValueError(f"Unknown method: {method}")

    top_features = feature_importance_df['Feature'].head(top_k).tolist()
    print("Top features (mean importance):")
    print(feature_importance_df.head(top_k).to_string(index=False))

    # Apply to train and test (ensure columns exist in both)
    selected_cols = [f for f in top_features if f in train_df.columns and f in test_df.columns]

    out_train_df = train_df[selected_cols].copy()
    out_test_df = test_df[selected_cols].copy()

    out_train_df['Label'] = train_df[label_col].values
    out_test_df['Label'] = test_df[label_col].values

    os.makedirs(os.path.dirname(out_train) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(out_test) or '.', exist_ok=True)

    out_train_df.to_csv(out_train, index=False)
    out_test_df.to_csv(out_test, index=False)

    fi_out = os.path.join(os.path.dirname(out_train) or '.', "feature_importances.csv")
    feature_importance_df.to_csv(fi_out, index=False)
    sel_out = os.path.join(os.path.dirname(out_train) or '.', "selected_features.json")
    with open(sel_out, 'w') as fh:
        json.dump(selected_cols, fh, ensure_ascii=False, indent=2)

    print(f"Saved selected features train/test: {out_train}, {out_test}")
    print(f"Saved feature importances: {fi_out}")


def main():
    parser = argparse.ArgumentParser(description="Feature selection. Uses TRAIN to select features (no leakage).")
    parser.add_argument('--train-input', default=None, help='Path to training CSV (if omitted, will split merged input)')
    parser.add_argument('--test-input', default=None, help='Path to test CSV (if omitted, will split merged input)')
    parser.add_argument('--merged-input', default=DEFAULT_MERGED, help='Merged cleaned dataset (used if train/test not provided)')
    parser.add_argument('--out-train', default=DEFAULT_OUT_TRAIN)
    parser.add_argument('--out-test', default=DEFAULT_OUT_TEST)
    parser.add_argument('--top-k', type=int, default=TOP_K_FEATURES)
    parser.add_argument('--corr-threshold', type=float, default=CORR_THRESHOLD)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method', choices=['rf', 'mutual_info'], default='rf', help='Feature scoring method')
    parser.add_argument('--cv-splits', type=int, default=5, help='Number of CV splits for RF aggregation')
    args = parser.parse_args()

    if args.train_input and args.test_input:
        train_df = pd.read_csv(args.train_input)
        test_df = pd.read_csv(args.test_input)
    else:
        if not os.path.exists(args.merged_input):
            raise FileNotFoundError(f"Merged input not found: {args.merged_input}")
        merged = pd.read_csv(args.merged_input)
        if 'Label' not in merged.columns and 'Label_ID' in merged.columns:
            merged['Label'] = merged['Label_ID']

        train_df, test_df = train_test_split(merged, train_size=0.8, random_state=args.seed, stratify=merged['Label'])

    run_feature_selection(train_df, test_df, args.out_train, args.out_test, top_k=args.top_k, corr_threshold=args.corr_threshold, seed=args.seed, method=args.method, n_splits=args.cv_splits)


if __name__ == '__main__':
    main()