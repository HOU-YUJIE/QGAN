#!/usr/bin/env python3
"""
Split VQC original dataset into train and test sets by Label,
stratified with default 8:2 ratio.

Default inputs:
  VQC/selected_features_dataset.csv

Default outputs:
  VQC/selected_features_train.csv
  VQC/selected_features_test.csv


"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Split VQC dataset 8:2 by category")
    parser.add_argument(
        "-i",
        "--input",
        default=os.path.join("VQC", "selected_features_dataset.csv"),
        help="Input CSV file path",
    )
    parser.add_argument(
        "-tr",
        "--train-output",
        default=os.path.join("VQC", "selected_features_train.csv"),
        help="Training set output path",
    )
    parser.add_argument(
        "-te",
        "--test-output",
        default=os.path.join("VQC", "selected_features_test.csv"),
        help="Test set output path",
    )
    parser.add_argument(
        "-r",
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio, default 0.8",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed, default 42",
    )
    parser.add_argument(
        "--label-col",
        default="Label",
        help="Label column name, default Label",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if args.label_col not in df.columns:
        raise ValueError(f"Label column not found in input file: {args.label_col}")

    train_df, test_df = train_test_split(
        df,
        train_size=args.train_ratio,
        random_state=args.seed,
        shuffle=True,
        stratify=df[args.label_col],
    )

    train_df = train_df.sort_index()
    test_df = test_df.sort_index()

    os.makedirs(os.path.dirname(args.train_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.test_output) or ".", exist_ok=True)

    train_df.to_csv(args.train_output, index=False)
    test_df.to_csv(args.test_output, index=False)

    print(f"Saved training set: {args.train_output} ({len(train_df)} rows)")
    print(f"Saved test set: {args.test_output} ({len(test_df)} rows)")

    train_counts = train_df[args.label_col].value_counts().sort_index()
    test_counts = test_df[args.label_col].value_counts().sort_index()
    print("Training set class distribution:")
    print(train_counts.to_string())
    print("Test set class distribution:")
    print(test_counts.to_string())


if __name__ == "__main__":
    main()