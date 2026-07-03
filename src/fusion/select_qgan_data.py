import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd

from src.config import (
    FEATURES16_TRAIN,
    BALANCED_QGAN_FILE,
    NUM_CLASSES,
    TARGET_TOTAL_SAMPLES,
    LABEL_COLUMN,
    SEED,
    get_majority_labels,
    qgan_synthetic_file,
    assert_feature_frame,
)

# 2. Core merging workflow

print(" Building mixed real+synthetic QGAN balanced training set...")

balanced_dfs = []

# Read full real training set
try:
    df_train_real = pd.read_csv(FEATURES16_TRAIN)
    assert_feature_frame(df_train_real, where=FEATURES16_TRAIN)
except Exception as e:
    print(f"Unable to read real training file {FEATURES16_TRAIN}: {e}")
    exit(1)

majority_labels = get_majority_labels(FEATURES16_TRAIN)

for category in range(NUM_CLASSES):
    # Extract current category
    df_real_cat = df_train_real[df_train_real[LABEL_COLUMN] == category].copy()
    num_real = len(df_real_cat)
    
    # Majority classes stay purely real data; only cap them to the target size.
    if category in majority_labels:
        df_cat_final = df_real_cat.sample(n=TARGET_TOTAL_SAMPLES, random_state=SEED)
        print(f"Category {category}: purely real data, undersampled from {num_real} to {TARGET_TOTAL_SAMPLES}.")
        balanced_dfs.append(df_cat_final)
        continue

    # If a minority class already exceeds the target size, keep only real data.
    if num_real >= TARGET_TOTAL_SAMPLES:
        df_cat_final = df_real_cat.sample(n=TARGET_TOTAL_SAMPLES, random_state=SEED)
        print(f"Category {category}: purely real data, undersampled from {num_real} to {TARGET_TOTAL_SAMPLES}.")
        balanced_dfs.append(df_cat_final)
        continue

    # Otherwise, supplement with synthetic data to reach the target size.
    qgan_file_path = qgan_synthetic_file(category)
    
    if os.path.exists(qgan_file_path):
        # Read synthetic data for this category
        df_syn_cat = pd.read_csv(qgan_file_path)
        df_syn_cat[LABEL_COLUMN] = category  # Ensure correct label
        num_syn = len(df_syn_cat)
        
        # Concatenate real data with synthetic data for this category
        df_cat_combined = pd.concat([df_real_cat, df_syn_cat], ignore_index=True)
        
        # Truncate to target size if generation produced extra samples
        if len(df_cat_combined) > TARGET_TOTAL_SAMPLES:
            df_cat_combined = df_cat_combined.sample(n=TARGET_TOTAL_SAMPLES, random_state=SEED)

        print(f"Category {category}: real ({num_real}) + synthetic ({num_syn}) -> merged to {len(df_cat_combined)} samples.")
        balanced_dfs.append(df_cat_combined)
    else:
        print(f"Category {category}: synthetic file not found, keeping only {num_real} real samples.")
        balanced_dfs.append(df_real_cat)


# 3. Shuffling
print("\n Aggregating and shuffling the merged dataset...")
df_final_balanced = pd.concat(balanced_dfs, ignore_index=True)

# Shuffle thoroughly to avoid MLP overfitting to class order
df_final_balanced = df_final_balanced.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# 4. Final check and save
print(f"Total samples: {len(df_final_balanced)}")
print("\nFinal class distribution:")
print(df_final_balanced[LABEL_COLUMN].value_counts().sort_index())

# Save file
os.makedirs(os.path.dirname(BALANCED_QGAN_FILE), exist_ok=True)
df_final_balanced.to_csv(BALANCED_QGAN_FILE, index=False)
print(f"\nBalanced training set saved to: {BALANCED_QGAN_FILE}")