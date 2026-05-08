import pandas as pd
import os

# 1. Paths and global configuration

ORIGINAL_TRAIN_FILE = './VQC/selected_features_train.csv'
OUTPUT_BALANCED_FILE = './VQC/Train_Balanced_QGAN.csv'

NUM_CLASSES = 10
TARGET_SAMPLES = 2000

# 2. Core merging workflow

print(">>> [*] Building mixed real+synthetic QGAN balanced training set...")

balanced_dfs = []

# Read full real training set
try:
    df_train_real = pd.read_csv(ORIGINAL_TRAIN_FILE)
except Exception as e:
    print(f"Unable to read real training file {ORIGINAL_TRAIN_FILE}: {e}")
    exit(1)

for category in range(NUM_CLASSES):
    # Extract current category
    df_real_cat = df_train_real[df_train_real['Label'] == category].copy()
    num_real = len(df_real_cat)
    
    # --- Category 0: majority class, just random undersampling ---
    if category == 0:
        if num_real >= TARGET_SAMPLES:
            df_cat_final = df_real_cat.sample(n=TARGET_SAMPLES, random_state=42)
            print(f"Category 0: purely real data, undersampled from {num_real} to {TARGET_SAMPLES}.")
        else:
            df_cat_final = df_real_cat
            print(f"Category 0: insufficient real data, keeping all {num_real} samples.")
        balanced_dfs.append(df_cat_final)
        continue
        
    # --- Categories 1-9: minority classes, mix real + synthetic ---
    qgan_file_path = f'./VQC/{category}/Synthetic_Traffic_16dim.csv'
    
    if os.path.exists(qgan_file_path):
        # Read synthetic data for this category
        df_syn_cat = pd.read_csv(qgan_file_path)
        df_syn_cat['Label'] = category  # Ensure correct label
        num_syn = len(df_syn_cat)
        
        # Concatenate real data with synthetic data for this category
        df_cat_combined = pd.concat([df_real_cat, df_syn_cat], ignore_index=True)
        
        # Truncate to target size if generation produced extra samples
        if len(df_cat_combined) > TARGET_SAMPLES:
            df_cat_combined = df_cat_combined.sample(n=TARGET_SAMPLES, random_state=42)
            
        print(f"Category {category}: real ({num_real}) + synthetic ({num_syn}) -> merged to {len(df_cat_combined)} samples.")
        balanced_dfs.append(df_cat_combined)


# 3. Shuffling
print("\n>>> Aggregating and shuffling the merged dataset...")
df_final_balanced = pd.concat(balanced_dfs, ignore_index=True)

# Shuffle thoroughly to avoid MLP overfitting to class order
df_final_balanced = df_final_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)


# 4. Final check and save
print(f"Total samples: {len(df_final_balanced)}")
print("\nFinal class distribution:")
print(df_final_balanced['Label'].value_counts().sort_index())

# Save file
df_final_balanced.to_csv(OUTPUT_BALANCED_FILE, index=False)
print(f"\nBalanced training set saved to: {OUTPUT_BALANCED_FILE}")