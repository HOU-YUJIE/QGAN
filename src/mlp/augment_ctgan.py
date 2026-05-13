import pandas as pd
import numpy as np
from ctgan import CTGAN
import warnings
import os

warnings.filterwarnings("ignore")

# 1. Configuration
INPUT_TRAIN_FILE = './data/processed/selected_features_train.csv'
OUTPUT_BALANCED_FILE = './outputs/synthetic_data/Train_Balanced_CTGAN.csv'
TARGET_TOTAL_SAMPLES = 2000

DISCRETE_COLUMNS = ['fwd_psh_flags', 'psh_flag_cnt']

print(">>>Loading real training data...")
df_train = pd.read_csv(INPUT_TRAIN_FILE)

# 2. CTGAN 
balanced_dataframes = []

for category in range(3, 7):
    df_cat_real = df_train[df_train['Label'] == category].copy()
    current_count = len(df_cat_real)
    
    print(f"Processing category: {category} | current samples: {current_count}")

    # If enough data, perform random undersampling
    if current_count >= TARGET_TOTAL_SAMPLES:
        print(f"Enough samples, undersampling to {TARGET_TOTAL_SAMPLES}...")
        df_cat_sampled = df_cat_real.sample(n=TARGET_TOTAL_SAMPLES, random_state=42)
        balanced_dataframes.append(df_cat_sampled)
        continue
        
    # Not enough data; compute how many to generate
    samples_to_generate = TARGET_TOTAL_SAMPLES - current_count
    print(f"Insufficient samples, using CTGAN to generate {samples_to_generate} synthetic samples...")
    
    train_features = df_cat_real.drop(columns=['Label']).copy()
    
    valid_discrete_cols = [c for c in DISCRETE_COLUMNS if c in train_features.columns]
    
    for col in valid_discrete_cols:
        train_features[col] = train_features[col].astype(str)

    ctgan = CTGAN(
        embedding_dim=128,
        generator_dim=(256, 256, 256),
        discriminator_dim=(256, 256, 256),
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        batch_size=500,
        epochs=300,
        verbose=False  
    )
    
    print(f"   -> Training CTGAN model (Epochs: 300)...")
    ctgan.fit(train_features, valid_discrete_cols)
    
    print(f"   -> Sampling synthetic data...")
    df_synthetic = ctgan.sample(samples_to_generate)
    
    # Attach label
    df_synthetic['Label'] = category
    
    # Combine real and synthetic data for this category
    df_combined_category = pd.concat([df_cat_real, df_synthetic], ignore_index=True)
    balanced_dataframes.append(df_combined_category)
    print(f"Category {category} augmented; now has {len(df_combined_category)} samples.")


print("\n>>> Aggregating all categories into balanced dataset...")
df_final_balanced = pd.concat(balanced_dataframes, ignore_index=True)

df_final_balanced = df_final_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"--- Final augmented training set distribution (total {len(df_final_balanced)} samples) ---")
print(df_final_balanced['Label'].value_counts())

df_final_balanced.to_csv(OUTPUT_BALANCED_FILE, index=False)
print(f"CTGAN-augmented training set saved to: {OUTPUT_BALANCED_FILE}")