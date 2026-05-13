import os
import glob
import pandas as pd
import numpy as np

import pathlib

# Use the explicit processed MalayaNetwork_GT/csv_output layout as the canonical source
BASE_DIR = os.path.join("data", "processed", "MalayaNetwork_GT", "csv_output")
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(
        f"Expected raw data at {BASE_DIR}. Place MalayaNetwork_GT/csv_output under data/processed/.")

OUTPUT_FILE = os.path.join("data", "processed", "merged_cleaned_dataset.csv")

COLUMNS_TO_DROP = [
    'src_ip',
    'dst_ip',
    'timestamp', 
    'src_port',
    'dst_port',
    'protocol',
]

def merge_and_clean():
    all_dataframes = []
    
    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"BASE_DIR not found: {BASE_DIR}")

    categories = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    print(f"{len(categories)} folders: {categories}")

    for label_idx, category_name in enumerate(categories):
        folder_path = os.path.join(BASE_DIR, category_name)
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        print(f"Processing {category_name} (Label: {label_idx}), contains {len(csv_files)} files...")
        
        for file in csv_files:
            try:
                df = pd.read_csv(file, low_memory=False)
                
                df.columns = df.columns.str.strip()
                
                cols_to_drop_actual = [col for col in COLUMNS_TO_DROP if col in df.columns]
                df.drop(columns=cols_to_drop_actual, inplace=True)
                
                df['Label_ID'] = label_idx
                df['Label_Name'] = category_name
                
                all_dataframes.append(df)
            except Exception as e:
                print(f"reading {file} error: {e}")


    merged_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"dim: {merged_df.shape}")

    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    rows_before = len(merged_df)
    
    merged_df.dropna(inplace=True)
    
    rows_after = len(merged_df)
    print(f"dim {merged_df.shape}")


    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    zero_variance_cols = [col for col in numeric_cols if merged_df[col].std() == 0]
    if zero_variance_cols:
        print(zero_variance_cols)
        # 'fwd_seg_size_min', 'fwd_urg_flags', 'bwd_urg_flags', 'urg_flag_cnt', 'ece_flag_cnt', 'active_max', 'active_min', 'active_mean', 'active_std', 'idle_max', 'idle_min', 'idle_mean', 'idle_std', 'cwr_flag_count'
        merged_df.drop(columns=zero_variance_cols, inplace=True)

    # 64dim
    # Ensure output directory exists
    out_dir = os.path.dirname(OUTPUT_FILE)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    merged_df.to_csv(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    merge_and_clean()