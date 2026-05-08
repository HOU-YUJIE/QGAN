import pandas as pd
import os


# 1. Define the 16-core features QGAN uses + 1 label dimension
TARGET_COLUMNS = [
    # Continuous features
    'pkt_len_mean', 'fwd_pkt_len_mean', 'bwd_pkt_len_mean', 
    'fwd_pkt_len_max', 'bwd_pkt_len_max', 'pkt_len_min', 'fwd_pkt_len_min',
    'pkt_len_var', 'pkt_len_std', 'fwd_byts_b_avg',
    'flow_byts_s', 'flow_pkts_s', 'init_bwd_win_byts', 'init_fwd_win_byts',
    # Discrete features
    'fwd_psh_flags', 'psh_flag_cnt',
    
    # Critical label column
    'Label'
]

# Files to trim down
FILES_TO_PRUNE = [
    './VQC/selected_features_train.csv',
    './VQC/selected_features_test.csv'
]

print("Starting dataset feature alignment to 16 dimensions...")

for file_path in FILES_TO_PRUNE:
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            original_shape = df.shape
            
            missing_cols = [c for c in TARGET_COLUMNS if c not in df.columns]
            if missing_cols:
                print(f"{file_path} is missing columns: {missing_cols}")
                continue
            
            # Perform truncation
            df_pruned = df[TARGET_COLUMNS]
            
            # Save overwriting original
            df_pruned.to_csv(file_path, index=False)
            
            print(f"Successfully cleaned: {file_path}")
            print(f"   -> Original dimensions: {original_shape[1]-1} features + 1 label")
            print(f"   -> New dimensions: {df_pruned.shape[1]-1} features + 1 label")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    else:
        print(f"File not found: {file_path}.")

print("\nComplete!")