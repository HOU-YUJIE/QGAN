import pennylane as qml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
import sys


# 0. Global configuration

CATEGORY = int(sys.argv[1]) if len(sys.argv) > 1 else 0

MODEL_FILE = f'./{CATEGORY}/qgan_generator_weights.pth'
SCALER_FILE = f'./{CATEGORY}/qgan_local_scaler.pkl'
OUTPUT_FILE = f'./{CATEGORY}/Synthetic_Traffic_16dim.csv'

# Set target data volume
TARGET_TOTAL_SAMPLES = 2000
# Real data path for calculating samples to generate
REAL_DATA_FILE = './selected_features_train.csv'

N_QUBITS = 16
N_LAYERS = 3
TARGET_LABEL = CATEGORY

device = torch.device("cpu")
print(f"Current generation device: {device}")

# Load PennyLane simulator
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
except Exception:
    dev = qml.device("default.qubit", wires=N_QUBITS)


# 1. Rebuild model architecture
@qml.qnode(dev, interface="torch", diff_method="adjoint")
def qgan_circuit(inputs, weights):
    # inputs: [batch_size, 16], weights: [N_LAYERS, 16]
    
    # Define wire groups
    num_wires = list(range(14))  # 0~13: numerical feature register
    cat_wires = [14, 15]         # 14~15: categorical flag register

    for l in range(N_LAYERS):
        # Subsystem 1: Numerical feature register (Wires: 0 to 13)

        # Data re-injection: re-input first 14 continuous dimensions each layer
        qml.AngleEmbedding(inputs[:, :14], wires=num_wires)

        # Continuous space exploration (consume first 14 weight parameters)
        for i in num_wires:
            qml.RY(weights[l, i], wires=i)

        # Ring entanglement among continuous features (parameter-free CZ)
        for i in range(13):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[13, 0])

        # Subsystem 2: Discrete flag register (Wires: 14, 15)
        # Do not use AngleEmbedding to preserve discrete boundaries
        # Inject via RZ gates for phase perturbation only
        qml.RZ(inputs[:, 14], wires=14)
        qml.RZ(inputs[:, 15], wires=15)

        # fwd_psh_flags (Wire 14) as primary control (consume 15th weight)
        qml.RY(weights[l, 14], wires=14)

        # psh_flag_cnt (Wire 15) as dependent node (consume 16th weight)
        # Constraint: Wire 15 transitions only when Wire 14 activates
        qml.CRY(weights[l, 15], wires=[14, 15])

        # Subsystem 3: Cross-register information interference (Numerical -> Categorical)
        # Establish causal entanglement between continuous flow metrics and discrete flags
        # Use deterministic CNOT to avoid gradient explosion
        qml.CNOT(wires=[13, 14])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
class TabularQuantumGenerator(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(qgan_circuit, weight_shapes)

    def forward(self, x):
        out = self.q_layer(x)
        out_mapped = (out + 1.0) * (np.pi / 2.0)
        return out_mapped

# 2. Calculate how many samples to generate
def calculate_samples_needed(category, target_total):
    try:
        # Read real dataset
        df_real = pd.read_csv(REAL_DATA_FILE)
        real_count = len(df_real[df_real['Label'] == category])
        
        samples_to_generate = max(0, target_total - real_count)
        
        print(f"--- Generation plan for category {category} ---")
        print(f"Real samples: {real_count}")
        print(f"Target total: {target_total}")
        print(f"To generate: {samples_to_generate}")
        print("-------------------------------")
        
        return samples_to_generate
    except Exception as e:
        print(f"Unable to read real dataset {REAL_DATA_FILE}, defaulting to generate 1000 samples. Error: {e}")
        return 1000

# 2. Generation workflow

def generate_data(num_samples):
    if num_samples <= 0:
        print(f">>> Category {TARGET_LABEL} already has >= {TARGET_TOTAL_SAMPLES} real samples; no generation needed.")
        return

    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        raise FileNotFoundError(f"Model or scaler file not found: {MODEL_FILE} or {SCALER_FILE}.")

    print(">>> Loading QGAN generator and preprocessor...")
    # Initialize empty skeleton and move to device
    gen = TabularQuantumGenerator(N_QUBITS, N_LAYERS).to(device)
    
    # Load trained weights
    gen.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    gen.eval()  # Switch to evaluation mode
    
    # Load scaler
    local_scaler = joblib.load(SCALER_FILE)
    
    feature_columns = [
        # Numerical Register
        'pkt_len_mean', 'fwd_pkt_len_mean', 'bwd_pkt_len_mean', 
        'fwd_pkt_len_max', 'bwd_pkt_len_max', 'pkt_len_min', 'fwd_pkt_len_min',
        'pkt_len_var', 'pkt_len_std', 'fwd_byts_b_avg',
        'flow_byts_s', 'flow_pkts_s', 'init_bwd_win_byts', 'init_fwd_win_byts',
        
        # Categorical Register 
        'fwd_psh_flags', 'psh_flag_cnt'
    ]

    print(f">>> Generating {num_samples} samples using quantum circuit...")
    with torch.no_grad():
        noise = (torch.rand(num_samples, N_QUBITS) * np.pi).to(device)
        synthetic_data_pi = gen(noise).cpu().numpy()

    print(">>> Performing inverse transform (Inverse Log1p & Scaler)...")
    synthetic_data_log = local_scaler.inverse_transform(synthetic_data_pi)
    synthetic_data_real_scale = np.expm1(synthetic_data_log)

    df_synthetic = pd.DataFrame(synthetic_data_real_scale, columns=feature_columns)
    df_synthetic['Label'] = TARGET_LABEL
    

    # # Physical and protocol constraints (Perfected Protocol Bounds)
    # print(">>> Executing strict physical truncation based on TCP/IP protocol specs...")

    # # 1. Force discrete flags to integers (must run first as the basis for causal masks)
    # flag_features = ['fwd_psh_flags', 'psh_flag_cnt']
    # for ff in flag_features:
    #     if ff in df_synthetic.columns:
    #         df_synthetic[ff] = np.round(df_synthetic[ff]).astype(int)

    # # 2. Global packet-length baseline guardrail (fix logical coverage bug)
    # # Before any size comparisons, enforce that all length features are at least the Ethernet minimum of 54 bytes
    # length_features = ['pkt_len_min', 'fwd_pkt_len_min', 'fwd_pkt_len_max', 'bwd_pkt_len_max', 'pkt_len_mean', 'fwd_pkt_len_mean', 'bwd_pkt_len_mean']
    # for lf in length_features:
    #     if lf in df_synthetic.columns:
    #         df_synthetic.loc[df_synthetic[lf] < 54.0, lf] = 54.0

    # # 3. Enhanced zero-packet attraction (Widened Gravity Well)
    # # Fix the issue where fwd_pkt_len_min gets stuck at 0.93
    # min_features = ['pkt_len_min', 'fwd_pkt_len_min']
    # for mf in min_features:
    #     if mf in df_synthetic.columns:
    #         # Any tiny noise below 150 bytes is physically mapped to 54-byte control packets
    #         df_synthetic.loc[df_synthetic[mf] < 150.0, mf] = 54.0

    # # 4. MTU upper-bound constraint
    # max_features = ['fwd_pkt_len_max', 'bwd_pkt_len_max']
    # for max_f in max_features:
    #     if max_f in df_synthetic.columns:
    #         df_synthetic.loc[df_synthetic[max_f] > 1514.0, max_f] = 1514.0

    # # 5. Ultimate causal mask - fix the fwd_byts_b_avg issue
    # if 'fwd_byts_b_avg' in df_synthetic.columns and 'fwd_psh_flags' in df_synthetic.columns:
    #     mask_no_flag = df_synthetic['fwd_psh_flags'] == 0
    #     mask_low_speed = df_synthetic['flow_byts_s'] < 500.0  
    #     df_synthetic.loc[mask_no_flag | mask_low_speed, 'fwd_byts_b_avg'] = 0.0

    # # 6. Statistical logic fixes
    # # Speed and variance cannot be negative
    # for col in ['flow_byts_s', 'flow_pkts_s', 'pkt_len_var', 'pkt_len_std']:
    #     if col in df_synthetic.columns:
    #         df_synthetic.loc[df_synthetic[col] < 0, col] = 0

    # # Ensure min <= mean <= max (already locked to >= 54 above, so it cannot drop below 54)
    # if all(c in df_synthetic.columns for c in ['pkt_len_min', 'bwd_pkt_len_max']):
    #     mask_min_max = df_synthetic['pkt_len_min'] > df_synthetic['bwd_pkt_len_max']
    #     df_synthetic.loc[mask_min_max, 'pkt_len_min'] = df_synthetic.loc[mask_min_max, 'bwd_pkt_len_max']
            
    # if all(c in df_synthetic.columns for c in ['pkt_len_mean', 'bwd_pkt_len_max']):
    #     mask_mean_max = df_synthetic['pkt_len_mean'] > df_synthetic['bwd_pkt_len_max']
    #     df_synthetic.loc[mask_mean_max, 'bwd_pkt_len_max'] = df_synthetic.loc[mask_mean_max, 'pkt_len_mean']

    # Save generated data
    df_synthetic.to_csv(OUTPUT_FILE, index=False)
    print(f"[√] Generated synthetic data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    # Calculate how many samples to generate and run generation
    num_to_generate = calculate_samples_needed(TARGET_LABEL, TARGET_TOTAL_SAMPLES)
    generate_data(num_to_generate)