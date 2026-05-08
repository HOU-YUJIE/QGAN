import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
import torch.autograd as autograd
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_FILE = os.path.join(BASE_DIR, "selected_features_class0_sample2000.csv")

N_QUBITS = 16          
N_LAYERS = 3           
BATCH_SIZE = 32
LAMBDA_GP = 10
N_CRITIC = 2

TARGET_LABELS = [0]

dev = qml.device("lightning.qubit", wires=N_QUBITS)
#dev = qml.device("lightning.gpu", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def qgan_circuit(inputs, weights):
    # inputs: [batch_size, 16], weights: [N_LAYERS, 16]
    
    num_wires = list(range(14))  # 0~13: numerical feature register
    cat_wires = [14, 15]         # 14~15: categorical flag register

    for l in range(N_LAYERS):

        # Subsystem 1: Numerical feature register (Wires: 0 to 13)

        # Data re-injection: re-input first 14 continuous dimensions each layer
        qml.AngleEmbedding(inputs[:, :14], wires=num_wires)

        # Continuous space exploration (consume first 14 weight parameters)
        for i in num_wires:
            qml.RY(weights[l, i], wires=i)

        # Internal ring entanglement among continuous features
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

class WGAN_Critic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def compute_gradient_penalty(D, real_samples, synthetic_samples, device):
    alpha = torch.rand((real_samples.size(0), 1)).to(device) 
    interpolates = (alpha * real_samples + ((1 - alpha) * synthetic_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradient_targets = torch.ones((real_samples.size(0), 1), requires_grad=False).to(device) 
    gradients = autograd.grad(
        outputs=d_interpolates, inputs=interpolates,
        grad_outputs=gradient_targets, create_graph=True,
        retain_graph=True, only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def main():

    device = torch.device("cpu")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.init()

    feature_columns = [
        # Numerical Register
        'pkt_len_mean', 'fwd_pkt_len_mean', 'bwd_pkt_len_mean', 
        'fwd_pkt_len_max', 'bwd_pkt_len_max', 'pkt_len_min', 'fwd_pkt_len_min',
        'pkt_len_var', 'pkt_len_std', 'fwd_byts_b_avg',
        'flow_byts_s', 'flow_pkts_s', 'init_bwd_win_byts', 'init_fwd_win_byts',
        
        # Categorical Register 
        'fwd_psh_flags', 'psh_flag_cnt'
    ]

    df = pd.read_csv(RAW_FILE)
    print(f"Dataset loaded: {RAW_FILE}")

    for target_label in TARGET_LABELS:
        output_dir = os.path.join(BASE_DIR, str(target_label))
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nLabel = {target_label}...")
        df_target = df[df['Label'] == target_label].copy()
        X_raw = df_target[feature_columns].values
        print(f" real data: {X_raw.shape[0]} ")

        X_log = np.log1p(X_raw)

        local_scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_scaled = local_scaler.fit_transform(X_log)
        
        X_train = torch.FloatTensor(X_scaled)
        loader_gan = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        gen = TabularQuantumGenerator(N_QUBITS, N_LAYERS).to(device)
        crit = WGAN_Critic(N_QUBITS).to(device)
        
        opt_g = optim.Adam(gen.parameters(), lr=0.015, betas=(0.0, 0.9)) 
        opt_d = optim.Adam(crit.parameters(), lr=0.001, betas=(0.0, 0.9))
        N_CRITIC_LOCAL = 2 

        print("starting training...")

        TOTAL_EPOCHS = 50 
        epoch_log = []
        
        for epoch in range(TOTAL_EPOCHS):
            loss_d_val = 0
            loss_g_val = 0
            
            for i, (real_batch,) in enumerate(loader_gan):
                batch_size = real_batch.size(0)
                real_batch = real_batch.to(device)
                
                opt_d.zero_grad()
                noise = (torch.rand(batch_size, N_QUBITS) * np.pi).to(device)
                with torch.no_grad():
                    synthetic_batch = gen(noise)
                    
                loss_d = torch.mean(crit(synthetic_batch)) - torch.mean(crit(real_batch)) + \
                         LAMBDA_GP * compute_gradient_penalty(crit, real_batch, synthetic_batch, device)
                loss_d.backward()
                opt_d.step()
                loss_d_val = loss_d.item()

                if (i + 1) % N_CRITIC_LOCAL == 0:
                    opt_g.zero_grad()
                    noise = (torch.rand(batch_size, N_QUBITS) * np.pi).to(device)
                    loss_g = -torch.mean(crit(gen(noise)))
                    loss_g.backward()
                    opt_g.step()
                    loss_g_val = loss_g.item()

            epoch_log.append((epoch + 1, loss_d_val, loss_g_val))
            print(f"Epoch [{epoch + 1:03d}/{TOTAL_EPOCHS}] | D Loss: {loss_d_val:.4f} | G Loss: {loss_g_val:.4f}")
        
        torch.save(gen.state_dict(), os.path.join(output_dir, 'qgan_generator_weights.pth'))
        joblib.dump(local_scaler, os.path.join(output_dir, 'qgan_local_scaler.pkl'))

        report_path = os.path.join(output_dir, 'report.txt')
        with open(report_path, 'w', encoding='utf-8') as report_file:
            report_file.write(f"Target label: {target_label}\n")
            report_file.write(f"Source file: {RAW_FILE}\n")
            report_file.write(f"Epochs: {TOTAL_EPOCHS}\n")
            report_file.write(f"Batch size: {BATCH_SIZE}\n")
            report_file.write("Last epoch losses:\n")
            report_file.write(f"D Loss: {loss_d_val:.6f}\n")
            report_file.write(f"G Loss: {loss_g_val:.6f}\n")
            report_file.write("\nEpoch log:\n")
            for epoch_index, d_loss, g_loss in epoch_log:
                report_file.write(f"{epoch_index:03d}\t{d_loss:.6f}\t{g_loss:.6f}\n")

        print(f"Saved training results for label {target_label} to: {output_dir}")

if __name__ == "__main__":
    main()