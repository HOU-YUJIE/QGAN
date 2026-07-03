"""Train the QGAN (WGAN-GP) for one class.

Engineering pass only - training behavior is identical to impg@735bdfc:
same architecture (via circuit.py), same hyperparameters (via config.py),
same preprocessing (log1p -> MinMax to [0, pi] per class).

What changed:
  * Circuit and all constants imported from shared modules (no local copies).
  * Seeded (per-class offset so classes are not identical runs).
  * Epoch log now records MEAN losses over the epoch; the previous log kept
    only the last batch's instantaneous loss, which made the curves noise.

Usage:
    python src/qgan/train.py <category 0-9> [--seed 42]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    ADAM_BETAS,
    BATCH_SIZE,
    FEATURES16_TRAIN,
    LABEL_COLUMN,
    LAMBDA_GP,
    LR_CRITIC,
    LR_GENERATOR,
    MANUAL_FEATURES_16,
    N_CRITIC,
    N_QUBITS,
    SEED,
    TOTAL_EPOCHS,
    assert_feature_frame,
    qgan_model_dir,
    set_seed,
)
from src.qgan.circuit import TabularQuantumGenerator, sample_noise

class WGANCritic(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def compute_gradient_penalty(critic, real, fake, device):
    alpha = torch.rand((real.size(0), 1), device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = critic(interpolates)
    grad_targets = torch.ones((real.size(0), 1), device=device)
    gradients = autograd.grad(
        outputs=d_interp, inputs=interpolates, grad_outputs=grad_targets,
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0].view(real.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def train_one_class(target_label: int, seed: int) -> None:
    set_seed(seed + target_label)  # per-class deterministic, mutually distinct
    device = torch.device("cpu")

    df = pd.read_csv(FEATURES16_TRAIN)
    assert_feature_frame(df, where=FEATURES16_TRAIN)
    df_target = df[df[LABEL_COLUMN] == target_label]
    X_raw = df_target[MANUAL_FEATURES_16].values
    print(f"Label {target_label}: {X_raw.shape[0]} real rows from {FEATURES16_TRAIN}")

    # Frozen preprocessing: log1p then per-class MinMax to [0, pi]
    X_log = np.log1p(X_raw)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_log)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_scaled)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )

    gen = TabularQuantumGenerator().to(device)
    crit = WGANCritic(N_QUBITS).to(device)
    opt_g = optim.Adam(gen.parameters(), lr=LR_GENERATOR, betas=ADAM_BETAS)
    opt_d = optim.Adam(crit.parameters(), lr=LR_CRITIC, betas=ADAM_BETAS)

    epoch_log = []
    for epoch in range(TOTAL_EPOCHS):
        d_losses, g_losses = [], []
        for i, (real_batch,) in enumerate(loader):
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            # --- critic step ---
            opt_d.zero_grad()
            with torch.no_grad():
                fake_batch = gen(sample_noise(bs, device))
            loss_d = (crit(fake_batch).mean() - crit(real_batch).mean()
                      + LAMBDA_GP * compute_gradient_penalty(crit, real_batch, fake_batch, device))
            loss_d.backward()
            opt_d.step()
            d_losses.append(loss_d.item())

            # --- generator step every N_CRITIC batches ---
            if (i + 1) % N_CRITIC == 0:
                opt_g.zero_grad()
                loss_g = -crit(gen(sample_noise(bs, device))).mean()
                loss_g.backward()
                opt_g.step()
                g_losses.append(loss_g.item())

        d_mean = float(np.mean(d_losses)) if d_losses else float("nan")
        g_mean = float(np.mean(g_losses)) if g_losses else float("nan")
        epoch_log.append((epoch + 1, d_mean, g_mean))
        print(f"Epoch [{epoch + 1:03d}/{TOTAL_EPOCHS}] | D(mean): {d_mean:.4f} | G(mean): {g_mean:.4f}")

    # --- persist ---
    out_dir = qgan_model_dir(target_label)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(gen.state_dict(), os.path.join(out_dir, "qgan_generator_weights.pth"))
    joblib.dump(scaler, os.path.join(out_dir, "qgan_local_scaler.pkl"))

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Target label: {target_label}\n")
        f.write(f"Source file: {FEATURES16_TRAIN}\n")
        f.write(f"Seed: {seed + target_label}\n")
        f.write(f"Epochs: {TOTAL_EPOCHS}\nBatch size: {BATCH_SIZE}\n")
        f.write("\nEpoch log (epoch, D loss mean, G loss mean):\n")
        for e, d, g in epoch_log:
            f.write(f"{e:03d}\t{d:.6f}\t{g:.6f}\n")

    print(f"Saved model, scaler and report for label {target_label} to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train QGAN for one category")
    parser.add_argument("category", type=int, nargs="?", default=0, help="Target label (0-9)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train_one_class(args.category, args.seed)


if __name__ == "__main__":
    main()
    main()