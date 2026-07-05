"""Train the QGAN (WGAN-GP) for one class, with model selection.

Step-4 version. New relative to the step-0 refactor:
  * --circuit {v1,v2}: v1 = frozen impg architecture, v2 = repaired
    (no dead params, learnable entanglers). See circuit.py.
  * --preproc {log_minmax,quantile}: see preprocessing.py.
  * 15% of the class rows are held out as a GAN-validation slice; every
    EVAL_EVERY epochs the generator is scored critic-independently
    (standardized per-feature Wasserstein + correlation-matrix distance)
    and the BEST checkpoint is kept alongside the last one.
  * model_manifest.json records circuit version, preproc kind, seed, best
    epoch/metric and the feature order - generate.py reads it, so a
    train/generate mismatch is structurally impossible.

Usage:
    python src/qgan/train.py <category> [--circuit v2] [--preproc quantile] [--seed 42]
"""

import argparse
import json
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
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    ADAM_BETAS, BATCH_SIZE, FEATURES16_TRAIN, LABEL_COLUMN, LAMBDA_GP,
    LR_CRITIC, LR_GENERATOR, MANUAL_FEATURES_16, N_CRITIC, N_QUBITS, SEED,
    TOTAL_EPOCHS, assert_feature_frame, qgan_model_dir, set_seed,
)
from src.qgan.circuit import TabularQuantumGenerator, sample_noise
from src.qgan.preprocessing import PREPROC_KINDS, fit_preproc, inverse_preproc

VAL_FRACTION = 0.15
EVAL_EVERY = 5


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
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grads = autograd.grad(
        outputs=critic(interp), inputs=interp,
        grad_outputs=torch.ones((real.size(0), 1), device=device),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0].view(real.size(0), -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def validation_metric(gen, preproc_state, X_val_raw, ref_std, ref_corr) -> dict:
    """Critic-independent generator quality on held-out real rows.

    std_w1   : per-feature Wasserstein-1 (original scale) / real train std
    corr_dist: mean |corr(gen) - corr(real train)| over feature pairs
    score    : std_w1 + corr_dist (lower is better)
    """
    gen.eval()
    with torch.no_grad():
        fake_pi = gen(sample_noise(len(X_val_raw))).numpy()
    gen.train()
    fake_raw = inverse_preproc(preproc_state, fake_pi)

    w1 = np.mean([wasserstein_distance(fake_raw[:, j], X_val_raw[:, j]) / ref_std[j]
                  for j in range(fake_raw.shape[1])])
    with np.errstate(invalid="ignore", divide="ignore"):
        fake_corr = np.nan_to_num(np.corrcoef(fake_raw, rowvar=False))
    corr_dist = float(np.abs(fake_corr - ref_corr).mean())
    return {"std_w1": float(w1), "corr_dist": corr_dist, "score": float(w1) + corr_dist}


def train_one_class(target_label: int, seed: int, circuit: str, preproc: str,
                    epochs: int = TOTAL_EPOCHS, patience: int = 6,
                    out_dir: str = None) -> None:
    set_seed(seed + target_label)
    device = torch.device("cpu")

    df = pd.read_csv(FEATURES16_TRAIN)
    assert_feature_frame(df, where=FEATURES16_TRAIN)
    X_all = df.loc[df[LABEL_COLUMN] == target_label, MANUAL_FEATURES_16].values
    if len(X_all) < 50:
        raise ValueError(f"class {target_label}: only {len(X_all)} rows")

    # GAN-validation holdout (never seen by the critic)
    idx = np.random.permutation(len(X_all))
    n_val = max(30, int(VAL_FRACTION * len(X_all)))
    X_val_raw, X_train_raw = X_all[idx[:n_val]], X_all[idx[n_val:]]
    print(f"Label {target_label}: {len(X_train_raw)} train / {len(X_val_raw)} val rows "
          f"| circuit={circuit} preproc={preproc}")

    preproc_state, X_scaled = fit_preproc(X_train_raw, preproc, seed=seed)
    ref_std = X_train_raw.std(axis=0) + 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        ref_corr = np.nan_to_num(np.corrcoef(X_train_raw, rowvar=False))

    loader = DataLoader(TensorDataset(torch.FloatTensor(X_scaled)),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    gen = TabularQuantumGenerator(circuit).to(device)
    crit = WGANCritic(N_QUBITS).to(device)
    opt_g = optim.Adam(gen.parameters(), lr=LR_GENERATOR, betas=ADAM_BETAS)
    opt_d = optim.Adam(crit.parameters(), lr=LR_CRITIC, betas=ADAM_BETAS)

    out_dir = out_dir or qgan_model_dir(target_label)
    os.makedirs(out_dir, exist_ok=True)

    epoch_log, best = [], {"score": float("inf"), "epoch": -1}
    evals_since_best, early_stopped = 0, False
    for epoch in range(1, epochs + 1):
        d_losses, g_losses = [], []
        for i, (real_batch,) in enumerate(loader):
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)

            opt_d.zero_grad()
            with torch.no_grad():
                fake_batch = gen(sample_noise(bs, device))
            loss_d = (crit(fake_batch).mean() - crit(real_batch).mean()
                      + LAMBDA_GP * compute_gradient_penalty(crit, real_batch, fake_batch, device))
            loss_d.backward()
            opt_d.step()
            d_losses.append(loss_d.item())

            if (i + 1) % N_CRITIC == 0:
                opt_g.zero_grad()
                loss_g = -crit(gen(sample_noise(bs, device))).mean()
                loss_g.backward()
                opt_g.step()
                g_losses.append(loss_g.item())

        row = {"epoch": epoch,
               "d_loss": float(np.mean(d_losses)) if d_losses else float("nan"),
               "g_loss": float(np.mean(g_losses)) if g_losses else float("nan")}

        if epoch % EVAL_EVERY == 0 or epoch == epochs:
            m = validation_metric(gen, preproc_state, X_val_raw, ref_std, ref_corr)
            row.update(m)
            if m["score"] < best["score"]:
                best = {"score": m["score"], "epoch": epoch, **m}
                torch.save(gen.state_dict(), os.path.join(out_dir, "weights_best.pth"))
                evals_since_best = 0
            else:
                evals_since_best += 1
            print(f"Epoch {epoch:03d} | D {row['d_loss']:+.4f} | G {row['g_loss']:+.4f} "
                  f"| W1 {m['std_w1']:.4f} corr {m['corr_dist']:.4f} "
                  f"score {m['score']:.4f}{' *best*' if best['epoch'] == epoch else ''}")
        else:
            print(f"Epoch {epoch:03d} | D {row['d_loss']:+.4f} | G {row['g_loss']:+.4f}")
        epoch_log.append(row)

        if patience > 0 and evals_since_best >= patience:
            early_stopped = True
            print(f"[early stop] no improvement for {patience} evaluations "
                  f"({patience * EVAL_EVERY} epochs); stopping at epoch {epoch}.")
            break

    torch.save(gen.state_dict(), os.path.join(out_dir, "weights_last.pth"))
    joblib.dump(preproc_state, os.path.join(out_dir, "preproc.pkl"))
    pd.DataFrame(epoch_log).to_csv(os.path.join(out_dir, "training_history.csv"), index=False)

    manifest = {"label": int(target_label), "circuit_version": circuit,
                "preproc": preproc, "seed": int(seed + target_label),
                "epochs": int(epochs), "batch_size": BATCH_SIZE,
                "n_train": int(len(X_train_raw)), "n_val": int(len(X_val_raw)),
                "best_epoch": int(best["epoch"]), "best_score": best["score"],
                "stopped_epoch": int(epoch), "early_stopped": early_stopped,
                "patience": int(patience),
                "feature_order": MANUAL_FEATURES_16}
    with open(os.path.join(out_dir, "model_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[saved] best epoch {best['epoch']} (score {best['score']:.4f}) -> {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Train QGAN for one category")
    p.add_argument("category", type=int, nargs="?", default=0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--circuit", choices=["v1", "v2"], default="v2")
    p.add_argument("--preproc", choices=list(PREPROC_KINDS), default="quantile")
    p.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    p.add_argument("--patience", type=int, default=6,
                   help="stop after N validation evals without improvement (0 = off); "
                        "with EVAL_EVERY=5, patience 6 = 30 epochs of no progress")
    p.add_argument("--out-dir", default=None,
                   help="override output directory (REQUIRED for ablation runs so "
                        "configs/seeds do not overwrite each other; main runs omit it)")
    a = p.parse_args()
    train_one_class(a.category, a.seed, a.circuit, a.preproc, a.epochs, a.patience, a.out_dir)


if __name__ == "__main__":
    main()
