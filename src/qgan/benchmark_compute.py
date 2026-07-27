"""Estimate wall-clock cost of QGAN training runs BEFORE launching them.

Benchmarks the two expensive primitives on THIS machine:
  * one critic step   (generator forward, no grad + critic fwd/bwd + GP)
  * one generator step (generator forward + adjoint backward)
  * one validation eval (generate n_val samples, no grad)

then projects, from the actual per-class row counts in features16_train.csv:
  * a full main run  (all minority classes, --epochs E)
  * the ablation grid ({v1,v2} x {log_minmax,quantile} x seeds on 3 classes)
  * the early-stopping expectation (budget if runs stop around E/2)

Run this on every machine you consider (local, Kaggle) - the numbers are
machine-specific. Set OMP_NUM_THREADS to your core count first; the
lightning.qubit simulator is CPU-parallel.

Usage:
    python src/qgan/benchmark_compute.py [--epochs 100] [--reps 3]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import pandas as pd
import torch

from src.config import (
    BATCH_SIZE, FEATURES16_TRAIN, LABEL_COLUMN, N_CRITIC, N_QUBITS,
    TARGET_TOTAL_SAMPLES, get_majority_labels,
)
from src.qgan.circuit import TabularQuantumGenerator, sample_noise
from src.qgan.train import WGANCritic, compute_gradient_penalty, VAL_FRACTION, EVAL_EVERY


def bench_version(version: str, reps: int) -> dict:
    device = torch.device("cpu")
    gen = TabularQuantumGenerator(version).to(device)
    crit = WGANCritic(N_QUBITS).to(device)
    real = torch.rand(BATCH_SIZE, N_QUBITS) * np.pi
    opt_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(crit.parameters(), lr=1e-3)

    # warmup (compilation paths)
    with torch.no_grad():
        gen(sample_noise(BATCH_SIZE))

    t0 = time.perf_counter()
    for _ in range(reps):
        opt_d.zero_grad()
        with torch.no_grad():
            fake = gen(sample_noise(BATCH_SIZE, device))
        loss_d = (crit(fake).mean() - crit(real).mean()
                  + 10 * compute_gradient_penalty(crit, real, fake, device))
        loss_d.backward()
        opt_d.step()
    t_critic = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    for _ in range(reps):
        opt_g.zero_grad()
        loss_g = -crit(gen(sample_noise(BATCH_SIZE, device))).mean()
        loss_g.backward()
        opt_g.step()
    t_gen = (time.perf_counter() - t0) / reps

    t0 = time.perf_counter()
    with torch.no_grad():
        gen(sample_noise(100))
    t_eval100 = time.perf_counter() - t0

    print(f"[{version}] critic step {t_critic:.2f}s | generator step {t_gen:.2f}s "
          f"| eval fwd (100 samples) {t_eval100:.2f}s")
    return {"critic": t_critic, "gen": t_gen, "eval100": t_eval100}


def project(times: dict, epochs: int) -> None:
    df = pd.read_csv(FEATURES16_TRAIN, usecols=[LABEL_COLUMN])
    counts = df[LABEL_COLUMN].value_counts().sort_index()
    majority = get_majority_labels(FEATURES16_TRAIN)
    minority = [c for c in counts.index if c not in majority]

    print(f"\nminority classes to train: {minority} (majority {sorted(majority)} skipped)")
    total = 0.0
    for c in minority:
        n_train = int(counts[c] * (1 - VAL_FRACTION))
        n_batches = n_train // BATCH_SIZE
        n_val = max(30, int(VAL_FRACTION * counts[c]))
        per_epoch = (n_batches * times["critic"]
                     + (n_batches // N_CRITIC) * times["gen"])
        eval_cost = (epochs // EVAL_EVERY) * times["eval100"] * (n_val / 100)
        cls_total = per_epoch * epochs + eval_cost
        total += cls_total
        print(f"  class {c}: {counts[c]:5d} rows, {n_batches:3d} batches/epoch "
              f"-> {per_epoch:6.1f}s/epoch, {cls_total/3600:5.2f} h @ {epochs} epochs")

    print(f"\n== full main run ({len(minority)} classes x {epochs} epochs): "
          f"{total/3600:.1f} h ==")
    print(f"== with early stopping (~E/2 expectation):            ~{total/7200:.1f} h ==")
    grid = total / len(minority) * 3 * 4 * 3  # 3 classes x 4 configs x 3 seeds (approx)
    print(f"== ablation grid (3 classes x 4 configs x 3 seeds):   ~{grid/3600:.1f} h "
          f"(before early stopping) ==")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--versions", nargs="+", default=["v1", "v2"])
    a = p.parse_args()
    print(f"machine: {os.cpu_count()} cores | OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    for v in a.versions:
        t = bench_version(v, a.reps)
        project(t, a.epochs)


if __name__ == "__main__":
    main()
