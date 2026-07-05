"""Single source of truth for the QGAN quantum circuits and generator.

Two circuit versions, selectable for controlled ablation:

  v1 - FROZEN architecture from impg@735bdfc. Known issues (all verified):
       14 dead parameters (final-layer RZ on numeric wires), parameter-free
       entanglers, phase wrap on discrete wires (inputs * pi with inputs
       already in [0, pi]).

  v2 - repaired architecture, same depth, weights (N_LAYERS, 3, N_QUBITS):
       * per-layer gate order RZ -> RY on numeric wires: every rotation is
         followed by a non-diagonal gate before measurement -> no dead params
       * CZ ring replaced by a parameterized CRY ring: inter-feature
         correlation strength is now learnable
       * cross-register coupling (numeric -> discrete) parameterized (CRY)
       * discrete-wire phase injection uses inputs directly (already radians)

Weight slot map for v2 (per layer l):
  [l, 0, 0:14] RZ on numeric wires        [l, 1, 0:14] RY on numeric wires
  [l, 2, 0:14] CRY ring (i -> i+1 mod 14)
  [l, 0, 14], [l, 0, 15]  RY on discrete wires
  [l, 1, 14]  CRY 14 -> 15 (flag dependency)
  [l, 1, 15]  CRY 0  -> 14 (cross-register)
  [l, 2, 14]  CRY 6  -> 14 (cross-register)
  [l, 2, 15]  CRY 13 -> 15 (cross-register)

Run `python -m src.qgan.circuit` for the gradient sanity check on both
versions (v1 expected: 14/96 dead; v2 expected: 0/144 dead).
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from src.config import N_QUBITS, N_LAYERS, NUM_NUMERIC

try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
except Exception:  # pragma: no cover
    dev = qml.device("default.qubit", wires=N_QUBITS)

NUM_WIRES = list(range(NUM_NUMERIC))
WEIGHT_SHAPES = {"v1": (N_LAYERS, 2, N_QUBITS), "v2": (N_LAYERS, 3, N_QUBITS)}


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit_v1(inputs, weights):
    """Frozen impg@735bdfc circuit. weights: [N_LAYERS, 2, 16]."""
    for l in range(N_LAYERS):
        if l % 2 == 0:
            qml.AngleEmbedding(inputs[:, :NUM_NUMERIC], wires=NUM_WIRES, rotation="X")
        else:
            qml.AngleEmbedding(inputs[:, :NUM_NUMERIC], wires=NUM_WIRES, rotation="Z")
        for i in NUM_WIRES:
            qml.RY(weights[l, 0, i], wires=i)
            qml.RZ(weights[l, 1, i], wires=i)
        for i in range(NUM_NUMERIC - 1):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[NUM_NUMERIC - 1, 0])
        qml.RY(weights[l, 0, 14], wires=14)
        qml.RY(weights[l, 0, 15], wires=15)
        qml.RZ(inputs[:, 14] * np.pi, wires=14)
        qml.RZ(inputs[:, 15] * np.pi, wires=15)
        qml.RY(weights[l, 1, 14], wires=14)
        qml.CRY(weights[l, 1, 15], wires=[14, 15])
        qml.CNOT(wires=[0, 14])
        qml.CNOT(wires=[6, 14])
        qml.CNOT(wires=[13, 14])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def circuit_v2(inputs, weights):
    """Repaired circuit. weights: [N_LAYERS, 3, 16]."""
    for l in range(N_LAYERS):
        # latent re-injection, alternating axis (inputs are NOISE, not data)
        if l % 2 == 0:
            qml.AngleEmbedding(inputs[:, :NUM_NUMERIC], wires=NUM_WIRES, rotation="X")
        else:
            qml.AngleEmbedding(inputs[:, :NUM_NUMERIC], wires=NUM_WIRES, rotation="Z")

        # numeric register: RZ THEN RY (non-diagonal last -> no dead params)
        for i in NUM_WIRES:
            qml.RZ(weights[l, 0, i], wires=i)
            qml.RY(weights[l, 1, i], wires=i)

        # learnable ring entanglement (correlation strengths are parameters)
        for i in NUM_WIRES:
            qml.CRY(weights[l, 2, i], wires=[i, (i + 1) % NUM_NUMERIC])

        # discrete register: phase injection WITHOUT the extra pi factor
        qml.RZ(inputs[:, 14], wires=14)
        qml.RZ(inputs[:, 15], wires=15)
        qml.RY(weights[l, 0, 14], wires=14)
        qml.RY(weights[l, 0, 15], wires=15)
        qml.CRY(weights[l, 1, 14], wires=[14, 15])   # flag dependency 14->15

        # parameterized cross-register coupling (numeric -> discrete)
        qml.CRY(weights[l, 1, 15], wires=[0, 14])
        qml.CRY(weights[l, 2, 14], wires=[6, 14])
        qml.CRY(weights[l, 2, 15], wires=[13, 15])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


CIRCUITS = {"v1": circuit_v1, "v2": circuit_v2}


class TabularQuantumGenerator(nn.Module):
    """Latent noise in [0, pi]^16 -> outputs in [0, pi]^16."""

    def __init__(self, version: str = "v2"):
        super().__init__()
        if version not in CIRCUITS:
            raise ValueError(f"unknown circuit version {version!r}; choose from {list(CIRCUITS)}")
        self.version = version
        self.q_layer = qml.qnn.TorchLayer(CIRCUITS[version], {"weights": WEIGHT_SHAPES[version]})

    def forward(self, x):
        out = self.q_layer(x)
        return (out + 1.0) * (np.pi / 2.0)


def sample_noise(batch_size: int, device=None) -> torch.Tensor:
    noise = torch.rand(batch_size, N_QUBITS) * np.pi
    return noise.to(device) if device is not None else noise


def gradient_sanity_check(version: str, batch_size: int = 8) -> int:
    gen = TabularQuantumGenerator(version)
    loss = gen(sample_noise(batch_size)).sum()
    loss.backward()
    g = dict(gen.named_parameters())["q_layer.weights"].grad.abs()
    dead = int((g < 1e-12).sum().item())
    print(f"[{version}] dead parameters (|grad| < 1e-12): {dead} / {g.numel()}")
    return dead


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    for v in ("v1", "v2"):
        gradient_sanity_check(v)
