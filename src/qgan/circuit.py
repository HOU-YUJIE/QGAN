"""Single source of truth for the QGAN quantum circuit and generator.

train.py and generate.py MUST import from this module. The previous setup
kept two hand-synced copies of the circuit, which silently diverged
(weights (L,16) vs (L,2,16)) and broke weight loading.

Architecture is FROZEN as of impg@735bdfc. Known issues, documented on
purpose and deliberately NOT fixed in this engineering pass (fix them later
as separate, measured experiments):

  1. Dead parameters: in the FINAL layer, the RZ gates on the numeric wires
     (weights[L-1, 1, 0:14]) and the CZ ring are diagonal and commute with
     the PauliZ measurement -> zero effect on outputs, zero gradient.
     Run `python -m src.qgan.circuit` to see the gradient check.
  2. Discrete-wire phase injection uses inputs * pi with inputs in [0, pi],
     so phases span [0, pi^2] > 2*pi (phase wrap-around).
  3. All learnable parameters are single-qubit rotations; the entangling
     gates (CZ / CNOT) carry no parameters.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from src.config import N_QUBITS, N_LAYERS, NUM_NUMERIC

try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
except Exception:  # pragma: no cover - fallback for environments w/o lightning
    dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="adjoint")
def qgan_circuit(inputs, weights):
    """inputs: [batch, N_QUBITS] in [0, pi]; weights: [N_LAYERS, 2, N_QUBITS]."""
    num_wires = list(range(NUM_NUMERIC))  # 0..13: numerical feature register

    for l in range(N_LAYERS):
        # --- Subsystem 1: numerical register (wires 0..13) ---
        # Latent re-injection each layer, alternating rotation axis.
        # (Note: `inputs` is latent NOISE, not real data - real data enters
        # the model only through the critic's gradients.)
        if l % 2 == 0:
            qml.AngleEmbedding(inputs[:, :NUM_NUMERIC], wires=num_wires, rotation="X")
        else:
            qml.AngleEmbedding(inputs[:, :NUM_NUMERIC], wires=num_wires, rotation="Z")

        for i in num_wires:
            qml.RY(weights[l, 0, i], wires=i)
            qml.RZ(weights[l, 1, i], wires=i)

        # Ring entanglement among numeric wires (parameter-free)
        for i in range(NUM_NUMERIC - 1):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[NUM_NUMERIC - 1, 0])

        # --- Subsystem 2: discrete flag register (wires 14, 15) ---
        qml.RY(weights[l, 0, 14], wires=14)
        qml.RY(weights[l, 0, 15], wires=15)
        qml.RZ(inputs[:, 14] * np.pi, wires=14)
        qml.RZ(inputs[:, 15] * np.pi, wires=15)
        qml.RY(weights[l, 1, 14], wires=14)
        qml.CRY(weights[l, 1, 15], wires=[14, 15])

        # --- Subsystem 3: cross-register coupling (parameter-free) ---
        qml.CNOT(wires=[0, 14])
        qml.CNOT(wires=[6, 14])
        qml.CNOT(wires=[13, 14])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class TabularQuantumGenerator(nn.Module):
    """Quantum generator: latent noise in [0, pi]^16 -> outputs in [0, pi]^16."""

    def __init__(self, n_qubits: int = N_QUBITS, n_layers: int = N_LAYERS):
        super().__init__()
        weight_shapes = {"weights": (n_layers, 2, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(qgan_circuit, weight_shapes)

    def forward(self, x):
        out = self.q_layer(x)                    # expvals in [-1, 1]
        return (out + 1.0) * (np.pi / 2.0)       # map to [0, pi]


def sample_noise(batch_size: int, device=None) -> torch.Tensor:
    """The one latent prior used everywhere: Uniform[0, pi]^N_QUBITS."""
    noise = torch.rand(batch_size, N_QUBITS) * np.pi
    return noise.to(device) if device is not None else noise


def gradient_sanity_check(batch_size: int = 8) -> None:
    """Print per-parameter gradient norms of a dummy loss.

    Run this after ANY circuit change. Entries that stay ~0 are dead
    parameters (currently: final-layer RZ on numeric wires).
    """
    gen = TabularQuantumGenerator()
    noise = sample_noise(batch_size)
    loss = gen(noise).sum()
    loss.backward()
    w = dict(gen.named_parameters())["q_layer.weights"]
    g = w.grad.abs()
    for l in range(N_LAYERS):
        for k, name in ((0, "RY"), (1, "RZ")):
            row = " ".join(f"{v:.0e}" for v in g[l, k].tolist())
            print(f"layer {l} {name}: {row}")
    dead = (g < 1e-12).sum().item()
    print(f"\ndead parameters (|grad| < 1e-12): {dead} / {g.numel()}")


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    gradient_sanity_check()
