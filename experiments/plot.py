import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import N_QUBITS, NUM_NUMERIC
from src.qgan.circuit import CIRCUITS, WEIGHT_SHAPES

PLOTS_DIR = PROJECT_ROOT / "outputs" / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def build_dummy_inputs(version: str) -> torch.Tensor:
    """Create a sample input tensor that matches the selected circuit signature."""
    return torch.full((N_QUBITS,), torch.pi / 4)


def build_dummy_weights(version: str) -> torch.Tensor:
    """Create a single-layer weight tensor for the selected circuit version."""
    return torch.randn(WEIGHT_SHAPES[version][1:])


def build_plot_circuit(version: str):
    """Build a one-layer circuit for visualization.

    The training circuit may have multiple layers, but the diagram should only
    show a single representative layer to stay readable.
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def plot_circuit(inputs, weights):
        if version == "v1":
            qml.AngleEmbedding(inputs[:NUM_NUMERIC], wires=list(range(NUM_NUMERIC)), rotation="X")
            for i in range(NUM_NUMERIC):
                qml.RY(weights[0, i], wires=i)
                qml.RZ(weights[1, i], wires=i)
            for i in range(NUM_NUMERIC - 1):
                qml.CZ(wires=[i, i + 1])
            qml.CZ(wires=[NUM_NUMERIC - 1, 0])
            qml.RY(weights[0, 14], wires=14)
            qml.RY(weights[0, 15], wires=15)
            qml.RZ(inputs[14] * np.pi, wires=14)
            qml.RZ(inputs[15] * np.pi, wires=15)
            qml.RY(weights[1, 14], wires=14)
            qml.CRY(weights[1, 15], wires=[14, 15])
            qml.CNOT(wires=[0, 14])
            qml.CNOT(wires=[6, 14])
            qml.CNOT(wires=[13, 14])
        else:
            qml.AngleEmbedding(inputs[:NUM_NUMERIC], wires=list(range(NUM_NUMERIC)), rotation="X")
            for i in range(NUM_NUMERIC):
                qml.RZ(weights[0, i], wires=i)
                qml.RY(weights[1, i], wires=i)
            for i in range(NUM_NUMERIC):
                qml.CRY(weights[2, i], wires=[i, (i + 1) % NUM_NUMERIC])
            qml.RZ(inputs[14], wires=14)
            qml.RZ(inputs[15], wires=15)
            qml.RY(weights[0, 14], wires=14)
            qml.RY(weights[0, 15], wires=15)
            qml.CRY(weights[1, 14], wires=[14, 15])
            qml.CRY(weights[1, 15], wires=[0, 14])
            qml.CRY(weights[2, 14], wires=[6, 14])
            qml.CRY(weights[2, 15], wires=[13, 15])

        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    return plot_circuit


def plot_circuit(version: str = "v2", output_path: Path | None = None):
    if version not in CIRCUITS:
        raise ValueError(f"unknown circuit version {version!r}; choose from {list(CIRCUITS)}")

    circuit = build_plot_circuit(version)
    dummy_inputs = build_dummy_inputs(version)
    dummy_weights = build_dummy_weights(version)

    fig, ax = qml.draw_mpl(circuit, decimals=2)(dummy_inputs, dummy_weights)
    ax.set_title(f"QGAN Circuit Diagram ({version}, one layer)", fontsize=16)

    if output_path is None:
        output_path = PLOTS_DIR / f"qgan_architecture_{version}.png"

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Circuit diagram saved to: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Render the unified QGAN circuit diagram.")
    parser.add_argument("--version", default="v2", choices=sorted(CIRCUITS), help="Circuit version to render")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for the image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_circuit(version=args.version, output_path=args.output)