from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

N_QUBITS = 16
SOURCE_LAYERS = 3
PLOT_LAYERS = 1

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PLOTS_DIR = PROJECT_ROOT / "outputs" / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def qgan_circuit(inputs, weights):
    num_wires = list(range(14))

    for l in range(PLOT_LAYERS):
        if l % 2 == 0:
            qml.AngleEmbedding(inputs[:14], wires=num_wires, rotation="X")
        else:
            qml.AngleEmbedding(inputs[:14], wires=num_wires, rotation="Z")

        for i in num_wires:
            qml.RY(weights[l, 0, i], wires=i)
            qml.RZ(weights[l, 1, i], wires=i)

        for i in range(13):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[13, 0])

        qml.RY(weights[l, 0, 14], wires=14)
        qml.RY(weights[l, 0, 15], wires=15)
        qml.RZ(inputs[14] * np.pi, wires=14)
        qml.RZ(inputs[15] * np.pi, wires=15)
        qml.RY(weights[l, 1, 14], wires=14)
        qml.CRY(weights[l, 1, 15], wires=[14, 15])

        qml.CNOT(wires=[0, 14])
        qml.CNOT(wires=[6, 14])
        qml.CNOT(wires=[13, 14])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


dummy_inputs = torch.ones(N_QUBITS)
dummy_weights = torch.randn(PLOT_LAYERS, 2, N_QUBITS)

fig, ax = plt.subplots(figsize=(16, 12))
qml.draw_mpl(qgan_circuit, ax=ax, decimals=2)(dummy_inputs, dummy_weights)

plt.title("Updated QGAN Architecture Visualization (1 Layer View)", fontsize=16)

output_path = PLOTS_DIR / "qgan_architecture.png"
plt.savefig(output_path, dpi=300)
print(f"Circuit diagram saved to: {output_path}")