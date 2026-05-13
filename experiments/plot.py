import pennylane as qml
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameter definition ---
# Set layers to 1 for clear observation of single-layer internal logic
DRAW_LAYERS = 1
N_QUBITS = 16

# --- 2. Circuit logic ---
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def single_layer_circuit(inputs, weights):
    num_wires = list(range(14))  # 0~13: numerical feature register
    cat_wires = [14, 15]         # 14~15: categorical flag register

    # Iterate only one layer
    for l in range(DRAW_LAYERS):
        # Subsystem 1: Numerical feature injection and entanglement
        qml.AngleEmbedding(inputs[:14], wires=num_wires)
        for i in num_wires:
            qml.RY(weights[l, i], wires=i)
        
        # Ring CZ entanglement chain
        for i in range(13):
            qml.CZ(wires=[i, i + 1])
        qml.CZ(wires=[13, 0])

        # Subsystem 2: Discrete flag injection (RZ maintains phase perturbation)
        qml.RZ(inputs[14], wires=14)
        qml.RZ(inputs[15], wires=15)
        qml.RY(weights[l, 14], wires=14)
        
        # Physical logic constraint: controlled RY
        qml.CRY(weights[l, 15], wires=[14, 15])

        # Subsystem 3: Cross-register interference (Numerical -> Categorical)
        qml.CNOT(wires=[13, 14])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# --- 3. Prepare plotting data ---
dummy_inputs = torch.ones(N_QUBITS)
# Weight shape must match DRAW_LAYERS
dummy_weights = torch.randn(DRAW_LAYERS, N_QUBITS)

# --- 4. Execute plotting ---
# qml.drawer.use_style("sketch") 

# Adjust figsize: narrow width (only one layer), keep height (16 qubits)
fig, ax = plt.subplots(figsize=(12, 10)) 

qml.draw_mpl(single_layer_circuit, ax=ax, decimals=2)(dummy_inputs, dummy_weights)

plt.title("QGAN Single Layer Architecture Visualization", fontsize=16)
plt.tight_layout()

# Save and display
plt.savefig("single_layer_qgan.png", dpi=300)
print("Single-layer circuit diagram saved.")