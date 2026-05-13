from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pennylane as qml
import torch.nn as nn
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.ctgan import Discriminator, Generator


DEFAULT_DISCRETE_COLUMNS = ["fwd_psh_flags", "psh_flag_cnt"]


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def build_ctgan_parameter_report(train_features: pd.DataFrame, discrete_columns: list[str]) -> dict[str, int]:
    transformer = DataTransformer()
    transformer.fit(train_features, discrete_columns)
    transformed = transformer.transform(train_features)
    sampler = DataSampler(transformed, transformer.output_info_list, log_frequency=True)

    data_dim = transformer.output_dimensions
    cond_vec_dim = sampler.dim_cond_vec()

    generator = Generator(128 + cond_vec_dim, (256, 256, 256), data_dim)
    discriminator = Discriminator(data_dim + cond_vec_dim, (256, 256, 256), pac=10)

    return {
        "ctgan_output_dimensions": data_dim,
        "ctgan_cond_vec_dim": cond_vec_dim,
        "ctgan_generator_params": count_trainable_parameters(generator),
        "ctgan_discriminator_params": count_trainable_parameters(discriminator),
    }


def build_vqc_parameter_report() -> dict[str, int]:
    n_qubits = 16
    n_layers = 3

    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def qgan_circuit(inputs, weights):
        num_wires = list(range(14))

        for l in range(n_layers):
            qml.AngleEmbedding(inputs[:, :14], wires=num_wires)

            for i in num_wires:
                qml.RY(weights[l, i], wires=i)

            for i in range(13):
                qml.CZ(wires=[i, i + 1])
            qml.CZ(wires=[13, 0])

            qml.RZ(inputs[:, 14], wires=14)
            qml.RZ(inputs[:, 15], wires=15)
            qml.RY(weights[l, 14], wires=14)
            qml.CRY(weights[l, 15], wires=[14, 15])
            qml.CNOT(wires=[13, 14])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    class TabularQuantumGenerator(nn.Module):
        def __init__(self, n_qubits: int, n_layers: int):
            super().__init__()
            weight_shapes = {"weights": (n_layers, n_qubits)}
            self.q_layer = qml.qnn.TorchLayer(qgan_circuit, weight_shapes)

        def forward(self, x):
            out = self.q_layer(x)
            return (out + 1.0) * (3.141592653589793 / 2.0)

    class WGANCritic(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x)

    generator = TabularQuantumGenerator(n_qubits, n_layers)
    critic = WGANCritic(n_qubits)

    return {
        "vqc_generator_params": count_trainable_parameters(generator),
        "vqc_critic_params": count_trainable_parameters(critic),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trainable parameter counts for CTGAN and VQC models.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "processed" / "selected_features_train.csv",
        help="Path to the training CSV used by CTGAN.",
    )
    parser.add_argument(
        "--discrete-columns",
        nargs="*",
        default=DEFAULT_DISCRETE_COLUMNS,
        help="Discrete columns used by CTGAN.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    if "Label" not in df.columns:
        raise ValueError("The CSV must contain a 'Label' column.")

    train_features = df.drop(columns=["Label"])
    ctgan_report = build_ctgan_parameter_report(train_features, list(args.discrete_columns))
    vqc_report = build_vqc_parameter_report()

    ctgan_total = ctgan_report["ctgan_generator_params"] + ctgan_report["ctgan_discriminator_params"]
    vqc_total = vqc_report["vqc_generator_params"] + vqc_report["vqc_critic_params"]

    print("CTGAN")
    print(f"  output_dimensions: {ctgan_report['ctgan_output_dimensions']}")
    print(f"  cond_vec_dim: {ctgan_report['ctgan_cond_vec_dim']}")
    print(f"  generator_params: {ctgan_report['ctgan_generator_params']}")
    print(f"  discriminator_params: {ctgan_report['ctgan_discriminator_params']}")
    print(f"  total_params: {ctgan_total}")

    print()
    print("VQC")
    print(f"  generator_params: {vqc_report['vqc_generator_params']}")
    print(f"  critic_params: {vqc_report['vqc_critic_params']}")
    print(f"  total_params: {vqc_total}")

    print()
    print("Ratio")
    print(f"  CTGAN/VQC total: {ctgan_total / vqc_total:.2f}x")
    print(f"  CTGAN generator/VQC generator: {ctgan_report['ctgan_generator_params'] / vqc_report['vqc_generator_params']:.2f}x")


    """
    CTGAN
  output_dimensions: 2598
  cond_vec_dim: 2505
  generator_params: 11059452
  discriminator_params: 13195777
  total_params: 24255229

    VQC
  generator_params: 48
  critic_params: 12545
  total_params: 12593

    Ratio
  CTGAN/VQC total: 1926.09x
  CTGAN generator/VQC generator: 230405.25x"""
    
    
if __name__ == "__main__":
    main()