import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = PROJECT_ROOT / "outputs" / "models" / "qgan_0-9"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results" / "qgan_training_curves"


def parse_report(report_path):
    target_label = None
    epochs = []
    d_losses = []
    g_losses = []
    in_epoch_log = False
    folder_label = int(report_path.parent.name)

    with report_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("Target label:"):
                target_label = int(line.split(":", 1)[1].strip())
                continue

            if line == "Epoch log:":
                in_epoch_log = True
                continue

            if in_epoch_log:
                parts = line.split()
                if len(parts) != 3:
                    continue
                try:
                    epoch = int(parts[0])
                    d_loss = float(parts[1])
                    g_loss = float(parts[2])
                except ValueError:
                    continue

                epochs.append(epoch)
                d_losses.append(d_loss)
                g_losses.append(g_loss)

    if target_label is not None and target_label != folder_label:
        print(
            f"Warning: {report_path} reports label {target_label}, "
            f"but folder name is {folder_label}; using folder label."
        )

    if not epochs:
        raise ValueError(f"No epoch log found in {report_path}")

    return pd.DataFrame(
        {
            "category": folder_label,
            "epoch": epochs,
            "d_loss": d_losses,
            "g_loss": g_losses,
        }
    )


def plot_per_category(history_df, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.set_theme(style="whitegrid")

    axes[0].plot(history_df["epoch"], history_df["d_loss"], color="#1f77b4", linewidth=2)
    axes[0].set_title(f"Category {int(history_df['category'].iloc[0])}: Discriminator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("D Loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_df["epoch"], history_df["g_loss"], color="#d62728", linewidth=2)
    axes[1].set_title(f"Category {int(history_df['category'].iloc[0])}: Generator Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("G Loss")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary(all_history, output_path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    sns.set_theme(style="whitegrid")

    for category, group in all_history.groupby("category"):
        axes[0].plot(group["epoch"], group["d_loss"], label=f"{category}", linewidth=1.8)
        axes[1].plot(group["epoch"], group["g_loss"], label=f"{category}", linewidth=1.8)

    axes[0].set_title("QGAN Discriminator Loss by Category")
    axes[0].set_ylabel("D Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(title="Category", ncol=2, fontsize=9)

    axes[1].set_title("QGAN Generator Loss by Category")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("G Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(title="Category", ncol=2, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_history = []
    for report_path in sorted(MODELS_ROOT.glob("*/report.txt")):
        try:
            history_df = parse_report(report_path)
        except ValueError as exc:
            print(f"Skipping {report_path}: {exc}")
            continue

        all_history.append(history_df)
        category = int(history_df["category"].iloc[0])

        per_category_csv = OUTPUT_DIR / f"qgan_training_history_{category:02d}.csv"
        history_df.to_csv(per_category_csv, index=False)

        per_category_png = OUTPUT_DIR / f"qgan_training_history_{category:02d}.png"
        plot_per_category(history_df, per_category_png)
        print(f"Saved category {category} history to {per_category_png}")

    if not all_history:
        raise RuntimeError(f"No QGAN report.txt files were parsed under {MODELS_ROOT}")

    summary_df = pd.concat(all_history, ignore_index=True)
    summary_csv = OUTPUT_DIR / "qgan_training_history_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    summary_png = OUTPUT_DIR / "qgan_training_history_summary.png"
    plot_summary(summary_df, summary_png)

    print(f"Saved summary history to {summary_csv}")
    print(f"Saved summary chart to {summary_png}")


if __name__ == "__main__":
    main()