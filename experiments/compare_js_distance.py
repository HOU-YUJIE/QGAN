import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import os
import warnings
warnings.filterwarnings('ignore')

REAL_FILE = "./data/processed/selected_features_train.csv"                # real training dataset
CTGAN_FILE = "./outputs/synthetic_data/Train_Balanced_CTGAN.csv"          # CTGAN augmented dataset
QGAN_FILE = "./outputs/synthetic_data/Train_Balanced_QGAN.csv"            # QGAN augmented dataset

NUM_CLASSES = 10                                         # number of classes (0-9)
MAJORITY_LABELS = {0, 3, 6}                              # majority classes (excluded from augmentation comparison)
MINORITY_LABELS = [c for c in range(NUM_CLASSES) if c not in MAJORITY_LABELS]
OUTPUT_DIR = "./outputs/results/js_comparison"          # output directory

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_and_filter(real_path, ctgan_path, qgan_path, label):
    df_real = pd.read_csv(real_path)
    df_ctgan = pd.read_csv(ctgan_path)
    df_qgan = pd.read_csv(qgan_path)
    
    # Filter rows for the target label and drop the label column
    df_real = df_real[df_real['Label'] == label].drop(columns=['Label'])
    df_ctgan = df_ctgan[df_ctgan['Label'] == label].drop(columns=['Label'])
    df_qgan = df_qgan[df_qgan['Label'] == label].drop(columns=['Label'])
    
    # Ensure feature columns are aligned across datasets
    common_cols = [c for c in df_real.columns if c in df_ctgan.columns and c in df_qgan.columns]
    
    print(f"Loaded Target Label: {label}")
    print(f"Real samples: {len(df_real)}, CTGAN samples: {len(df_ctgan)}, QGAN samples: {len(df_qgan)}")
    
    return df_real[common_cols], df_ctgan[common_cols], df_qgan[common_cols], common_cols

def calculate_comparative_js(df_real, df_ctgan, df_qgan, columns):
    metrics = []
    
    for col in columns:
        real_vals = df_real[col].dropna().values
        ctgan_vals = df_ctgan[col].dropna().values
        qgan_vals = df_qgan[col].dropna().values
        
        if len(real_vals) == 0 or len(ctgan_vals) == 0 or len(qgan_vals) == 0:
            continue
            
        # Find global min/max across the three distributions to ensure identical bin boundaries
        min_val = min(np.min(real_vals), np.min(ctgan_vals), np.min(qgan_vals))
        max_val = max(np.max(real_vals), np.max(ctgan_vals), np.max(qgan_vals))
        
        if min_val == max_val:
            js_ctgan, js_qgan = 0.0, 0.0
        else:
            # Partition the range into 50 bins
            bins = np.linspace(min_val, max_val, 50)
            
            # Compute histogram counts
            count_real, _ = np.histogram(real_vals, bins=bins)
            count_ctgan, _ = np.histogram(ctgan_vals, bins=bins)
            count_qgan, _ = np.histogram(qgan_vals, bins=bins)
            
            # Convert counts to probability distributions (must sum to 1)
            p_real = count_real / np.sum(count_real) if np.sum(count_real) > 0 else count_real
            p_ctgan = count_ctgan / np.sum(count_ctgan) if np.sum(count_ctgan) > 0 else count_ctgan
            p_qgan = count_qgan / np.sum(count_qgan) if np.sum(count_qgan) > 0 else count_qgan
            
            # Compute Jensen-Shannon distances
            js_ctgan = jensenshannon(p_real, p_ctgan)
            js_qgan = jensenshannon(p_real, p_qgan)
            
        metrics.append({
            'Feature': col,
            'CTGAN': js_ctgan,
            'QGAN (Proposed)': js_qgan
        })
        
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


# Generate publication-quality grouped bar chart

def plot_academic_js_comparison(metrics_df, output_path):
    # Melt metrics into long format for Seaborn grouped barplot
    df_melted = metrics_df.melt(id_vars="Feature", 
                                value_vars=["CTGAN", "QGAN (Proposed)"],
                                var_name="Model", 
                                value_name="JS Distance")
    
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(x="Feature", y="JS Distance", hue="Model", data=df_melted, palette=["#FF9999", "#66B2FF"])
    
    plt.title("Jensen-Shannon Distance Comparison: CTGAN vs. Proposed QGAN", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Network Traffic Features", fontsize=14, labelpad=10)
    plt.ylabel("JS Distance (Lower is Better)", fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    # 添加图例
    plt.legend(title="Generative Model", fontsize=12, title_fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to: {output_path}")


# Main execution
if __name__ == "__main__":
    print("Starting JS Distance Comparison for All Categories...\n")
    
    # Store results for all categories
    all_results = []
    
    # Loop over minority classes only (skip majority: 0,3,6)
    for target_label in MINORITY_LABELS:
        print(f"\n{'='*60}")
        print(f"Processing Category {target_label}")
        print('='*60)
        
        try:
            # Load data filtered by category
            df_real, df_ctgan, df_qgan, columns = load_and_filter(REAL_FILE, CTGAN_FILE, QGAN_FILE, target_label)
            
            # Calculate per-feature JS metrics
            df_metrics = calculate_comparative_js(df_real, df_ctgan, df_qgan, columns)
            
            # Compute average JS distances
            avg_ctgan = df_metrics['CTGAN'].mean()
            avg_qgan = df_metrics['QGAN (Proposed)'].mean()
            
            print("\n" + "-"*50)
            print(f"JS Distance Results for Category {target_label}:")
            print("-"*50)
            print(f"Average JS Distance - CTGAN: {avg_ctgan:.4f}")
            print(f"Average JS Distance - QGAN : {avg_qgan:.4f}")
            print(f"Performance Improvement  : {((avg_ctgan - avg_qgan) / avg_ctgan * 100):.2f}%" if avg_ctgan > 0 else "N/A")
            
            # Save per-category CSV of JS metrics
            csv_path = os.path.join(OUTPUT_DIR, f'js_comparison_category_{target_label:02d}.csv')
            df_metrics.to_csv(csv_path, index=False)
            print(f"✓ Saved to: {csv_path}")
            
            # Save per-category plot
            img_path = os.path.join(OUTPUT_DIR, f'js_comparison_category_{target_label:02d}.png')
            plot_academic_js_comparison(df_metrics, img_path)
            
            all_results.append({
                'Category': target_label,
                'Real_Samples': len(df_real),
                'CTGAN_Samples': len(df_ctgan),
                'QGAN_Samples': len(df_qgan),
                'Num_Features': len(columns),
                'Avg_CTGAN_JS': avg_ctgan,
                'Avg_QGAN_JS': avg_qgan,
                'QGAN_Improvement_%': ((avg_ctgan - avg_qgan) / avg_ctgan * 100) if avg_ctgan > 0 else 0
            })
            
        except Exception as e:
            print(f"WARNING: Error processing category {target_label}: {e}")
            continue
    
    # Produce summary report
    print(f"\n\n{'='*70}")
    print("SUMMARY REPORT: All Categories")
    print('='*70)
    
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    
    summary_csv = os.path.join(OUTPUT_DIR, 'js_comparison_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Summary saved to: {summary_csv}")
    
    # Create summary plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Average JS distance per category
    x = np.arange(len(summary_df))
    width = 0.35
    axes[0].bar(x - width/2, summary_df['Avg_CTGAN_JS'], width, label='CTGAN', color='#FF9999')
    axes[0].bar(x + width/2, summary_df['Avg_QGAN_JS'], width, label='QGAN (Proposed)', color='#66B2FF')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Average JS Distance (Lower is Better)')
    axes[0].set_title('Average JS Distance per Category')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary_df['Category'].astype(int))
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: QGAN performance improvement vs CTGAN
    axes[1].bar(summary_df['Category'], summary_df['QGAN_Improvement_%'], color='#66B2FF')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('QGAN Improvement (%)')
    axes[1].set_title('Performance Improvement: QGAN vs CTGAN')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=1)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    summary_plot = os.path.join(OUTPUT_DIR, 'js_comparison_summary_plot.png')
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    print(f"✓ Summary plot saved to: {summary_plot}")
    
    # Print overall statistics
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print('='*70)
    print(f"Average CTGAN JS Distance: {summary_df['Avg_CTGAN_JS'].mean():.4f}")
    print(f"Average QGAN JS Distance : {summary_df['Avg_QGAN_JS'].mean():.4f}")
    print(f"Average Improvement      : {summary_df['QGAN_Improvement_%'].mean():.2f}%")
    print(f"Best Category            : {summary_df.loc[summary_df['Avg_QGAN_JS'].idxmin(), 'Category']}")
    print(f"Worst Category           : {summary_df.loc[summary_df['Avg_QGAN_JS'].idxmax(), 'Category']}")
    print('='*70)