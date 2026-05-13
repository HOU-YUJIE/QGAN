import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

# Real data
REAL_FILE = "./data/processed/selected_features_train.csv"  
MAJORITY_LABELS = {0, 3, 6}
MINORITY_LABELS = [label for label in range(10) if label not in MAJORITY_LABELS]


def load_data(target_label, synthetic_file):
    """Load real and synthetic data for a specific category.
    
    Args:
        target_label: The category label to extract
        synthetic_file: Path to synthetic data CSV file
        
    Returns:
        df_real, df_synthetic, common_cols
    """
    df_real = pd.read_csv(REAL_FILE)
    df_synthetic = pd.read_csv(synthetic_file)
    
    # Extract same category for fair comparison
    df_real = df_real[df_real['Label'] == target_label].drop(columns=['Label'])
    df_synthetic = df_synthetic.drop(columns=['Label'], errors='ignore')
    
    # Ensure both sides have matching column names
    common_cols = [c for c in df_real.columns if c in df_synthetic.columns]
    df_real = df_real[common_cols]
    df_synthetic = df_synthetic[common_cols]
    
    return df_real, df_synthetic, common_cols

def plot_distributions(df_real, df_synthetic, cols_to_plot, output_dir, n_cols=3):
    """Plot univariate KDE (Kernel Density Estimation) to compare 1D distributions."""
    print("\n1. Generating [feature distribution comparison chart]...")
    n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_to_plot):
        sns.kdeplot(df_real[col], ax=axes[i], color='blue', label='Real Data', fill=True, alpha=0.3)
        sns.kdeplot(df_synthetic[col], ax=axes[i], color='red', label='Synthetic Data', fill=True, alpha=0.3)
        axes[i].set_title(col)
        axes[i].legend()
    
    # Hide extra subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'evaluation_distributions.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved as {output_path}")

def plot_correlations(df_real, df_synthetic, output_dir):
    """Plot correlation matrices to check if feature associations are preserved."""
    print("2. Generating [correlation matrix comparison chart]...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(df_real.corr(), ax=axes[0], cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title("Real Data Correlation Matrix")
    
    sns.heatmap(df_synthetic.corr(), ax=axes[1], cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title("Synthetic Data Correlation Matrix")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'evaluation_correlations.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved as {output_path}")

def calculate_statistical_metrics(df_real, df_synthetic, columns, output_dir):
    """Calculate statistical distances (Wasserstein and JS distance); lower is more similar."""
    print("3. Calculating [statistical difference metrics]...")
    metrics = []
    for col in columns:
        real_vals = df_real[col].dropna().values
        synthetic_vals = df_synthetic[col].dropna().values
        
        if len(real_vals) == 0 or len(synthetic_vals) == 0:
            continue
            
        # Wasserstein Distance
        w_dist = wasserstein_distance(real_vals, synthetic_vals)
        
        # Jensen-Shannon (JS) Distance
        min_val = min(np.min(real_vals), np.min(synthetic_vals))
        max_val = max(np.max(real_vals), np.max(synthetic_vals))
        
        if min_val == max_val:
            js_dist = 0.0
        else:
            # Create common bins to generate probability distributions
            bins = np.linspace(min_val, max_val, 50)
            p, _ = np.histogram(real_vals, bins=bins)
            q, _ = np.histogram(synthetic_vals, bins=bins)
            js_dist = jensenshannon(p, q)
        
        metrics.append({
            'Feature': col,
            'Wasserstein_Distance': w_dist,
            'JS_Distance': js_dist
        })
    
    metrics_df = pd.DataFrame(metrics).sort_values(by='JS_Distance')
    print("\n--- Feature similarity (Top 5 best) ---")
    print(metrics_df.head(5).to_string(index=False))
    print("\n--- Feature similarity (Bottom 5 worst) ---")
    print(metrics_df.tail(5).to_string(index=False))
    
    avg_js = metrics_df['JS_Distance'].mean()
    print(f"\n[Overall average JS distance]: {avg_js:.4f} (< 0.2 is generally very good)")
    
    # Save metrics to CSV
    metrics_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"   -> Metrics saved to {metrics_path}")
    
    return metrics_df, avg_js

def plot_pca_overlap(df_real, df_synthetic, output_dir):
    """Use PCA to reduce to 2D and visualize high-dimensional overlap."""
    print("4. Generating [PCA 2D projection chart]...")
    
    # Concatenate data with labels
    df_real_pca = df_real.copy()
    df_synthetic_pca = df_synthetic.copy()
    df_real_pca['Type'] = 'Real'
    df_synthetic_pca['Type'] = 'Synthetic'
    df_combined = pd.concat([df_real_pca, df_synthetic_pca], ignore_index=True)
    
    # Standardize for fair PCA
    features = df_combined.drop(columns=['Type'])
    features_scaled = (features - features.mean()) / features.std()
    features_scaled = features_scaled.fillna(0)  # defensive fill
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features_scaled)
    
    df_combined['PCA1'] = pca_result[:, 0]
    df_combined['PCA2'] = pca_result[:, 1]
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df_combined, x='PCA1', y='PCA2', hue='Type', 
        palette={'Real': 'blue', 'Synthetic': 'red'}, alpha=0.5, s=20
    )
    plt.title("PCA 2D Projection: Real vs Synthetic")
    output_path = os.path.join(output_dir, 'evaluation_pca.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   -> Saved as {output_path}")

if __name__ == "__main__":
    # Evaluate each minority category only
    base_dir = Path("./outputs/models/qgan_0-9")
    
    for category in MINORITY_LABELS:
        print(f"\n{'='*60}")
        print(f"Evaluating Category {category}")
        print('='*60)
        
        # Define paths
        category_dir = base_dir / str(category)
        synthetic_file = category_dir / "Synthetic_Traffic_16dim.csv"
        output_dir = str(category_dir)
        
        # Check if synthetic file exists
        if not synthetic_file.exists():
            print(f"WARNING: Synthetic file not found at {synthetic_file}, skipping...")
            continue
        
        # Load data
        print(f"Loading data for category {category}...")
        try:
            df_real, df_synthetic, common_cols = load_data(category, str(synthetic_file))
            print(f"Data loaded: {len(df_real)} real samples, {len(df_synthetic)} synthetic samples, comparing {len(common_cols)} features.")
            
            if len(df_real) == 0 or len(df_synthetic) == 0:
                print(f"WARNING: No data found for category {category}, skipping...")
                continue
            
            # Plot top 6 important features
            top_6_features = common_cols[:6] if len(common_cols) >= 6 else common_cols
            
            # Generate evaluation results
            plot_distributions(df_real, df_synthetic, top_6_features, output_dir)
            plot_correlations(df_real, df_synthetic, output_dir)
            metrics_df, avg_js = calculate_statistical_metrics(df_real, df_synthetic, common_cols, output_dir)
            plot_pca_overlap(df_real, df_synthetic, output_dir)
            
            # Generate summary report
            report_path = os.path.join(output_dir, 'evaluation_report.txt')
            with open(report_path, 'w') as f:
                f.write(f"Category {category} Evaluation Report\n")
                f.write("="*50 + "\n\n")
                f.write(f"Real Data Samples: {len(df_real)}\n")
                f.write(f"Synthetic Data Samples: {len(df_synthetic)}\n")
                f.write(f"Number of Features: {len(common_cols)}\n")
                f.write(f"Average JS Distance: {avg_js:.4f}\n\n")
                f.write("Detailed Metrics:\n")
                f.write("-"*50 + "\n")
                f.write(metrics_df.to_string(index=False))
                f.write("\n\n")
                f.write("Generated Files:\n")
                f.write("- evaluation_distributions.png\n")
                f.write("- evaluation_correlations.png\n")
                f.write("- evaluation_pca.png\n")
                f.write("- evaluation_metrics.csv\n")
            
            print(f"\nCategory {category} evaluation complete!")
            print(f"  Results saved to: {output_dir}")
            
        except Exception as e:
            print(f"ERROR processing category {category}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All evaluations complete!")
    print('='*60)