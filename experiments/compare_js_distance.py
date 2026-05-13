import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 配置文件路径
# ==========================================
REAL_FILE = "./data/processed/selected_features_train.csv"                # 原始真实训练集
CTGAN_FILE = "./outputs/synthetic_data/Train_Balanced_CTGAN.csv"          # CTGAN 增强后的数据集
QGAN_FILE = "./outputs/synthetic_data/Train_Balanced_QGAN.csv"            # QGAN 增强后的数据集

NUM_CLASSES = 10                                         # 类别数 0-9
MAJORITY_LABELS = {0, 3, 6}                              # 多数类，不参与增强对比
MINORITY_LABELS = [c for c in range(NUM_CLASSES) if c not in MAJORITY_LABELS]
OUTPUT_DIR = "./outputs/results/js_comparison"          # 输出目录

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 数据加载与预处理
# ==========================================
def load_and_filter(real_path, ctgan_path, qgan_path, label):
    """加载数据并提取特定的类别"""
    df_real = pd.read_csv(real_path)
    df_ctgan = pd.read_csv(ctgan_path)
    df_qgan = pd.read_csv(qgan_path)
    
    # 过滤出目标类
    df_real = df_real[df_real['Label'] == label].drop(columns=['Label'])
    df_ctgan = df_ctgan[df_ctgan['Label'] == label].drop(columns=['Label'])
    df_qgan = df_qgan[df_qgan['Label'] == label].drop(columns=['Label'])
    
    # 确保特征列对齐
    common_cols = [c for c in df_real.columns if c in df_ctgan.columns and c in df_qgan.columns]
    
    print(f"Loaded Target Label: {label}")
    print(f"Real samples: {len(df_real)}, CTGAN samples: {len(df_ctgan)}, QGAN samples: {len(df_qgan)}")
    
    return df_real[common_cols], df_ctgan[common_cols], df_qgan[common_cols], common_cols

# ==========================================
# 3. 核心计算：公平的 JS 距离对比
# ==========================================
def calculate_comparative_js(df_real, df_ctgan, df_qgan, columns):
    metrics = []
    
    for col in columns:
        real_vals = df_real[col].dropna().values
        ctgan_vals = df_ctgan[col].dropna().values
        qgan_vals = df_qgan[col].dropna().values
        
        if len(real_vals) == 0 or len(ctgan_vals) == 0 or len(qgan_vals) == 0:
            continue
            
        # 【关键步骤】：寻找三者全局的最小值和最大值，确保分箱边界绝对一致
        min_val = min(np.min(real_vals), np.min(ctgan_vals), np.min(qgan_vals))
        max_val = max(np.max(real_vals), np.max(ctgan_vals), np.max(qgan_vals))
        
        if min_val == max_val:
            js_ctgan, js_qgan = 0.0, 0.0
        else:
            # 统一划分为 50 个区间
            bins = np.linspace(min_val, max_val, 50)
            
            # 计算频数
            count_real, _ = np.histogram(real_vals, bins=bins)
            count_ctgan, _ = np.histogram(ctgan_vals, bins=bins)
            count_qgan, _ = np.histogram(qgan_vals, bins=bins)
            
            # 转化为概率分布 (Probabilities must sum to 1)
            p_real = count_real / np.sum(count_real) if np.sum(count_real) > 0 else count_real
            p_ctgan = count_ctgan / np.sum(count_ctgan) if np.sum(count_ctgan) > 0 else count_ctgan
            p_qgan = count_qgan / np.sum(count_qgan) if np.sum(count_qgan) > 0 else count_qgan
            
            # 计算 JS 距离
            js_ctgan = jensenshannon(p_real, p_ctgan)
            js_qgan = jensenshannon(p_real, p_qgan)
            
        metrics.append({
            'Feature': col,
            'CTGAN': js_ctgan,
            'QGAN (Proposed)': js_qgan
        })
        
    metrics_df = pd.DataFrame(metrics)
    return metrics_df

# ==========================================
# 4. 生成学术级别的分组柱状图
# ==========================================
def plot_academic_js_comparison(metrics_df, output_path):
    # 将数据转换为长格式（Long format），方便 Seaborn 画分组柱状图
    df_melted = metrics_df.melt(id_vars="Feature", 
                                value_vars=["CTGAN", "QGAN (Proposed)"],
                                var_name="Model", 
                                value_name="JS Distance")
    
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    
    # 绘制分组柱状图
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

# ==========================================
# 5. 主函数执行
# ==========================================
if __name__ == "__main__":
    print("Starting JS Distance Comparison for All Categories...\n")
    
    # 存储所有类别的结果
    all_results = []
    
    # 循环处理每个少数类（跳过 0/3/6）
    for target_label in MINORITY_LABELS:
        print(f"\n{'='*60}")
        print(f"Processing Category {target_label}")
        print('='*60)
        
        try:
            # 加载数据
            df_real, df_ctgan, df_qgan, columns = load_and_filter(REAL_FILE, CTGAN_FILE, QGAN_FILE, target_label)
            
            # 计算指标
            df_metrics = calculate_comparative_js(df_real, df_ctgan, df_qgan, columns)
            
            # 计算平均值
            avg_ctgan = df_metrics['CTGAN'].mean()
            avg_qgan = df_metrics['QGAN (Proposed)'].mean()
            
            print("\n" + "-"*50)
            print(f"JS Distance Results for Category {target_label}:")
            print("-"*50)
            print(f"Average JS Distance - CTGAN: {avg_ctgan:.4f}")
            print(f"Average JS Distance - QGAN : {avg_qgan:.4f}")
            print(f"Performance Improvement  : {((avg_ctgan - avg_qgan) / avg_ctgan * 100):.2f}%" if avg_ctgan > 0 else "N/A")
            
            # 保存该类别的 CSV
            csv_path = os.path.join(OUTPUT_DIR, f'js_comparison_category_{target_label:02d}.csv')
            df_metrics.to_csv(csv_path, index=False)
            print(f"✓ Saved to: {csv_path}")
            
            # 保存图表
            img_path = os.path.join(OUTPUT_DIR, f'js_comparison_category_{target_label:02d}.png')
            plot_academic_js_comparison(df_metrics, img_path)
            
            # 聚合结果
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
    
    # 生成汇总报告
    print(f"\n\n{'='*70}")
    print("SUMMARY REPORT: All Categories")
    print('='*70)
    
    summary_df = pd.DataFrame(all_results)
    print(summary_df.to_string(index=False))
    
    # 保存汇总 CSV
    summary_csv = os.path.join(OUTPUT_DIR, 'js_comparison_summary.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✓ Summary saved to: {summary_csv}")
    
    # 生成汇总图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图表 1: 各类别的平均 JS 距离对比
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
    
    # 图表 2: QGAN 相对于 CTGAN 的性能提升
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
    
    # 打印总体统计
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print('='*70)
    print(f"Average CTGAN JS Distance: {summary_df['Avg_CTGAN_JS'].mean():.4f}")
    print(f"Average QGAN JS Distance : {summary_df['Avg_QGAN_JS'].mean():.4f}")
    print(f"Average Improvement      : {summary_df['QGAN_Improvement_%'].mean():.2f}%")
    print(f"Best Category            : {summary_df.loc[summary_df['Avg_QGAN_JS'].idxmin(), 'Category']}")
    print(f"Worst Category           : {summary_df.loc[summary_df['Avg_QGAN_JS'].idxmax(), 'Category']}")
    print('='*70)