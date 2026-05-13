import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix

RESULTS_TEXT_PATH = "./outputs/results/result.txt"

def set_global_seed(seed=42):
    print(f"Setting global random seed to {seed}...")
    
    # 1. Python built-in random module
    random.seed(seed)
    
    # 2. OS-level environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 3. Numpy random seed
    np.random.seed(seed)
    
    # 4. PyTorch random seed (CPU)
    torch.manual_seed(seed)
    
    # 5. PyTorch random seed (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
        
        # Force deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




# 1. Experiment configuration

# Training file paths
TRAIN_FILES = {
    "Baseline": "./data/processed/selected_features_train.csv", 
    "CTGAN Augmented": "./outputs/synthetic_data/Train_Balanced_CTGAN.csv",
    "QGAN Augmented": "./outputs/synthetic_data/Train_Balanced_QGAN.csv"
}

# Real test set
TEST_FILE = "./data/processed/selected_features_test.csv"

INPUT_DIM = 16
NUM_CLASSES = 10
EPOCHS = 100
BATCH_SIZE = 128
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current compute device: {device}")


# 2. MLP

class TrafficClassifierMLP(nn.Module):
    def __init__(self, input_dim=16, num_classes=10):
        super(TrafficClassifierMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            #nn.Dropout(0.2),
            
            nn.Linear(32, num_classes)  # CrossEntropyLoss includes Softmax
        )

    def forward(self, x):
        return self.network(x)


# 3. Core training and evaluation functions
def train_and_evaluate(train_path, test_path, experiment_name):
    print(f"\n" + "="*50)
    print(f"Starting experiment: {experiment_name}")
    print("="*50)
    
    # 1. Load data
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    feature_cols = [c for c in df_train.columns if c != 'Label']
    
    X_train = df_train[feature_cols].values
    y_train = df_train['Label'].values.astype(np.int64)
    X_test = df_test[feature_cols].values
    y_test = df_test['Label'].values.astype(np.int64)
    
    # 2. Independent standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train)), 
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test)), 
                             batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize model
    model = TrafficClassifierMLP(INPUT_DIM, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. Training loop
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
    # 5. Evaluation on test set
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
    # 6. Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    macro_recall = recall_score(all_targets, all_preds, average='macro')

    cm_save_path = f"./outputs/results/plots/CM_{experiment_name.replace(' ', '_')}.png"
    plot_academic_confusion_matrix(all_targets, all_preds, 
                                   title=f"Confusion Matrix: {experiment_name}", 
                                   save_path=cm_save_path)
    
    print(f"Final results for {experiment_name}:")
    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Macro F1-Score   : {macro_f1:.4f}")
    print(f"Macro Recall     : {macro_recall:.4f}")
    
    return {
        "Experiment": experiment_name,
        "Accuracy": acc,
        "Macro-F1": macro_f1,
        "Macro-Recall": macro_recall,
        "Report": classification_report(all_targets, all_preds, digits=4)
    }

def build_results_text(results):
    lines = []
    for result in results:
        lines.append(f">>> Group: {result['Experiment']}" )
        lines.append(result["Report"].rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

def plot_academic_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """
    Plot and save the confusion matrix.

    Args:
    y_true: list or array of true labels
    y_pred: list or array of predicted labels
    title: chart title
    save_path: path to save image
    """
    # 1. Compute confusion matrix data
    classes = np.arange(10)  # classes 0 through 9
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # 2. Compute percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Prevent NaN from division by zero
    
    # 3. Set plot style and figure size
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")
    
    # 4. Draw heatmap
    ax = sns.heatmap(cm_normalized,
                     annot=True,         # show values
                     fmt='.2f',          # format to two decimals
                     cmap='Blues',       # blue colormap
                     square=True,        # force square cells
                     cbar_kws={'label': 'Prediction Probability'},
                     xticklabels=classes,
                     yticklabels=classes)
    
    # 5. Styling details
    plt.title(title, fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    
    # Adjust tick label font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12, rotation=0)
    
    plt.tight_layout()
    
    # 6. Save or display
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# 4. Main function and visualization
if __name__ == "__main__":

    set_global_seed(seed=42)

    results = []
    
    for exp_name, train_file in TRAIN_FILES.items():
        res = train_and_evaluate(train_file, TEST_FILE, exp_name)
        results.append(res)
        
    print("\n" + "#"*50)
    print("10-class comparison report (Classification Report)")
    print("#"*50)
    for r in results:
        print(f"\n>>> Group: {r['Experiment']}")
        print(r['Report'])

    os.makedirs(os.path.dirname(RESULTS_TEXT_PATH), exist_ok=True)
    with open(RESULTS_TEXT_PATH, "w", encoding="utf-8") as f:
        f.write(build_results_text(results))
    print(f"\nResults text saved to: {RESULTS_TEXT_PATH}")
        
    df_results = pd.DataFrame(results)
    
    df_melted = df_results.melt(id_vars="Experiment", 
                                value_vars=["Accuracy", "Macro-F1", "Macro-Recall"],
                                var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="Metric", y="Score", hue="Experiment", data=df_melted, palette="Set2")
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)
        
    plt.title("Evaluation", fontsize=14, pad=15)
    plt.ylim(0, 1.1)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Evaluation Metrics", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    # Save image
    plt.savefig("./outputs/results/plots/Experiment_Comparison_Chart.png", dpi=300)
    print("\nExperiment comparison chart saved to: ./outputs/results/plots/Experiment_Comparison_Chart.png")