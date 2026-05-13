import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


INPUT_FILE = "./data/processed/merged_cleaned_dataset.csv"
OUTPUT_FILE = "./data/processed/selected_features_dataset.csv"
CORR_THRESHOLD = 0.90                         
TOP_K_FEATURES = 25                           


def feature_selection_no_scale():

    df = pd.read_csv(INPUT_FILE)
    
    X = df.drop(columns=['Label_ID', 'Label_Name'])
    y = df['Label_ID']
    
    corr_matrix = X.corr().abs()
    
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop_corr = [column for column in upper_tri.columns if any(upper_tri[column] > CORR_THRESHOLD)]
    
    for column in to_drop_corr:

        correlated_with = upper_tri.index[upper_tri[column] > CORR_THRESHOLD].tolist()
        print(f"Drop: '{column}'")
        print(f"Because it has a correlation coefficient greater than {CORR_THRESHOLD} with {correlated_with}")

    X_filtered = X.drop(columns=to_drop_corr)

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_filtered, y)
    
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_filtered.columns, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    top_features = feature_importance_df['Feature'].head(TOP_K_FEATURES).tolist()
    
    print(feature_importance_df.head(25))
    
    df_final = X_filtered[top_features].copy()
    
    df_final['Label'] = y.values
    
    df_final.to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    feature_selection_no_scale()