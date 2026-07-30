import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import joblib
import duckdb
from pathlib import Path

def load_data_for_explanation():
    processed_dir = Path("C:/Users/rohit/MultiModal/AEGIS/data/processed")
    features_df = pd.read_parquet(processed_dir / "physiological_features.parquet")
    labels_df = pd.read_csv(processed_dir / "ground_truth_labels.csv")
    splits_df = pd.read_csv(processed_dir / "split_assignments.csv")
    cohort_path = "C:/Users/rohit/MultiModal/AEGIS/data/processed/target_cohort.csv"
    
    conn = duckdb.connect()
    conn.register("features_df", features_df)
    conn.register("labels_df", labels_df)
    conn.register("splits_df", splits_df)
    
    query = f"""
    WITH cohort AS (
        SELECT stay_id, CAST(intime AS TIMESTAMP) as intime
        FROM read_csv_auto('{cohort_path}')
    ),
    features_with_time AS (
        SELECT f.*, 
               c.intime + INTERVAL (f.win_id * 4) HOUR as window_end_time
        FROM features_df f
        JOIN cohort c ON f.stay_id = c.stay_id
    ),
    joined AS (
        SELECT 
            f.*,
            s.split,
            CASE WHEN EXISTS (
                SELECT 1 FROM labels_df l
                WHERE l.stay_id = f.stay_id 
                  AND TRY_CAST(l.event_time AS TIMESTAMP) >= f.window_end_time 
                  AND TRY_CAST(l.event_time AS TIMESTAMP) <= f.window_end_time + INTERVAL 4 HOUR
            ) THEN 1 ELSE 0 END as target
        FROM features_with_time f
        JOIN splits_df s ON f.stay_id = s.stay_id
    )
    SELECT * FROM joined
    """
    df = conn.execute(query).df()
    
    feature_cols = [
        col for col in df.columns 
        if col.endswith(("_mean", "_min_val", "_max_val", "_std", "_slope", "_obs_count"))
    ]
    derived_flags = ["spo2_drop_flag", "tachypnea_flag", "bradycardia_flag", "fever_flag"]
    feature_cols.extend([col for col in derived_flags if col in df.columns])
    
    df[feature_cols] = df[feature_cols].fillna(0)
    
    test_df = df[df["split"] == "test"].copy()
    return test_df, feature_cols

def explain_tree_model(model, X_test, feature_names, save_path):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    np.save(save_path / "shap_values.npy", shap_values)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(save_path / "shap_summary.png", bbox_inches='tight')
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(save_path / "shap_bar.png", bbox_inches='tight')
    plt.close()
    
    global_importances = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(global_importances)[::-1][:5]
    for idx in top_indices:
        plt.figure()
        shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_names, show=False)
        plt.savefig(save_path / f"shap_dep_{feature_names[idx]}.png", bbox_inches='tight')
        plt.close()
        
    return shap_values

def generate_json_rankings(shap_vals, X_test, df_test, feature_names, save_path):
    save_path = Path(save_path)
    rankings = []
    
    X_test_vals = X_test.values
    stay_ids = df_test["stay_id"].values
    
    for i in range(len(X_test)):
        pat_shap = shap_vals[i]
        indices = np.argsort(np.abs(pat_shap))[::-1][:5]
        
        drivers = []
        for idx in indices:
            drivers.append({
                "feature": feature_names[idx],
                "value": float(X_test_vals[i, idx]),
                "importance": float(pat_shap[idx])
            })
            
        # Convert types for json
        s_id = int(stay_ids[i]) if hasattr(stay_ids[i], "item") else stay_ids[i]
        
        rankings.append({
            "stay_id": s_id,
            "win_id": int(df_test.iloc[i]["win_id"]),
            "top_features": drivers
        })
        
    with open(save_path / "explainability_rankings.json", "w") as f:
        json.dump(rankings, f, indent=4)

def main():
    print("Loading data...")
    test_df, feature_names = load_data_for_explanation()
    X_test = test_df[feature_names]
    
    print("Loading XGBoost model...")
    model_path = Path("C:/Users/rohit/MultiModal/AEGIS/data/models/xgboost_baseline.pkl")
    model = joblib.load(model_path)
    
    save_dir = Path("C:/Users/rohit/MultiModal/AEGIS/data/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating SHAP explanations...")
    shap_vals = explain_tree_model(model, X_test, feature_names, save_dir)
    print("Saving rankings to JSON...")
    generate_json_rankings(shap_vals, X_test, test_df, feature_names, save_dir)
    print("Explainability complete.")

if __name__ == "__main__":
    main()
