import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import duckdb
from pathlib import Path
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

def load_data_for_ablation():
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
    
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    return train_df, test_df, feature_cols

def run_feature_ablation(X_train, y_train, X_test, y_test, feature_groups, model_params, save_dir=None):
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    full_model = XGBClassifier(**model_params)
    full_model.fit(X_train, y_train)
    full_preds = full_model.predict_proba(X_test)[:, 1]
    full_auroc = roc_auc_score(y_test, full_preds)

    results = {"full_model_auroc": float(full_auroc), "ablations": {}}

    for group_name, patterns in feature_groups.items():
        cols_to_drop = [c for c in X_train.columns if any(p in c for p in patterns)]
        if not cols_to_drop:
            continue
            
        X_train_ablated = X_train.drop(columns=cols_to_drop)
        X_test_ablated = X_test.drop(columns=cols_to_drop)

        ablated_model = XGBClassifier(**model_params)
        ablated_model.fit(X_train_ablated, y_train)
        ablated_preds = ablated_model.predict_proba(X_test_ablated)[:, 1]
        
        ablated_auroc = roc_auc_score(y_test, ablated_preds)
        delta = full_auroc - ablated_auroc

        results["ablations"][group_name] = {
            "auroc": float(ablated_auroc),
            "drop": float(delta),
            "dropped_features_count": len(cols_to_drop)
        }

    if save_dir:
        with open(save_dir / "ablation_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        groups = list(results["ablations"].keys())
        drops = [results["ablations"][g]["drop"] for g in groups]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=drops, y=groups, palette="viridis")
        plt.xlabel("AUROC Drop (Full - Ablated)")
        plt.ylabel("Feature Group")
        plt.title("Feature Ablation Impact")
        plt.axvline(0, color='k', linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / "ablation_plot.png")
        plt.close()

    return results

def analyze_errors_with_uncertainty(predictions_df, threshold=0.5, save_dir=None):
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
    y_true = predictions_df["y_true"].values
    y_prob = predictions_df["XGBoost"].values
    uncertainty = predictions_df["Ensemble_variance"].values
    
    y_pred = (y_prob >= threshold).astype(int)
    
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    total = len(y_true)
    stats = {
        "TP": {"count": int(tp_mask.sum()), "prop": float(tp_mask.sum()/total), "mean_uncertainty": float(uncertainty[tp_mask].mean()) if tp_mask.sum() > 0 else 0},
        "TN": {"count": int(tn_mask.sum()), "prop": float(tn_mask.sum()/total), "mean_uncertainty": float(uncertainty[tn_mask].mean()) if tn_mask.sum() > 0 else 0},
        "FP": {"count": int(fp_mask.sum()), "prop": float(fp_mask.sum()/total), "mean_uncertainty": float(uncertainty[fp_mask].mean()) if fp_mask.sum() > 0 else 0},
        "FN": {"count": int(fn_mask.sum()), "prop": float(fn_mask.sum()/total), "mean_uncertainty": float(uncertainty[fn_mask].mean()) if fn_mask.sum() > 0 else 0}
    }

    if save_dir:
        with open(save_dir / "error_analysis_uncertainty.json", "w") as f:
            json.dump(stats, f, indent=4)
            
        labels = ["TP", "TN", "FP", "FN"]
        means = [stats[l]["mean_uncertainty"] for l in labels]
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=labels, y=means, palette="Set2")
        plt.title("Mean Prediction Uncertainty by Error Quadrant")
        plt.ylabel("Mean Ensemble Variance (Uncertainty)")
        plt.tight_layout()
        plt.savefig(save_dir / "error_uncertainty_plot.png")
        plt.close()

    return stats

def main():
    print("Loading real data for ablation...")
    train_df, test_df, feature_cols = load_data_for_ablation()
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target"].values
    
    feature_groups = {
        "HR": ["hr_"],
        "RR": ["rr_"],
        "SpO2": ["spo2_"],
        "PIP": ["pip_"],
        "Temp": ["temp_"],
        "GCS": ["gcs_"],
        "Derived Flags": ["flag"]
    }
    
    print("Loading exact training config...")
    config_path = Path("C:/Users/rohit/MultiModal/AEGIS/data/models/training_config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            model_params = json.load(f)
    else:
        print("WARNING: training_config.json not found, using default fixed params.")
        model_params = {"n_estimators": 100, "random_state": 42, "learning_rate": 0.1, "max_depth": 5}
    
    save_dir = Path("C:/Users/rohit/MultiModal/AEGIS/data/results")
    
    print("Running feature ablation...")
    ablation_res = run_feature_ablation(
        X_train, y_train, X_test, y_test, 
        feature_groups, model_params, save_dir
    )
    
    print("Running error analysis with uncertainty...")
    preds_path = Path("C:/Users/rohit/MultiModal/AEGIS/data/processed/calibrated_predictions_with_uncertainty.parquet")
    if preds_path.exists():
        preds_df = pd.read_parquet(preds_path)
        if "split" in preds_df.columns:
            preds_df = preds_df[preds_df["split"] == "test"]
            
        error_res = analyze_errors_with_uncertainty(preds_df, threshold=0.5, save_dir=save_dir)
    else:
        print(f"Error: {preds_path} not found.")
    
    print("Done.")

if __name__ == "__main__":
    main()
