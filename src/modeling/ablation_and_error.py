import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier

def run_feature_ablation(model_class, X_train, y_train, X_test, y_test, feature_groups, model_params=None, save_dir=None):
    """
    Run ablation. Drop one group at a time. Return results.
    """
    if model_params is None:
        model_params = {}
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Full model
    full_model = model_class(**model_params)
    full_model.fit(X_train, y_train)
    full_preds = full_model.predict_proba(X_test)[:, 1]
    full_auroc = roc_auc_score(y_test, full_preds)

    results = {"full_model_auroc": full_auroc, "ablations": {}}

    for group_name, patterns in feature_groups.items():
        cols_to_drop = [c for c in X_train.columns if any(p in c for p in patterns)]
        
        if not cols_to_drop:
            continue
            
        X_train_ablated = X_train.drop(columns=cols_to_drop)
        X_test_ablated = X_test.drop(columns=cols_to_drop)

        ablated_model = model_class(**model_params)
        ablated_model.fit(X_train_ablated, y_train)
        ablated_preds = ablated_model.predict_proba(X_test_ablated)[:, 1]
        
        ablated_auroc = roc_auc_score(y_test, ablated_preds)
        delta = full_auroc - ablated_auroc

        results["ablations"][group_name] = {
            "auroc": ablated_auroc,
            "drop": delta,
            "dropped_features_count": len(cols_to_drop)
        }

    if save_dir:
        with open(save_dir / "ablation_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        # Plot
        groups = list(results["ablations"].keys())
        drops = [results["ablations"][g]["drop"] for g in groups]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=drops, y=groups, hue=groups, palette="viridis", legend=False)
        plt.xlabel("AUROC Drop (Full - Ablated)")
        plt.ylabel("Feature Group")
        plt.title("Feature Ablation Impact")
        plt.axvline(0, color='k', linestyle='--')
        plt.tight_layout()
        plt.savefig(save_dir / "ablation_plot.png")
        plt.close()

    return results

def analyze_errors(y_true, y_prob, features_df, demographics_df, threshold=0.5, save_dir=None):
    """
    Analyze errors. Compare FP/TP and FN/TN.
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
    y_pred = (y_prob >= threshold).astype(int)
    
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    total = len(y_true)
    stats = {
        "TP": {"count": int(tp_mask.sum()), "prop": float(tp_mask.sum()/total)},
        "TN": {"count": int(tn_mask.sum()), "prop": float(tn_mask.sum()/total)},
        "FP": {"count": int(fp_mask.sum()), "prop": float(fp_mask.sum()/total)},
        "FN": {"count": int(fn_mask.sum()), "prop": float(fn_mask.sum()/total)}
    }

    results = {"counts": stats, "fp_analysis": {}, "fn_analysis": {}, "demographics": {}}

    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    # FP vs TP
    for col in numeric_cols:
        fp_vals = features_df.loc[fp_mask, col].dropna()
        tp_vals = features_df.loc[tp_mask, col].dropna()
        if len(fp_vals) > 5 and len(tp_vals) > 5:
            stat, pval = ks_2samp(fp_vals, tp_vals)
            if pval < 0.05:
                results["fp_analysis"][col] = {"ks_stat": stat, "p_val": pval, "fp_mean": fp_vals.mean(), "tp_mean": tp_vals.mean()}

    # FN vs TN
    for col in numeric_cols:
        fn_vals = features_df.loc[fn_mask, col].dropna()
        tn_vals = features_df.loc[tn_mask, col].dropna()
        if len(fn_vals) > 5 and len(tn_vals) > 5:
            stat, pval = ks_2samp(fn_vals, tn_vals)
            if pval < 0.05:
                results["fn_analysis"][col] = {"ks_stat": stat, "p_val": pval, "fn_mean": fn_vals.mean(), "tn_mean": tn_vals.mean()}

    # Demographics
    for demo_col in demographics_df.columns:
        results["demographics"][demo_col] = {
            "FP": demographics_df.loc[fp_mask, demo_col].value_counts().to_dict(),
            "FN": demographics_df.loc[fn_mask, demo_col].value_counts().to_dict()
        }

    if save_dir:
        with open(save_dir / "error_analysis.json", "w") as f:
            json.dump(results, f, indent=4)
            
        # Plot top 3 discrepant features for FP vs TP
        top_fp = sorted(results["fp_analysis"].items(), key=lambda x: x[1]["ks_stat"], reverse=True)[:3]
        for col, _ in top_fp:
            plt.figure(figsize=(6, 4))
            sns.kdeplot(features_df.loc[fp_mask, col], label='FP', fill=True)
            sns.kdeplot(features_df.loc[tp_mask, col], label='TP', fill=True)
            plt.title(f"FP vs TP: {col}")
            plt.legend()
            plt.savefig(save_dir / f"fp_vs_tp_{col}.png")
            plt.close()

    return results

def generate_ablation_report(ablation_results, error_results, save_path):
    """
    Generate markdown report.
    """
    save_path = Path(save_path)
    
    report = f"# AEGIS Ablation and Error Analysis Report\n\n"
    report += f"## Ablation Results\n"
    report += f"- **Full Model AUROC**: {ablation_results['full_model_auroc']:.4f}\n\n"
    report += "| Feature Group | AUROC | Drop |\n|---|---|---|\n"
    
    for grp, data in ablation_results['ablations'].items():
        report += f"| {grp} | {data['auroc']:.4f} | {data['drop']:.4f} |\n"
        
    report += f"\n## Error Analysis\n"
    report += f"Counts:\n"
    for k, v in error_results['counts'].items():
        report += f"- **{k}**: {v['count']} ({v['prop']:.2%})\n"
        
    report += f"\n### Key FP Patterns (KS test vs TP)\n"
    for col, data in list(error_results['fp_analysis'].items())[:5]:
        report += f"- **{col}**: FP mean {data['fp_mean']:.2f} vs TP mean {data['tp_mean']:.2f} (p={data['p_val']:.4f})\n"

    report += f"\n### Key FN Patterns (KS test vs TN)\n"
    for col, data in list(error_results['fn_analysis'].items())[:5]:
        report += f"- **{col}**: FN mean {data['fn_mean']:.2f} vs TN mean {data['tn_mean']:.2f} (p={data['p_val']:.4f})\n"
        
    with open(save_path, "w") as f:
        f.write(report)

def main():
    # Synthetic demo
    np.random.seed(42)
    n_samples = 1000
    
    # Fake features
    features = pd.DataFrame({
        "hr_mean": np.random.normal(80, 15, n_samples),
        "hr_max": np.random.normal(100, 20, n_samples),
        "rr_mean": np.random.normal(18, 4, n_samples),
        "spo2_min": np.random.normal(92, 5, n_samples),
        "pip_max": np.random.normal(25, 5, n_samples),
        "temp_mean": np.random.normal(37.5, 1, n_samples),
        "gcs_min": np.random.randint(3, 15, n_samples),
        "spo2_drop_flag": np.random.randint(0, 2, n_samples),
        "tachypnea_flag": np.random.randint(0, 2, n_samples),
        "bradycardia_flag": np.random.randint(0, 2, n_samples),
        "fever_flag": np.random.randint(0, 2, n_samples)
    })
    
    y = np.random.randint(0, 2, n_samples)
    
    demos = pd.DataFrame({
        "age": np.random.randint(18, 90, n_samples),
        "sex": np.random.choice(["M", "F"], n_samples),
        "icu_type": np.random.choice(["MICU", "SICU", "NeuroICU"], n_samples)
    })
    
    X_train, X_test = features.iloc[:800], features.iloc[800:]
    y_train, y_test = y[:800], y[800:]
    
    feature_groups = {
        "HR": ["hr_"],
        "RR": ["rr_"],
        "SpO2": ["spo2_"],
        "PIP": ["pip_"],
        "Temp": ["temp_"],
        "GCS": ["gcs_"],
        "Derived Flags": ["flag"]
    }
    
    save_dir = Path("c:/Users/rohit/MultiModal/AEGIS/data/results")
    
    print("Run ablation...")
    ablation_res = run_feature_ablation(
        RandomForestClassifier, X_train, y_train, X_test, y_test, 
        feature_groups, {"n_estimators": 10, "random_state": 42}, save_dir
    )
    
    # Generate fake predictions for error analysis
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("Run error analysis...")
    error_res = analyze_errors(y_test, y_prob, X_test, demos.iloc[800:], 0.5, save_dir)
    
    print("Generate report...")
    generate_ablation_report(ablation_res, error_res, save_dir / "ablation_report.md")
    print("Done.")

if __name__ == "__main__":
    main()
