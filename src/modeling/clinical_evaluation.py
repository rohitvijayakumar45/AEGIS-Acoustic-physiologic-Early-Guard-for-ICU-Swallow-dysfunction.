import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve, IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from collections import defaultdict
import warnings

def calibrate_predictions(y_true, y_pred, method='isotonic'):
    if method == 'isotonic':
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(y_pred, y_true)
        return ir.predict(y_pred)
    elif method == 'temperature':
        lr = LogisticRegression()
        eps = 1e-15
        y_pred_clip = np.clip(y_pred, eps, 1 - eps)
        logits = np.log(y_pred_clip / (1 - y_pred_clip)).reshape(-1, 1)
        lr.fit(logits, y_true)
        return lr.predict_proba(logits)[:, 1]
    return y_pred

def plot_reliability_diagram(y_true, y_pred, save_path, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    plt.figure(figsize=(6,6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Perfect calibration')
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def compute_brier_score(y_true, y_pred):
    return brier_score_loss(y_true, y_pred)

def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95, patient_ids=None):
    if patient_ids is not None:
        unique_patients = np.unique(patient_ids)
        scores = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(unique_patients, size=len(unique_patients), replace=True)
            # Reconstruct y_true, y_pred for these patients
            sample_y_true = []
            sample_y_pred = []
            for p_id in idx:
                p_mask = (patient_ids == p_id)
                sample_y_true.extend(y_true[p_mask])
                sample_y_pred.extend(y_pred[p_mask])
            try:
                scores.append(metric_fn(np.array(sample_y_true), np.array(sample_y_pred)))
            except:
                pass
    else:
        scores = []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = np.random.choice(np.arange(n), size=n, replace=True)
            try:
                scores.append(metric_fn(y_true[idx], y_pred[idx]))
            except:
                pass
                
    scores = np.array(scores)
    scores = scores[~np.isnan(scores)]
    point_est = metric_fn(y_true, y_pred)
    
    if len(scores) == 0:
        return point_est, np.nan, np.nan
        
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return point_est, lower, upper

def delong_test(y_true, y_pred_a, y_pred_b):
    # Approximate DeLong test
    auc_a = roc_auc_score(y_true, y_pred_a)
    auc_b = roc_auc_score(y_true, y_pred_b)
    
    n1 = np.sum(y_true)
    n0 = len(y_true) - n1
    
    if n1 == 0 or n0 == 0:
        return 0.0, 1.0
        
    def q1(auc): return auc / (2 - auc)
    def q2(auc): return 2 * auc**2 / (1 + auc)
    
    def var_auc(auc):
        return (auc * (1 - auc) + (n1 - 1) * (q1(auc) - auc**2) + (n0 - 1) * (q2(auc) - auc**2)) / (n1 * n0)
    
    var_a = var_auc(auc_a)
    var_b = var_auc(auc_b)
    
    cov = 0.5 * np.sqrt(var_a * var_b)
    
    denominator = np.sqrt(var_a + var_b - 2 * cov + 1e-10)
    z = (auc_a - auc_b) / denominator
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return float(z), float(p_value)

def mcnemar_test(y_true, y_pred_a, y_pred_b, threshold=0.5):
    pred_a_bin = (y_pred_a >= threshold).astype(int)
    pred_b_bin = (y_pred_b >= threshold).astype(int)
    
    b = np.sum((pred_a_bin == 1) & (pred_b_bin == 0))
    c = np.sum((pred_a_bin == 0) & (pred_b_bin == 1))
    
    if b + c == 0:
        return 0.0, 1.0
        
    stat = (abs(b - c) - 1)**2 / (b + c)
    p_value = stats.chi2.sf(stat, 1)
    return float(stat), float(p_value)

def decision_curve_analysis(y_true, y_pred, thresholds):
    net_benefits = []
    n = len(y_true)
    for pt in thresholds:
        pred_bin = (y_pred >= pt).astype(int)
        tp = np.sum((pred_bin == 1) & (y_true == 1))
        fp = np.sum((pred_bin == 1) & (y_true == 0))
        
        if pt == 1.0:
            pt = 0.9999
        
        weight = pt / (1 - pt)
        net_benefit = (tp / n) - (fp / n) * weight
        net_benefits.append(net_benefit)
    return np.array(net_benefits)

def plot_dca(y_true, y_pred_dict, save_path):
    thresholds = np.linspace(0.01, 0.99, 100)
    plt.figure(figsize=(8,6))
    
    n = len(y_true)
    prev = np.sum(y_true) / n
    
    treat_all = [prev - (1 - prev) * (pt / (1 - pt)) for pt in thresholds]
    plt.plot(thresholds, treat_all, label='Treat All', linestyle='--')
    plt.plot(thresholds, np.zeros_like(thresholds), label='Treat None', linestyle='--')
    
    for name, y_pred in y_pred_dict.items():
        nb = decision_curve_analysis(y_true, y_pred, thresholds)
        plt.plot(thresholds, nb, label=name)
        
    plt.ylim(ymin=-0.05, ymax=prev+0.05)
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def simulate_prospective(y_true, y_pred, patient_ids, stay_ids, threshold=0.5):
    alerts = (y_pred >= threshold).astype(int)
    
    df = pd.DataFrame({
        'y_true': y_true,
        'alert': alerts,
        'patient_id': patient_ids,
        'stay_id': stay_ids
    })
    
    total_alerts = df['alert'].sum()
    true_alerts = df[(df['alert'] == 1) & (df['y_true'] == 1)].shape[0]
    
    alert_precision = true_alerts / total_alerts if total_alerts > 0 else 0
    
    avg_alerts_per_stay = total_alerts / df['stay_id'].nunique() if df['stay_id'].nunique() > 0 else 0
    
    warning_times = []
    for _, stay_df in df.groupby('stay_id'):
        alert_idx = stay_df.index[stay_df['alert'] == 1].tolist()
        event_idx = stay_df.index[stay_df['y_true'] == 1].tolist()
        if alert_idx and event_idx:
            first_alert = alert_idx[0]
            first_event = event_idx[0]
            if first_alert <= first_event:
                warning_times.append(first_event - first_alert)
                
    median_warning_time = np.median(warning_times) if warning_times else 0
    
    total_shifts = len(df) / 8.0
    alarm_burden = total_alerts / total_shifts if total_shifts > 0 else 0
    
    return {
        'alert_precision': float(alert_precision),
        'avg_alerts_per_stay': float(avg_alerts_per_stay),
        'median_warning_time_hr': float(median_warning_time),
        'alarm_burden_per_8h_shift': float(alarm_burden)
    }

def plot_learning_curves(train_losses, val_losses, aurocs, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(8,6))
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, train_losses, color='tab:blue', linestyle='-', label='Train Loss')
    ax1.plot(epochs, val_losses, color='tab:blue', linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('AUROC', color='tab:red')
    ax2.plot(epochs, aurocs, color='tab:red', linestyle='-', label='Val AUROC')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.tight_layout()
    plt.title('Learning Curves')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()

def subgroup_analysis(y_true, y_pred, demographics_df, subgroup_cols):
    overall_auroc = roc_auc_score(y_true, y_pred)
    results = {'overall_auroc': float(overall_auroc), 'subgroups': {}}
    
    for col in subgroup_cols:
        results['subgroups'][col] = {}
        unique_vals = demographics_df[col].unique()
        for val in unique_vals:
            mask = (demographics_df[col] == val)
            if np.sum(y_true[mask]) > 0 and np.sum(~y_true[mask]) > 0:
                sub_auroc = roc_auc_score(y_true[mask], y_pred[mask])
                flag = (overall_auroc - sub_auroc) > 0.05
                results['subgroups'][col][str(val)] = {
                    'auroc': float(sub_auroc),
                    'n_samples': int(np.sum(mask)),
                    'flag_drop': bool(flag)
                }
            else:
                results['subgroups'][col][str(val)] = {
                    'auroc': None,
                    'n_samples': int(np.sum(mask)),
                    'flag_drop': False
                }
    return results

def deep_ensemble_uncertainty(predictions_list):
    preds = np.stack(predictions_list, axis=0)
    mean_pred = np.mean(preds, axis=0)
    epistemic_uncertainty = np.std(preds, axis=0)
    
    eps = 1e-15
    p = np.clip(mean_pred, eps, 1 - eps)
    predictive_entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
    
    return mean_pred, epistemic_uncertainty, predictive_entropy

def save_results(results, filepath):
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    np.random.seed(42)
    N = 1000
    y_true = np.random.binomial(1, 0.2, N)
    
    y_pred = y_true * np.random.beta(5, 2, N) + (1 - y_true) * np.random.beta(2, 5, N)
    y_pred_b = y_true * np.random.beta(4, 2, N) + (1 - y_true) * np.random.beta(2, 4, N)
    
    patient_ids = np.random.randint(0, 200, N)
    stay_ids = np.random.randint(0, 250, N)
    
    out_dir = Path("c:/Users/rohit/MultiModal/AEGIS/src/modeling/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cal_pred = calibrate_predictions(y_true, y_pred)
    plot_reliability_diagram(y_true, cal_pred, out_dir / "reliability.png")
    brier = compute_brier_score(y_true, cal_pred)
    
    pt_est, lower, upper = bootstrap_ci(y_true, cal_pred, roc_auc_score, n_bootstrap=100, patient_ids=patient_ids)
    
    delong_z, delong_p = delong_test(y_true, y_pred, y_pred_b)
    mcnemar_stat, mcnemar_p = mcnemar_test(y_true, y_pred, y_pred_b)
    
    plot_dca(y_true, {'Model A': y_pred, 'Model B': y_pred_b}, out_dir / "dca.png")
    
    sim_res = simulate_prospective(y_true, y_pred, patient_ids, stay_ids)
    
    df = pd.DataFrame({'age_bin': np.random.choice(['young', 'old'], N),
                       'sex': np.random.choice(['M', 'F'], N)})
    subgroup_res = subgroup_analysis(y_true, y_pred, df, ['age_bin', 'sex'])
    
    results = {
        'brier_score': float(brier),
        'auroc': {
            'point': float(pt_est),
            'lower_ci': float(lower),
            'upper_ci': float(upper)
        },
        'delong': {'z': float(delong_z), 'p': float(delong_p)},
        'mcnemar': {'stat': float(mcnemar_stat), 'p': float(mcnemar_p)},
        'simulation': sim_res,
        'subgroups': subgroup_res
    }
    
    save_results(results, out_dir / "eval_results.json")
    print("Clinical evaluation finished. Results saved.")

if __name__ == "__main__":
    main()
