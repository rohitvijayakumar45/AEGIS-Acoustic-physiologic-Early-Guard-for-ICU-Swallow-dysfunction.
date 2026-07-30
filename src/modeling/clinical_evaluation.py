import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, precision_recall_curve,
    roc_curve, auc, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_totals, _ = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1))
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals[non_empty_bins] / len(y_prob)
    return float(np.sum(bin_weights * np.abs(prob_true - prob_pred)))

class TemperatureScaling:
    def __init__(self):
        self.temperature = 1.0
        
    def fit(self, y_true, probs):
        eps = 1e-7
        p = np.clip(probs, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        
        def objective(temp):
            scaled_logits = logits / temp[0]
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            scaled_probs = np.clip(scaled_probs, eps, 1 - eps)
            return -np.mean(y_true * np.log(scaled_probs) + (1 - y_true) * np.log(1 - scaled_probs))
            
        res = minimize(objective, [1.0], bounds=[(0.01, 10.0)])
        self.temperature = res.x[0]
        return self
        
    def predict_proba(self, probs):
        eps = 1e-7
        p = np.clip(probs, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        scaled_logits = logits / self.temperature
        return 1 / (1 + np.exp(-scaled_logits))

def net_benefit(y_true, y_pred, pt):
    n = len(y_true)
    if n == 0: return 0.0
    pred_bin = (y_pred >= pt).astype(int)
    tp = np.sum((pred_bin == 1) & (y_true == 1))
    fp = np.sum((pred_bin == 1) & (y_true == 0))
    if pt == 1.0:
        pt = 0.9999
    weight = pt / (1 - pt)
    return (tp / n) - (fp / n) * weight

def find_operating_point(y_true, y_prob, mode):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    
    if mode == 'sensitivity':
        for pt in reversed(thresholds):
            pred = (y_prob >= pt).astype(int)
            if recall_score(y_true, pred, zero_division=0) >= 0.95:
                best_thresh = pt
                break
    elif mode == 'precision':
        for pt in thresholds:
            pred = (y_prob >= pt).astype(int)
            if precision_score(y_true, pred, zero_division=0) >= 0.80:
                best_thresh = pt
                break
    elif mode == 'f1':
        best_f1 = -1
        for pt in thresholds:
            pred = (y_prob >= pt).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = pt
    elif mode == 'net_benefit':
        best_nb = -float('inf')
        for pt in thresholds:
            nb = net_benefit(y_true, y_prob, pt)
            if nb > best_nb:
                best_nb = nb
                best_thresh = pt
    return best_thresh

def calculate_metrics(y_true, y_prob, threshold):
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = precision_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    nb = net_benefit(y_true, y_prob, threshold)
    return sens, spec, prec, f1, nb

def bootstrap_ci(y_true, y_prob, threshold, n_bootstrap=100):
    metrics = {'sens': [], 'spec': [], 'prec': [], 'f1': [], 'nb': []}
    n = len(y_true)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        s, sp, pr, f, n_b = calculate_metrics(y_true[idx], y_prob[idx], threshold)
        metrics['sens'].append(s)
        metrics['spec'].append(sp)
        metrics['prec'].append(pr)
        metrics['f1'].append(f)
        metrics['nb'].append(n_b)
    
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in metrics.items()}

def plot_calibration(y_true, y_prob_before, y_prob_after, model_name, ax):
    prob_true_b, prob_pred_b = calibration_curve(y_true, y_prob_before, n_bins=10)
    prob_true_a, prob_pred_a = calibration_curve(y_true, y_prob_after, n_bins=10)
    ax.plot(prob_pred_b, prob_true_b, marker='o', label='Before')
    ax.plot(prob_pred_a, prob_true_a, marker='s', label='After')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_title(f'Calibration: {model_name}')
    ax.legend()

def main():
    data_dir = Path("c:/Users/rohit/MultiModal/AEGIS/data")
    out_dir = data_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = data_dir / "processed" / "raw_predictions.parquet"
    if not raw_path.exists():
        print(f"File not found: {raw_path}")
        return
        
    df = pd.read_parquet(raw_path)
    
    # Pivot from long to wide format
    if 'model_name' in df.columns:
        df = df.pivot_table(index=['stay_id', 'split', 'y_true'], 
                            columns='model_name', 
                            values='y_prob').reset_index()
                            
    df = df.dropna()
    
    model_cols = [c for c in df.columns if c not in ['stay_id', 'split', 'y_true']]
        
    df_val = df[df['split'] == 'val'].copy()
    df_test = df[df['split'] == 'test'].copy()
    
    scalers = {}
    calibrated_val = pd.DataFrame(index=df_val.index)
    calibrated_test = pd.DataFrame(index=df_test.index)
    
    fig_cal, axes_cal = plt.subplots(1, len(model_cols), figsize=(5*len(model_cols), 5))
    if len(model_cols) == 1: axes_cal = [axes_cal]
    
    for i, m in enumerate(model_cols):
        y_val = df_val['y_true'].values
        p_val = df_val[m].values
        
        ece_before = expected_calibration_error(y_val, p_val)
        brier_before = brier_score_loss(y_val, p_val)
        
        ts = TemperatureScaling()
        ts.fit(y_val, p_val)
        scalers[m] = ts
        
        p_val_cal = ts.predict_proba(p_val)
        calibrated_val[m] = p_val_cal
        
        ece_after = expected_calibration_error(y_val, p_val_cal)
        brier_after = brier_score_loss(y_val, p_val_cal)
        print(f"{m} Calibration - ECE: {ece_before:.4f} -> {ece_after:.4f}, Brier: {brier_before:.4f} -> {brier_after:.4f}")
        
        p_test = df_test[m].values
        calibrated_test[m] = ts.predict_proba(p_test)
        
        plot_calibration(y_val, p_val, p_val_cal, m, axes_cal[i])
        
    fig_cal.savefig(out_dir / 'calibration_curve.png')
    plt.close(fig_cal)
    
    if len(model_cols) > 1:
        calibrated_val['Ensemble'] = calibrated_val[model_cols].mean(axis=1)
        calibrated_test['Ensemble'] = calibrated_test[model_cols].mean(axis=1)
        eval_models = model_cols + ['Ensemble']
    else:
        eval_models = model_cols
    
    df_cal = df.copy()
    for m in model_cols:
        p_all = df[m].values
        df_cal[m] = scalers[m].predict_proba(p_all)
    df_cal['Ensemble_mean'] = df_cal[model_cols].mean(axis=1) if len(model_cols) > 1 else df_cal[model_cols[0]]
    df_cal['Ensemble_variance'] = df_cal[model_cols].var(axis=1) if len(model_cols) > 1 else 0.0
    df_cal.to_parquet(data_dir / "processed" / "calibrated_predictions_with_uncertainty.parquet")
    
    modes = ['sensitivity', 'precision', 'f1', 'net_benefit']
    operating_points = {}
    
    for m in eval_models:
        operating_points[m] = {}
        for mode in modes:
            best_th = find_operating_point(df_val['y_true'].values, calibrated_val[m].values, mode)
            operating_points[m][mode] = float(best_th)
            
    with open(out_dir / "operating_points.json", "w") as f:
        json.dump(operating_points, f, indent=4)
        
    summary_data = []
    for m in eval_models:
        y_test = df_test['y_true'].values
        p_test = calibrated_test[m].values
        
        precision, recall, _ = precision_recall_curve(y_test, p_test)
        plt.figure()
        plt.plot(recall, precision)
        plt.title(f"{m} PR Curve")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(out_dir / f"{m}_precision_recall_curve.png")
        plt.close()
        
        fpr, tpr, _ = roc_curve(y_test, p_test)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"{m} ROC Curve")
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.savefig(out_dir / f"{m}_roc_curve.png")
        plt.close()
        
        thresholds_dca = np.linspace(0.01, 0.99, 100)
        nb = [net_benefit(y_test, p_test, pt) for pt in thresholds_dca]
        plt.figure()
        plt.plot(thresholds_dca, nb)
        plt.title(f"{m} Decision Curve")
        plt.xlabel('Threshold')
        plt.ylabel('Net Benefit')
        plt.savefig(out_dir / f"{m}_decision_curve.png")
        plt.close()
        
        for mode, th in operating_points[m].items():
            sens, spec, prec, f1, nb_val = calculate_metrics(y_test, p_test, th)
            cis = bootstrap_ci(y_test, p_test, th, n_bootstrap=50) # kept low for speed
            
            summary_data.append({
                'Model': m,
                'Operating Point': mode,
                'Threshold': th,
                'Sensitivity': f"{sens:.3f} ({cis['sens'][0]:.3f}-{cis['sens'][1]:.3f})",
                'Specificity': f"{spec:.3f} ({cis['spec'][0]:.3f}-{cis['spec'][1]:.3f})",
                'Precision': f"{prec:.3f} ({cis['prec'][0]:.3f}-{cis['prec'][1]:.3f})",
                'F1': f"{f1:.3f} ({cis['f1'][0]:.3f}-{cis['f1'][1]:.3f})",
                'Net Benefit': f"{nb_val:.3f} ({cis['nb'][0]:.3f}-{cis['nb'][1]:.3f})"
            })
            
            pred = (p_test >= th).astype(int)
            cm = confusion_matrix(y_test, pred)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{m} - {mode} (th={th:.2f})")
            plt.savefig(out_dir / f"{m}_{mode}_confusion_matrix.png")
            plt.close()
            
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(out_dir / "threshold_summary_table.csv", index=False)
    print("Clinical evaluation complete.")

if __name__ == "__main__":
    main()
