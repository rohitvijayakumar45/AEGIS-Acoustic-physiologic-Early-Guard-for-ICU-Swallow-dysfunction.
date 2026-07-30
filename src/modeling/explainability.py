import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from captum.attr import IntegratedGradients
from pathlib import Path

def plot_feature_importance(importances, feature_names, save_path, title="Feature Importance"):
    """Plot generic bar chart."""
    plt.figure(figsize=(10, 6))
    indices = np.argsort(np.abs(importances))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def top_k_features(attributions, feature_names, k=5):
    """Return top-k contributing features."""
    indices = np.argsort(np.abs(attributions))[::-1][:k]
    return [(feature_names[i], attributions[i]) for i in indices]

def explain_tree_model(model, X_test, feature_names, save_path):
    """Explain tabular models using TreeSHAP."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Handle list return for some models (e.g., RandomForest classifier)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] # Assume binary classification, take positive class
        
    np.save(save_path / "shap_values.npy", shap_values)
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.savefig(save_path / "shap_summary.png", bbox_inches='tight')
    plt.close()
    
    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig(save_path / "shap_bar.png", bbox_inches='tight')
    plt.close()
    
    # Top 5 dependence plots
    global_importances = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(global_importances)[::-1][:5]
    for idx in top_indices:
        plt.figure()
        shap.dependence_plot(idx, shap_values, X_test, feature_names=feature_names, show=False)
        plt.savefig(save_path / f"shap_dep_{feature_names[idx]}.png", bbox_inches='tight')
        plt.close()
        
    return shap_values

def explain_deep_model(model, X_test, feature_names, save_path, target_class=1):
    """Explain deep models using Integrated Gradients."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    if not isinstance(X_test, torch.Tensor):
        X_test = torch.tensor(X_test, dtype=torch.float32)
        
    X_test.requires_grad_()
    
    ig = IntegratedGradients(model)
    attributions = ig.attribute(X_test, target=target_class)
    attr_np = attributions.detach().cpu().numpy()
    
    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(attr_np, xticklabels=feature_names, cmap="coolwarm", center=0)
    plt.title("Integrated Gradients Attributions Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(save_path / "ig_heatmap.png")
    plt.close()
    
    # Feature Importance Bar
    global_attr = np.mean(attr_np, axis=0)
    plot_feature_importance(global_attr, feature_names, save_path / "ig_bar.png", "IG Feature Importance")
    
    return attr_np

def generate_patient_report(patient_id, prediction, attributions, feature_names, feature_values):
    """Output human-readable text explaining the prediction."""
    risk_level = "HIGH RISK" if prediction > 0.5 else "LOW RISK"
    top_features = top_k_features(attributions, feature_names, k=3)
    
    drivers = []
    for name, attr in top_features:
        idx = feature_names.index(name)
        val = feature_values[idx]
        drivers.append(f"{name} (val: {val:.2f}, attr: {attr:.2f})")
        
    drivers_str = ", ".join(drivers)
    report = f"Patient {patient_id}: {risk_level} ({prediction:.2f}). Primary drivers: {drivers_str}."
    return report

def main():
    """Synthetic demo."""
    import xgboost as xgb
    
    print("Running explainability demo...")
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Synthetic data
    N = 100
    D = 10
    feature_names = [f"Feature_{i}" for i in range(D)]
    feature_names[0] = "SpO2"
    feature_names[1] = "RR"
    feature_names[2] = "GCS"
    
    X = np.random.randn(N, D)
    y = (X[:, 0] * -1.5 + X[:, 1] * 1.2 + X[:, 2] * -0.8 + np.random.randn(N) * 0.5 > 0).astype(int)
    
    # 1. TreeSHAP Demo
    tree_dir = Path("./demo_output/tree")
    tree_model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    tree_model.fit(X, y)
    print("Tree model fitted. Generating SHAP explanations...")
    shap_vals = explain_tree_model(tree_model, X, feature_names, tree_dir)
    print(f"SHAP saved to {tree_dir}")
    
    # 2. Integrated Gradients Demo
    deep_dir = Path("./demo_output/deep")
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(D, 16)
            self.fc2 = nn.Linear(16, 2) # output classes
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
            
    nn_model = SimpleNN()
    print("Deep model initialized. Generating IG explanations...")
    ig_vals = explain_deep_model(nn_model, X, feature_names, deep_dir, target_class=1)
    print(f"IG saved to {deep_dir}")
    
    # 3. Patient Report Demo
    idx = 0
    pred = 0.91 # Mock prediction
    patient_id = "P10293"
    report = generate_patient_report(patient_id, pred, shap_vals[idx], feature_names, X[idx])
    print("\nSample Patient Report:")
    print(report)

if __name__ == "__main__":
    main()
