import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_femh(path: Path):
    logger.info(f"Load FEMH from {path}")
    # codebook_file = path / "codebook.xlsx"
    # Dummy load. Real implementation needs specific feature extraction pipeline.
    features = np.random.randn(100, 50)
    labels = np.random.randint(0, 2, 100)
    return features, labels

def load_svd(path: Path):
    logger.info(f"Load SVD from {path}")
    # Dummy load.
    features = np.random.randn(150, 50)
    labels = np.random.randint(0, 2, 150)
    return features, labels

def load_swallowing_data(path: Path):
    logger.info(f"Load Swallowing_data from {path}")
    labels_dir = path / "labels"
    signals_dir = path / "signals"
    
    # Process NPY and CSVs here.
    # Dummy load for template.
    features = np.random.randn(200, 50)
    labels = np.random.randint(0, 2, 200)
    return features, labels

def load_external_dataset(dataset_name: str, dataset_path: Path):
    if dataset_name == "FEMH":
        return load_femh(dataset_path)
    elif dataset_name == "SVD":
        return load_svd(dataset_path)
    elif dataset_name == "Swallowing_data":
        return load_swallowing_data(dataset_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def evaluate_frozen_model(model, features, labels):
    logger.info("Eval frozen model.")
    if hasattr(model, "predict"):
        preds = model.predict(features)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        return {"accuracy": float(acc), "f1": float(f1)}
    return {"accuracy": 0.0, "f1": 0.0}

def evaluate_with_adaptation(model, features, labels):
    logger.info("Eval with adaptation.")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.8, random_state=42)
    if hasattr(model, "fit"):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        return {"accuracy": float(acc), "f1": float(f1)}
    return {"accuracy": 0.0, "f1": 0.0}

def normalize_features(source_features, target_features):
    logger.info("Normalize target features with source stats.")
    mean = np.mean(source_features, axis=0)
    std = np.std(source_features, axis=0)
    std[std == 0] = 1e-6
    return (target_features - mean) / std

def check_domain_shift(source_features, target_features):
    logger.info("Compute domain shift (KL div).")
    def _to_prob(feats):
        hist, _ = np.histogram(feats.flatten(), bins=50, density=True)
        hist = hist + 1e-8
        return hist / np.sum(hist)
    
    p = _to_prob(source_features)
    q = _to_prob(target_features)
    kl_div = entropy(p, q)
    return {"kl_divergence": float(kl_div)}

def main():
    base_out = Path(r"c:\Users\rohit\MultiModal\AEGIS\data\models")
    base_out.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "FEMH": Path(r"c:\Users\rohit\MultiModal\dataset\FEMH"),
        "SVD": Path(r"c:\Users\rohit\MultiModal\dataset\SVD_data\SVD"),
        "Swallowing_data": Path(r"c:\Users\rohit\MultiModal\dataset\Swallowing_data\Swallowing data")
    }
    
    # Mock source model/data
    source_features = np.random.randn(1000, 50)
    model = RandomForestClassifier(random_state=42)
    model.fit(source_features, np.random.randint(0, 2, 1000))
    
    results = {}
    
    for name, path in datasets.items():
        try:
            feats, labels = load_external_dataset(name, path)
            norm_feats = normalize_features(source_features, feats)
            shift = check_domain_shift(source_features, feats)
            
            frozen_res = evaluate_frozen_model(model, norm_feats, labels)
            
            # Re-init model for fresh adaptation
            adapt_model = RandomForestClassifier(random_state=42)
            adapt_res = evaluate_with_adaptation(adapt_model, norm_feats, labels)
            
            results[name] = {
                "domain_shift": shift,
                "frozen_metrics": frozen_res,
                "adapted_metrics": adapt_res
            }
        except Exception as e:
            logger.error(f"Error processing {name}: {e}")
            
    out_file = base_out / "external_validation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Save results to {out_file}")

if __name__ == "__main__":
    main()
