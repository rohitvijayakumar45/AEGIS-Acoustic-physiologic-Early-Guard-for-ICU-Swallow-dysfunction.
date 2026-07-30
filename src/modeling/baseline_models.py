import argparse
import json
import logging
import os
import random
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
SEED = 42
DATA_DIR = Path("C:/Users/rohit/MultiModal/AEGIS/data")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_and_prep_data(max_rows=None):
    logger.info("Load data.")
    processed_dir = DATA_DIR / "processed"
    
    features_df = pd.read_parquet(processed_dir / "physiological_features.parquet")
    labels_df = pd.read_csv(processed_dir / "ground_truth_labels.csv")
    splits_df = pd.read_csv(processed_dir / "split_assignments.csv")
    
    if max_rows:
        features_df = features_df.head(max_rows)
        
    logger.info("Join data.")
    # Assuming labels have stay_id, event_time, event_type
    # Assuming features have stay_id, window_end_time
    import duckdb
    
    conn = duckdb.connect()
    conn.register("features_df", features_df)
    conn.register("labels_df", labels_df)
    conn.register("splits_df", splits_df)
    
    # cohort to get intime
    cohort_path = "C:/Users/rohit/MultiModal/AEGIS/data/processed/target_cohort.csv"
    
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
    
    # Feature columns
    feature_cols = [
        col for col in df.columns 
        if col.endswith(("_mean", "_min_val", "_max_val", "_std", "_slope", "_obs_count"))
    ]
    derived_flags = ["spo2_drop_flag", "tachypnea_flag", "bradycardia_flag", "fever_flag"]
    feature_cols.extend([col for col in derived_flags if col in df.columns])
    
    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(0)
    
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    
    return train_df, val_df, test_df, feature_cols


def evaluate_model(y_true, y_prob, stay_ids=None, split=None, model_name=None):
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = 0.5
        
    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = 0.0
        
    brier = brier_score_loss(y_true, y_prob)
    
    if stay_ids is not None and split is not None and model_name is not None:
        raw_df = pd.DataFrame({
            "stay_id": stay_ids,
            "split": split,
            "model_name": model_name,
            "y_true": y_true,
            "y_prob": y_prob
        })
        out_path = Path("C:/Users/rohit/MultiModal/AEGIS/data/processed/raw_predictions.parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if out_path.exists():
            existing_df = pd.read_parquet(out_path)
            raw_df = pd.concat([existing_df, raw_df], ignore_index=True)
            
        raw_df.to_parquet(out_path, index=False)
    
    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "Brier": float(brier)
    }

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.BCELoss(reduction='none')(inputs, targets)
        p_t = torch.exp(-bce)
        loss = self.alpha * (1 - p_t)**self.gamma * bce
        return loss.mean()


def heuristic_baseline(df):
    logger.info("Run heuristic baseline.")
    spo2 = df.get("spo2_drop_flag", pd.Series(np.zeros(len(df))))
    tachy = df.get("tachypnea_flag", pd.Series(np.zeros(len(df))))
    
    preds = (spo2 == 1) | (tachy == 1)
    return preds.astype(float).values


def train_logistic_regression(X_train, y_train, X_test):
    logger.info("Train Logistic Regression.")
    model = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=1000)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def train_random_forest(X_train, y_train, X_test):
    logger.info("Train Random Forest.")
    model = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_estimators=100, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def optimize_xgboost(X_train, y_train, groups, n_trials=20):
    logger.info("Optimize XGBoost.")
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0

    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": SEED,
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 2, 5, 10, 20, 40, 60, 80]),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        
        cv = GroupKFold(n_splits=3)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train, groups):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "auc"
    best_params["tree_method"] = "hist"
    best_params["random_state"] = SEED
    best_params["scale_pos_weight"] = scale_pos_weight
    
    return best_params


def train_xgboost(X_train, y_train, groups, X_test, n_trials=20):
    best_params = optimize_xgboost(X_train, y_train, groups, n_trials)
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def optimize_lightgbm(X_train, y_train, groups, n_trials=20):
    logger.info("Optimize LightGBM.")
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0

    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "random_state": SEED,
            "scale_pos_weight": trial.suggest_categorical("scale_pos_weight", [1, 2, 5, 10, 20, 40, 60, 80]),
            "verbose": -1,
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        
        cv = GroupKFold(n_splits=3)
        scores = []
        
        for train_idx, val_idx in cv.split(X_train, y_train, groups):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
            
            preds = model.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y_va, preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_params["objective"] = "binary"
    best_params["metric"] = "auc"
    best_params["random_state"] = SEED
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["verbose"] = -1
    
    return best_params


def train_lightgbm(X_train, y_train, groups, X_test, n_trials=20):
    best_params = optimize_lightgbm(X_train, y_train, groups, n_trials)
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


class SequenceDataset(Dataset):
    def __init__(self, df, feature_cols, max_len=24):
        self.samples = []
        grouped = df.groupby("stay_id")
        
        for stay_id, group in grouped:
            # Sort by time if window column exists
            if "window" in group.columns:
                group = group.sort_values("window")
            
            features = group[feature_cols].values
            targets = group["target"].values
            
            # Pad or truncate
            if len(features) > max_len:
                features = features[-max_len:]
                targets = targets[-max_len:]
            else:
                pad_len = max_len - len(features)
                features = np.pad(features, ((0, pad_len), (0, 0)), mode="constant")
                targets = np.pad(targets, (0, pad_len), mode="constant")
                
            self.samples.append((torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32), stay_id))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return torch.sigmoid(out).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x is (batch, seq_len, input_size)
        x = x.transpose(1, 2) # (batch, input_size, seq_len)
        out = self.network(x)
        out = out.transpose(1, 2)
        out = self.fc(out)
        return torch.sigmoid(out).squeeze(-1)


def train_dl_model(model, train_loader, val_loader, epochs=10, pos_weight=1.0, use_focal=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if use_focal:
        criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)
    else:
        criterion = nn.BCELoss(weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            
            # Mask out padding targets (assuming pad values are 0 which might mix with real 0s, 
            # ideally use a mask, but simple BCE here)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
    return model


def evaluate_dl_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_targets = []
    all_stay_ids = []
    
    with torch.no_grad():
        for X_batch, y_batch, stay_ids_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(y_batch.numpy().flatten())
            
            batch_size, seq_len = preds.shape
            for sid in stay_ids_batch:
                all_stay_ids.extend([sid] * seq_len)
            
    return np.concatenate(all_preds), np.concatenate(all_targets), np.array(all_stay_ids)


def main(max_rows=None, epochs=10, n_trials=20):
    set_seed()
    
    train_df, val_df, test_df, feature_cols = load_and_prep_data(max_rows)
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    groups_train = train_df["stay_id"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["target"].values
    
    X_val = val_df[feature_cols]
    y_val = val_df["target"].values
    
    # Optional: Clear old predictions file to start fresh for this run
    raw_path = Path("C:/Users/rohit/MultiModal/AEGIS/data/processed/raw_predictions.parquet")
    if raw_path.exists():
        raw_path.unlink()

    results = {}
    
    # 1. Heuristic
    preds_heuristic_val = heuristic_baseline(val_df)
    evaluate_model(y_val, preds_heuristic_val, val_df["stay_id"].values, "val", "Heuristic")
    preds_heuristic = heuristic_baseline(test_df)
    results["Heuristic"] = evaluate_model(y_test, preds_heuristic, test_df["stay_id"].values, "test", "Heuristic")
    
    # 2. Logistic Regression
    preds_lr_val = train_logistic_regression(X_train, y_train, X_val)
    evaluate_model(y_val, preds_lr_val, val_df["stay_id"].values, "val", "Logistic Regression")
    preds_lr = train_logistic_regression(X_train, y_train, X_test)
    results["Logistic Regression"] = evaluate_model(y_test, preds_lr, test_df["stay_id"].values, "test", "Logistic Regression")
    
    # 3. Random Forest
    preds_rf_val = train_random_forest(X_train, y_train, X_val)
    evaluate_model(y_val, preds_rf_val, val_df["stay_id"].values, "val", "Random Forest")
    preds_rf = train_random_forest(X_train, y_train, X_test)
    results["Random Forest"] = evaluate_model(y_test, preds_rf, test_df["stay_id"].values, "test", "Random Forest")
    
    # 4. XGBoost
    best_xgb_params = optimize_xgboost(X_train, y_train, groups_train, n_trials)
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(X_train, y_train)
    preds_xgb_val = xgb_model.predict_proba(X_val)[:, 1]
    evaluate_model(y_val, preds_xgb_val, val_df["stay_id"].values, "val", "XGBoost")
    preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
    results["XGBoost"] = evaluate_model(y_test, preds_xgb, test_df["stay_id"].values, "test", "XGBoost")
    
    # 5. LightGBM
    preds_lgb_val = train_lightgbm(X_train, y_train, groups_train, X_val, n_trials)
    evaluate_model(y_val, preds_lgb_val, val_df["stay_id"].values, "val", "LightGBM")
    preds_lgb = train_lightgbm(X_train, y_train, groups_train, X_test, n_trials)
    results["LightGBM"] = evaluate_model(y_test, preds_lgb, test_df["stay_id"].values, "test", "LightGBM")
    
    # DL Data Prep
    train_dataset = SequenceDataset(train_df, feature_cols)
    val_dataset = SequenceDataset(val_df, feature_cols)
    test_dataset = SequenceDataset(test_df, feature_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    
    # Helper to train and evaluate DL models on val and test
    def run_dl(name, model, use_focal=False):
        logger.info(f"Train {name}.")
        trained = train_dl_model(model, train_loader, val_loader, epochs, pos_weight, use_focal=use_focal)
        preds_val, y_val_dl, stay_ids_val = evaluate_dl_model(trained, val_loader)
        evaluate_model(y_val_dl, preds_val, stay_ids_val, "val", name)
        
        preds_test, y_test_dl, stay_ids_test = evaluate_dl_model(trained, test_loader)
        results[name] = evaluate_model(y_test_dl, preds_test, stay_ids_test, "test", name)
        return trained

    # 6. LSTM
    lstm_model = LSTMModel(input_dim=len(feature_cols))
    lstm_trained = run_dl("LSTM", lstm_model, use_focal=True)
    
    # 7. TCN
    tcn_model = TCNModel(input_size=len(feature_cols), num_channels=[64, 64, 64])
    run_dl("TCN", tcn_model, use_focal=True)
    
    # Save results
    out_dir = DATA_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best models
    joblib.dump(xgb_model, out_dir / "xgboost_baseline.pkl")
    torch.save(lstm_trained, out_dir / "lstm_baseline.pt")
    
    # Save training config
    training_config = {
        "seed": SEED,
        "feature_list": feature_cols,
        "xgboost_best_params": best_xgb_params
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=4)
    
    out_path = out_dir / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Results saved to {out_path}")
    logger.info(json.dumps(results, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=20)
    args = parser.parse_args()
    
    main(args.max_rows, args.epochs, args.n_trials)
