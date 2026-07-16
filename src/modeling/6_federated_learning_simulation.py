"""
Phase 6: Federated Learning Simulation.

Partitions ICU stays into virtual hospitals and compares:
- centralized training
- FedAvg-style federated training

Uses the installed Flower package as the FL framework dependency, while keeping the
simulation local and Ray-free for Windows reliability.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import flwr
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[1]))
from modeling.multimodal_fusion import ICUFusionModel  # noqa: E402


PROJECT_DIR = Path(r"C:\Users\rohit\MultiModal\icu_predictive_system")
DATA_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PHYSIO_FEATURES = [
    "spo2_4h_mean",
    "spo2_4h_min",
    "spo2_4h_std",
    "spo2_4h_slope",
    "rr_4h_mean",
    "rr_4h_max",
    "rr_4h_std",
    "rr_4h_slope",
]
TARGET = "swallow_flag_within_4h"


def label_future_swallow(physio: pd.DataFrame, horizon_hours: int = 4) -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIR / "ground_truth_labels.csv", usecols=["stay_id", "event_time", "label_type"])
    labels = labels[(labels["label_type"] == "chartevents_swallow_flag") & labels["stay_id"].notna()].copy()
    labels["stay_id"] = labels["stay_id"].astype("int64")
    labels["event_time"] = pd.to_datetime(labels["event_time"])

    physio = physio.copy()
    physio[TARGET] = 0
    horizon = pd.Timedelta(hours=horizon_hours)
    event_map = {stay_id: group["event_time"].sort_values().to_numpy() for stay_id, group in labels.groupby("stay_id")}
    for stay_id, idx in physio.groupby("stay_id").groups.items():
        events = event_map.get(int(stay_id))
        if events is None:
            continue
        times = physio.loc[idx, "window_end_time"].to_numpy()
        pos = np.searchsorted(events, times, side="left")
        valid = pos < len(events)
        hit = np.zeros(len(times), dtype=bool)
        event_times = np.empty(len(times), dtype="datetime64[ns]")
        event_times[:] = np.datetime64("NaT")
        event_times[valid] = events[pos[valid]]
        hit[valid] = event_times[valid] <= (times[valid] + horizon.to_timedelta64())
        physio.loc[idx, TARGET] = hit.astype(int)
    return physio


def load_arrays(max_rows: int, seed: int):
    df = pd.read_parquet(DATA_DIR / "physiological_features.parquet")
    df["window_end_time"] = pd.to_datetime(df["window_end_time"])
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df = label_future_swallow(df)
    x = df[PHYSIO_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.float32).reshape(-1, 1)
    hospital = (df["subject_id"].astype("int64") % 3).to_numpy(dtype=np.int64)
    return train_test_split(x, y, hospital, test_size=0.2, random_state=seed, stratify=y)


def standardize(x_train: np.ndarray, x_val: np.ndarray):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (x_train - mean) / std, (x_val - mean) / std


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    physio = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    acoustic = torch.zeros((len(x), 128), dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(physio, acoustic, labels), batch_size=batch_size, shuffle=shuffle)


def train_local(model: ICUFusionModel, data_loader: DataLoader, epochs: int, lr: float) -> ICUFusionModel:
    model = deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        for physio, acoustic, y in data_loader:
            opt.zero_grad()
            loss = loss_fn(model(physio, acoustic), y)
            loss.backward()
            opt.step()
    return model


def average_state_dicts(states: list[dict[str, torch.Tensor]], weights: list[int]) -> dict[str, torch.Tensor]:
    total = float(sum(weights))
    avg = {}
    for key in states[0]:
        avg[key] = sum(state[key] * (weight / total) for state, weight in zip(states, weights))
    return avg


def evaluate(model: ICUFusionModel, data_loader: DataLoader) -> dict[str, float]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for physio, acoustic, y in data_loader:
            pred = model(physio, acoustic)
            preds.extend(pred.squeeze(1).numpy().tolist())
            labels.extend(y.squeeze(1).numpy().tolist())
    y_true = np.array(labels)
    y_pred = np.array(preds)
    return {
        "auroc": float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y_true, y_pred)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train, x_val, y_train, y_val, h_train, _ = load_arrays(args.max_rows, args.seed)
    x_train, x_val = standardize(x_train, x_val)
    val_loader = loader(x_val, y_val, args.batch_size, False)

    centralized = ICUFusionModel(physio_dim=len(PHYSIO_FEATURES), acoustic_dim=128)
    centralized = train_local(centralized, loader(x_train, y_train, args.batch_size, True), args.central_epochs, args.lr)

    federated = ICUFusionModel(physio_dim=len(PHYSIO_FEATURES), acoustic_dim=128)
    hospital_sizes = {}
    for rnd in range(args.rounds):
        states, weights = [], []
        for hospital_id in range(args.hospitals):
            mask = h_train == hospital_id
            if not mask.any():
                continue
            local_loader = loader(x_train[mask], y_train[mask], args.batch_size, True)
            local_model = train_local(federated, local_loader, args.local_epochs, args.lr)
            states.append(local_model.state_dict())
            weights.append(int(mask.sum()))
            hospital_sizes[str(hospital_id)] = int(mask.sum())
        federated.load_state_dict(average_state_dicts(states, weights))
        print(f"round={rnd + 1} clients={len(states)} examples={sum(weights):,}")

    metrics = {
        "flower_version": flwr.__version__,
        "target": TARGET,
        "hospitals": args.hospitals,
        "rounds": args.rounds,
        "hospital_train_sizes": hospital_sizes,
        "centralized": evaluate(centralized, val_loader),
        "federated_fedavg": evaluate(federated, val_loader),
    }
    (MODEL_DIR / "phase6_federated_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(federated.state_dict(), MODEL_DIR / "phase6_fedavg_swallow_model.pt")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--hospitals", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--central-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    print(f"PHASE 6 COMPLETE metrics={MODEL_DIR / 'phase6_federated_metrics.json'}")


if __name__ == "__main__":
    main()
