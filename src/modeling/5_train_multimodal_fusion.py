"""
Phase 5: Multimodal Fusion Training.

Conservative baseline:
- Uses existing ICUFusionModel without architecture changes.
- Builds simulated cross-dataset acoustic pairing.
- Trains separate binary models for:
  1. suction need within 4 hours
  2. swallowing difficulty flag within 4 hours
- Reports fused, physio-only, and acoustic-only ablation AUROC/AP.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

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
ACOUSTIC_FEATURES = [f"feature_{idx:03d}" for idx in range(128)]


@dataclass
class TrainConfig:
    target: str
    epochs: int = 5
    batch_size: int = 256
    max_rows: int = 250000
    learning_rate: float = 1e-3
    seed: int = 42


class FusionDataset(Dataset):
    def __init__(self, physio: np.ndarray, acoustic: np.ndarray, labels: np.ndarray):
        self.physio = torch.tensor(physio, dtype=torch.float32).unsqueeze(1)
        self.acoustic = torch.tensor(acoustic, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.physio[idx], self.acoustic[idx], self.labels[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_physio(max_rows: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "physiological_features.parquet")
    df["window_end_time"] = pd.to_datetime(df["window_end_time"])
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    df = df.sort_values(["stay_id", "window_end_time"]).reset_index(drop=True)
    return df


def label_future_events(physio: pd.DataFrame, label_type: str, target_name: str, horizon_hours: int = 4) -> pd.DataFrame:
    labels = pd.read_csv(DATA_DIR / "ground_truth_labels.csv", usecols=["stay_id", "event_time", "label_type"])
    labels = labels[(labels["label_type"] == label_type) & labels["stay_id"].notna()].copy()
    labels["stay_id"] = labels["stay_id"].astype("int64")
    labels["event_time"] = pd.to_datetime(labels["event_time"])

    physio = physio.copy()
    physio[target_name] = 0
    horizon = pd.Timedelta(hours=horizon_hours)

    event_map = {stay_id: group["event_time"].sort_values().to_numpy() for stay_id, group in labels.groupby("stay_id")}
    for stay_id, idx in physio.groupby("stay_id").groups.items():
        events = event_map.get(int(stay_id))
        if events is None or len(events) == 0:
            continue
        times = physio.loc[idx, "window_end_time"].to_numpy()
        pos = np.searchsorted(events, times, side="left")
        hit = np.zeros(len(times), dtype=bool)
        valid = pos < len(events)
        event_times = np.empty(len(times), dtype="datetime64[ns]")
        event_times[:] = np.datetime64("NaT")
        event_times[valid] = events[pos[valid]]
        hit[valid] = event_times[valid] <= (times[valid] + horizon.to_timedelta64())
        physio.loc[idx, target_name] = hit.astype(int)

    return physio


def load_acoustic(seed: int) -> np.ndarray:
    cough = pd.read_parquet(DATA_DIR / "cough_features.parquet", columns=ACOUSTIC_FEATURES + ["label_type"])
    speech = pd.read_parquet(DATA_DIR / "speech_features.parquet", columns=ACOUSTIC_FEATURES + ["label_type"])
    acoustic = pd.concat(
        [
            cough[cough["label_type"] != "error"][ACOUSTIC_FEATURES],
            speech[speech["label_type"] != "error"][ACOUSTIC_FEATURES],
        ],
        ignore_index=True,
    )
    acoustic = acoustic.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return acoustic.sample(frac=1.0, random_state=seed).to_numpy(dtype=np.float32)


def make_arrays(df: pd.DataFrame, target: str, acoustic_pool: np.ndarray, seed: int):
    physio = df[PHYSIO_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    labels = df[target].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    acoustic_idx = rng.integers(0, len(acoustic_pool), size=len(df))
    acoustic = acoustic_pool[acoustic_idx]
    return physio, acoustic, labels


def standardize(train: np.ndarray, val: np.ndarray):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train - mean) / std, (val - mean) / std


def evaluate(model: nn.Module, loader: DataLoader, mode: str) -> dict[str, float]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for physio, acoustic, y in loader:
            if mode == "physio_only":
                acoustic = torch.zeros_like(acoustic)
            elif mode == "acoustic_only":
                physio = torch.zeros_like(physio)
            pred = model(physio, acoustic)
            preds.extend(pred.squeeze(1).cpu().numpy().tolist())
            labels.extend(y.squeeze(1).cpu().numpy().tolist())

    y_true = np.array(labels)
    y_pred = np.array(preds)
    out = {"average_precision": float(average_precision_score(y_true, y_pred))}
    out["auroc"] = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan")
    return out


def train_one(config: TrainConfig) -> dict[str, object]:
    set_seed(config.seed)
    target_to_label = {
        "suction_within_4h": "suction_procedure",
        "swallow_flag_within_4h": "chartevents_swallow_flag",
    }
    label_type = target_to_label[config.target]

    physio_df = load_physio(config.max_rows, config.seed)
    physio_df = label_future_events(physio_df, label_type, config.target)
    positives = int(physio_df[config.target].sum())
    print(f"{config.target}: rows={len(physio_df):,} positives={positives:,}")

    acoustic_pool = load_acoustic(config.seed)
    physio, acoustic, labels = make_arrays(physio_df, config.target, acoustic_pool, config.seed)
    p_train, p_val, a_train, a_val, y_train, y_val = train_test_split(
        physio, acoustic, labels, test_size=0.2, random_state=config.seed, stratify=labels if len(np.unique(labels)) > 1 else None
    )
    p_train, p_val = standardize(p_train, p_val)
    a_train, a_val = standardize(a_train, a_val)

    train_ds = FusionDataset(p_train, a_train, y_train)
    val_ds = FusionDataset(p_val, a_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = ICUFusionModel(physio_dim=len(PHYSIO_FEATURES), acoustic_dim=len(ACOUSTIC_FEATURES))
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], dtype=torch.float32)
    loss_fn = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        model.train()
        losses = []
        for physio_batch, acoustic_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(physio_batch, acoustic_batch)
            # Weighted BCE, kept explicit because model already outputs sigmoid probabilities.
            weights = torch.where(y_batch > 0, pos_weight, torch.ones_like(y_batch))
            loss = (loss_fn(pred, y_batch) * weights).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        print(f"  epoch={epoch + 1} loss={np.mean(losses):.4f}")

    metrics = {
        "target": config.target,
        "rows": int(len(physio_df)),
        "positives": positives,
        "positive_rate": float(positives / max(len(physio_df), 1)),
        "fused": evaluate(model, val_loader, "fused"),
        "physio_only": evaluate(model, val_loader, "physio_only"),
        "acoustic_only": evaluate(model, val_loader, "acoustic_only"),
    }
    model_path = MODEL_DIR / f"{config.target}_fusion_model.pt"
    torch.save(model.state_dict(), model_path)
    metrics["model_path"] = str(model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=250000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    all_metrics = []
    for target in ["suction_within_4h", "swallow_flag_within_4h"]:
        metrics = train_one(
            TrainConfig(target=target, epochs=args.epochs, batch_size=args.batch_size, max_rows=args.max_rows)
        )
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))

    metrics_path = MODEL_DIR / "phase5_ablation_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(f"PHASE 5 COMPLETE metrics={metrics_path}")


if __name__ == "__main__":
    main()
