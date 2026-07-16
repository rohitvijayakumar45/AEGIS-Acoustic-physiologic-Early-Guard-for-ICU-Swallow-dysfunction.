"""
Phase 4: Acoustic Feature Extraction.

Extracts compact acoustic descriptors from:
- COUGHVID cough WAV files + JSON metadata
- TORGO dysarthric/control speech WAV files

Outputs:
- data/processed/cough_features.parquet
- data/processed/speech_features.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_DIR = Path(r"C:\Users\rohit\MultiModal\dataset")
OUTPUT_DIR = Path(r"C:\Users\rohit\MultiModal\icu_predictive_system\data\processed")
COUGHVID_DIR = DATA_DIR / "COVIDVID"
TORGO_DIRS = [DATA_DIR / name for name in ["F_Dys", "M_Dys", "F_Con", "M_Con"]]

SAMPLE_RATE = 16000
MAX_DURATION = 12.0
N_MFCC = 20
EMBEDDING_DIM = 128


def _to_float(value: Any) -> float | None:
    try:
        if value in [None, ""]:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def acoustic_embedding(wav_path: Path) -> dict[str, float]:
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, duration=MAX_DURATION)
    if y.size == 0:
        raise ValueError("empty audio")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    values: list[float] = []
    for arr in [mfcc, delta, contrast]:
        values.extend(np.nanmean(arr, axis=1).tolist())
        values.extend(np.nanstd(arr, axis=1).tolist())
    for arr in [centroid, bandwidth, rolloff, zcr, rms]:
        values.append(float(np.nanmean(arr)))
        values.append(float(np.nanstd(arr)))

    duration = librosa.get_duration(y=y, sr=sr)
    values.extend([float(duration), float(np.mean(np.abs(y))), float(np.max(np.abs(y)))])

    if len(values) < EMBEDDING_DIM:
        values.extend([0.0] * (EMBEDDING_DIM - len(values)))
    values = values[:EMBEDDING_DIM]
    return {f"feature_{idx:03d}": float(value) for idx, value in enumerate(values)}


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def extract_coughvid(max_files: int | None = None) -> pd.DataFrame:
    wavs = sorted(COUGHVID_DIR.glob("*.wav"))
    if max_files:
        wavs = wavs[:max_files]

    rows = []
    for wav_path in tqdm(wavs, desc="COUGHVID"):
        metadata = read_json(wav_path.with_suffix(".json"))
        try:
            row = {
                "recording_id": wav_path.stem,
                "audio_path": str(wav_path),
                "dataset": "COUGHVID",
                "label_type": "cough_status",
                "label_value": metadata.get("status", "unknown"),
                "cough_detected": _to_float(metadata.get("cough_detected")),
                "age": _to_float(metadata.get("age")),
                "gender": metadata.get("gender"),
                "respiratory_condition": metadata.get("respiratory_condition"),
                "fever_muscle_pain": metadata.get("fever_muscle_pain"),
            }
            row.update(acoustic_embedding(wav_path))
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "recording_id": wav_path.stem,
                    "audio_path": str(wav_path),
                    "dataset": "COUGHVID",
                    "label_type": "error",
                    "label_value": str(exc)[:200],
                }
            )

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "cough_features.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {out} rows={len(df):,}")
    return df


def torgo_label(path: Path) -> dict[str, str]:
    parts = {part.lower() for part in path.parts}
    is_dys = "f_dys" in parts or "m_dys" in parts
    speaker_root = next((part for part in path.parts if part in {"F_Dys", "M_Dys", "F_Con", "M_Con"}), "")
    mic = "headMic" if "headmic" in str(path).lower() else "arrayMic"
    return {
        "dataset": "TORGO",
        "label_type": "speech_dysarthria",
        "label_value": "dysarthric" if is_dys else "control",
        "speaker_group": speaker_root,
        "mic_type": mic,
    }


def extract_torgo(include_arraymic: bool = False, max_files: int | None = None) -> pd.DataFrame:
    wavs: list[Path] = []
    for root in TORGO_DIRS:
        wavs.extend(root.rglob("*.wav"))

    if not include_arraymic:
        wavs = [path for path in wavs if "headmic" in str(path).lower()]
    wavs = sorted(wavs)
    if max_files:
        wavs = wavs[:max_files]

    rows = []
    for wav_path in tqdm(wavs, desc="TORGO"):
        try:
            row = {
                "recording_id": wav_path.stem,
                "audio_path": str(wav_path),
                **torgo_label(wav_path),
            }
            row.update(acoustic_embedding(wav_path))
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "recording_id": wav_path.stem,
                    "audio_path": str(wav_path),
                    "dataset": "TORGO",
                    "label_type": "error",
                    "label_value": str(exc)[:200],
                }
            )

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "speech_features.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {out} rows={len(df):,}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cough", type=int, default=None)
    parser.add_argument("--max-speech", type=int, default=None)
    parser.add_argument("--include-arraymic", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cough_df = extract_coughvid(max_files=args.max_cough)
    speech_df = extract_torgo(include_arraymic=args.include_arraymic, max_files=args.max_speech)
    print(
        "PHASE 4 COMPLETE "
        f"cough_rows={len(cough_df):,} speech_rows={len(speech_df):,} "
        f"speech_arraymic={'yes' if args.include_arraymic else 'no'}"
    )


if __name__ == "__main__":
    main()
