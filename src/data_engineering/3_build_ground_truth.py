"""
Phase 2: Ground Truth Label Construction.

Outputs:
- target_cohort.csv
- swallow_flags.csv
- note_markers.csv
- suction_events.csv
- ground_truth_labels.csv
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import duckdb
import pandas as pd


DATA_DIR = Path(r"C:\Users\rohit\MultiModal\dataset")
OUTPUT_DIR = Path(r"C:\Users\rohit\MultiModal\icu_predictive_system\data\processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHARTEVENTS_PATH = DATA_DIR / "chartevents_filtered.csv"
NOTES_PATH = (
    DATA_DIR
    / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2-20260514T144228Z-3-001"
    / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2"
    / "note"
    / "discharge.csv.gz"
)
PROC_PATH = DATA_DIR / "mimic-iv-3.1-20260514T144455Z-3-004" / "mimic-iv-3.1" / "icu" / "procedureevents.csv.gz"
DIAG_PATH = DATA_DIR / "mimic-iv-3.1-20260514T144455Z-3-005" / "mimic-iv-3.1" / "hosp" / "diagnoses_icd.csv.gz"
ICU_PATH = DATA_DIR / "mimic-iv-3.1-20260514T144455Z-3-004" / "mimic-iv-3.1" / "icu" / "icustays.csv.gz"

KEYWORDS = [
    "aspiration pneumonia",
    "oropharyngeal dysphagia",
    "difficulty swallowing",
    "failed water trial",
    "swallow assessment",
    "failed swallow",
    "swallowing dysfunction",
    "speech pathology",
    "speech therapy",
    "swallow eval",
    "swallow screen",
    "wet voice",
    "dysphagia",
    "aspiration",
    "tracheostomy",
    "decannulation",
]
NOTE_PATTERN = re.compile("|".join(re.escape(k) for k in KEYWORDS) + r"|NPO.{0,40}swallow", re.IGNORECASE)


def p(path: Path) -> str:
    return str(path).replace("\\", "/")


def extract_swallow_flags(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("STEP 1: ItemID 225118 swallowing flags")
    query = f"""
    SELECT subject_id, hadm_id, stay_id, charttime AS event_time,
           'chartevents_swallow_flag' AS label_type,
           COALESCE(CAST(valuenum AS VARCHAR), '1') AS label_value,
           itemid
    FROM read_csv_auto('{p(CHARTEVENTS_PATH)}')
    WHERE itemid = 225118
    """
    df = con.execute(query).fetchdf()
    df.to_csv(OUTPUT_DIR / "swallow_flags.csv", index=False)
    print(f"  rows={len(df):,} patients={df['subject_id'].nunique():,}")
    return df


def extract_note_markers(chunksize: int = 25000) -> pd.DataFrame:
    print("STEP 2: streaming discharge note markers")
    marker_path = OUTPUT_DIR / "note_markers.csv"
    rows = []

    usecols = ["note_id", "subject_id", "hadm_id", "charttime", "text"]
    for chunk_idx, chunk in enumerate(pd.read_csv(NOTES_PATH, compression="gzip", usecols=usecols, chunksize=chunksize)):
        for row in chunk.itertuples(index=False):
            text = "" if pd.isna(row.text) else str(row.text)
            matches = sorted({m.group(0).lower() for m in NOTE_PATTERN.finditer(text)})
            for match in matches:
                rows.append(
                    {
                        "note_id": row.note_id,
                        "subject_id": row.subject_id,
                        "hadm_id": row.hadm_id,
                        "stay_id": pd.NA,
                        "event_time": row.charttime,
                        "label_type": "note_marker",
                        "label_value": match,
                    }
                )
        print(f"  chunks={chunk_idx + 1} markers={len(rows):,}")

    df = pd.DataFrame(rows)
    df.to_csv(marker_path, index=False)
    print(f"  rows={len(df):,} patients={df['subject_id'].nunique() if not df.empty else 0:,}")
    return df


def extract_suction_events(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("STEP 3: suctioning procedure labels")
    query = f"""
    SELECT subject_id, hadm_id, stay_id, starttime AS event_time,
           'suction_procedure' AS label_type,
           COALESCE(CAST(value AS VARCHAR), CAST(itemid AS VARCHAR)) AS label_value,
           itemid
    FROM read_csv_auto('{p(PROC_PATH)}')
    WHERE LOWER(CAST(value AS VARCHAR)) LIKE '%suction%'
       OR itemid IN (224385, 225794, 225795, 225796, 225797)
    """
    df = con.execute(query).fetchdf()
    df.to_csv(OUTPUT_DIR / "suction_events.csv", index=False)
    print(f"  rows={len(df):,} patients={df['subject_id'].nunique():,}")
    return df


def build_target_cohort(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    print("STEP 4: target ICU cohort")
    query = f"""
    SELECT DISTINCT d.subject_id, d.hadm_id, i.stay_id,
           i.first_careunit, i.intime, i.outtime, i.los,
           d.icd_code
    FROM read_csv_auto('{p(DIAG_PATH)}') d
    INNER JOIN read_csv_auto('{p(ICU_PATH)}') i
      ON d.subject_id = i.subject_id AND d.hadm_id = i.hadm_id
    WHERE d.icd_code LIKE 'I63%'
       OR d.icd_code LIKE 'I46%'
       OR d.icd_code LIKE 'R13%'
       OR d.icd_code LIKE 'J69%'
       OR d.icd_code LIKE 'J95%'
    """
    df = con.execute(query).fetchdf()
    df.to_csv(OUTPUT_DIR / "target_cohort.csv", index=False)
    print(f"  rows={len(df):,} patients={df['subject_id'].nunique():,}")
    return df


def write_unified_labels(swallow_df: pd.DataFrame, notes_df: pd.DataFrame, suction_df: pd.DataFrame) -> pd.DataFrame:
    print("STEP 5: unified ground_truth_labels.csv")
    columns = ["subject_id", "hadm_id", "stay_id", "event_time", "label_type", "label_value"]
    frames = []
    for df in [swallow_df, notes_df, suction_df]:
        if not df.empty:
            frames.append(df.reindex(columns=columns))

    labels = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    labels = labels.drop_duplicates().sort_values(["subject_id", "hadm_id", "event_time", "label_type"], na_position="last")
    labels.to_csv(OUTPUT_DIR / "ground_truth_labels.csv", index=False)
    print(f"  rows={len(labels):,} patients={labels['subject_id'].nunique() if not labels.empty else 0:,}")
    return labels


def main() -> None:
    con = duckdb.connect(database=":memory:")
    try:
        swallow_df = extract_swallow_flags(con)
        notes_df = extract_note_markers()
        suction_df = extract_suction_events(con)
        build_target_cohort(con)
        write_unified_labels(swallow_df, notes_df, suction_df)
        print("PHASE 2 COMPLETE")
    finally:
        con.close()


if __name__ == "__main__":
    main()
