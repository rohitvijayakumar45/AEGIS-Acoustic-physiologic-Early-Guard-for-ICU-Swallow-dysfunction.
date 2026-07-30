"""
Extract 4-hour rolling physiological features from chartevents.
Integrates Signal Quality Index (SQI).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    from signal_quality import range_filter, compute_sqi_score, apply_sqi_filter
except ImportError:
    logging.warning("SQI module not found. Using passthrough stubs.")
    def range_filter(df): return df
    def compute_sqi_score(df):
        df['sqi_score'] = 1.0
        return df
    def apply_sqi_filter(df, min_sqi):
        return df.copy()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

CHARTEVENTS_PATH = Path("C:/Users/rohit/MultiModal/dataset/chartevents_filtered.csv")
COHORT_PATH = Path("C:/Users/rohit/MultiModal/AEGIS/data/processed/target_cohort.csv")
OUTPUT_PATH = Path("C:/Users/rohit/MultiModal/AEGIS/data/processed/physiological_features.parquet")

def main():
    logging.info("Start feature extraction.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()

    ITEMIDS = (220277, 220210, 224695, 220045, 223761, 223762, 228112, 198)

    logging.info("Load cohort, filter chartevents, map signals.")
    query_load = f"""
        SELECT
            c.subject_id,
            c.hadm_id,
            c.stay_id,
            CAST(ce.charttime AS TIMESTAMP) as charttime,
            ce.itemid,
            ce.valuenum,
            CASE
                WHEN ce.itemid = 220277 THEN 'spo2'
                WHEN ce.itemid = 220210 THEN 'rr'
                WHEN ce.itemid = 224695 THEN 'pip'
                WHEN ce.itemid = 220045 THEN 'hr'
                WHEN ce.itemid IN (223761, 223762) THEN 'temp'
                WHEN ce.itemid IN (228112, 198) THEN 'gcs'
            END as signal
        FROM read_csv_auto('{COHORT_PATH}') c
        JOIN read_csv_auto('{CHARTEVENTS_PATH}') ce
          ON c.subject_id = ce.subject_id
         AND c.hadm_id = ce.hadm_id
         AND c.stay_id = ce.stay_id
        WHERE ce.itemid IN {ITEMIDS}
          AND ce.valuenum IS NOT NULL
    """
    df = conn.execute(query_load).df()
    logging.info(f"Loaded {len(df)} rows.")

    logging.info("Apply range filter.")
    df = range_filter(df)

    logging.info("Aggregate features per 4-hour window.")
    conn.register('filtered_data', df)

    query_agg = """
        WITH time_calc AS (
            SELECT
                subject_id, hadm_id, stay_id, signal, valuenum, charttime,
                EXTRACT(EPOCH FROM (charttime - MIN(charttime) OVER (PARTITION BY stay_id))) / 3600.0 as time_hrs
            FROM filtered_data
        ),
        windowed AS (
            SELECT
                subject_id, hadm_id, stay_id, signal, valuenum, time_hrs,
                CAST(FLOOR(time_hrs / 4.0) AS INT) as window
            FROM time_calc
        )
        SELECT
            subject_id, hadm_id, stay_id, window, signal,
            AVG(valuenum) as mean,
            MIN(valuenum) as min_val,
            MAX(valuenum) as max_val,
            COALESCE(STDDEV(valuenum), 0) as std,
            COUNT(valuenum) as obs_count,
            COALESCE(REGR_SLOPE(valuenum, time_hrs), 0) as slope
        FROM windowed
        GROUP BY subject_id, hadm_id, stay_id, window, signal
    """
    aggs = conn.execute(query_agg).df()

    logging.info("Pivot to wide format.")
    index_cols = ['subject_id', 'hadm_id', 'stay_id', 'window']
    wide = aggs.pivot_table(index=index_cols, columns='signal',
                            values=['mean', 'min_val', 'max_val', 'std', 'obs_count', 'slope'],
                            aggfunc='first')
    wide.columns = [f"{col[1]}_{col[0]}" for col in wide.columns]
    wide = wide.reset_index()
    
    logging.info("Compute SQI score.")
    # SQI module expects 'itemid' column — skip if pivoted already
    # Instead compute a simple completeness-based SQI on the wide table
    signal_count_cols = [c for c in wide.columns if c.endswith('_obs_count')]
    if signal_count_cols:
        wide['sqi_score'] = wide[signal_count_cols].fillna(0).mean(axis=1) / 4.0  # normalize by expected 4 obs/window
        wide['sqi_score'] = wide['sqi_score'].clip(0, 1)
        total_windows = len(wide)
        wide = wide[wide['sqi_score'] >= 0.25].copy()
        logging.info(f"SQI filter rejected {total_windows - len(wide)} / {total_windows} windows.")

    logging.info("Compute derived flags.")
    wide['spo2_drop_flag'] = (wide.get('spo2_min_val', pd.Series(dtype=float)) < 90).astype(int) if 'spo2_min_val' in wide.columns else 0
    wide['tachypnea_flag'] = (wide.get('rr_max_val', pd.Series(dtype=float)) > 30).astype(int) if 'rr_max_val' in wide.columns else 0
    wide['bradycardia_flag'] = (wide.get('hr_min_val', pd.Series(dtype=float)) < 50).astype(int) if 'hr_min_val' in wide.columns else 0
    wide['fever_flag'] = (wide.get('temp_max_val', pd.Series(dtype=float)) > 38.3).astype(int) if 'temp_max_val' in wide.columns else 0

    wide = wide.fillna(0)

    logging.info("Save to parquet.")
    wide.to_parquet(OUTPUT_PATH, index=False)
    logging.info(f"Saved {len(wide)} windows for {wide['stay_id'].nunique()} stays.")
    logging.info("Done.")

if __name__ == "__main__":
    main()

