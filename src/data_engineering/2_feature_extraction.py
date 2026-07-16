"""
Phase 3: Physiological Feature Pipeline.

Builds 4-hour rolling ICU time-series features from filtered chartevents:
- SpO2: itemid 220277
- Respiratory rate: itemid 220210
- Peak inspiratory pressure: itemid 224695
- Heart rate: itemid 220045

Output:
- data/processed/physiological_features.parquet
"""
from __future__ import annotations

from pathlib import Path

import duckdb


DATA_DIR = Path(r"C:\Users\rohit\MultiModal\dataset")
OUTPUT_DIR = Path(r"C:\Users\rohit\MultiModal\icu_predictive_system\data\processed")
CHARTEVENTS_PATH = DATA_DIR / "chartevents_filtered.csv"
COHORT_PATH = OUTPUT_DIR / "target_cohort.csv"
OUTPUT_PATH = OUTPUT_DIR / "physiological_features.parquet"


ITEMS = {
    220277: "spo2",
    220210: "resp_rate",
    224695: "peak_insp_pressure",
    220045: "heart_rate",
}


def p(path: Path) -> str:
    return str(path).replace("\\", "/")


def build_physiological_features() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(database=":memory:")
    try:
        print("Reading cohort and filtered chartevents with DuckDB")
        item_list = ",".join(str(item) for item in ITEMS)
        query = f"""
        COPY (
            WITH cohort AS (
                SELECT DISTINCT subject_id, hadm_id, stay_id
                FROM read_csv_auto('{p(COHORT_PATH)}')
                WHERE stay_id IS NOT NULL
            ),
            raw AS (
                SELECT
                    c.subject_id,
                    c.hadm_id,
                    c.stay_id,
                    date_trunc('hour', CAST(c.charttime AS TIMESTAMP)) AS hour_time,
                    CASE c.itemid
                        WHEN 220277 THEN 'spo2'
                        WHEN 220210 THEN 'resp_rate'
                        WHEN 224695 THEN 'peak_insp_pressure'
                        WHEN 220045 THEN 'heart_rate'
                    END AS signal,
                    c.valuenum
                FROM read_csv_auto('{p(CHARTEVENTS_PATH)}') c
                INNER JOIN cohort t
                  ON c.subject_id = t.subject_id
                 AND c.hadm_id = t.hadm_id
                 AND c.stay_id = t.stay_id
                WHERE c.itemid IN ({item_list})
                  AND c.valuenum IS NOT NULL
            ),
            hourly AS (
                SELECT
                    subject_id,
                    hadm_id,
                    stay_id,
                    hour_time,
                    AVG(CASE WHEN signal = 'spo2' THEN valuenum END) AS spo2_mean,
                    AVG(CASE WHEN signal = 'resp_rate' THEN valuenum END) AS rr_mean,
                    AVG(CASE WHEN signal = 'peak_insp_pressure' THEN valuenum END) AS pip_mean,
                    AVG(CASE WHEN signal = 'heart_rate' THEN valuenum END) AS hr_mean,
                    COUNT(CASE WHEN signal = 'spo2' THEN 1 END) AS spo2_count,
                    COUNT(CASE WHEN signal = 'resp_rate' THEN 1 END) AS rr_count,
                    COUNT(CASE WHEN signal = 'peak_insp_pressure' THEN 1 END) AS pip_count,
                    COUNT(CASE WHEN signal = 'heart_rate' THEN 1 END) AS hr_count
                FROM raw
                GROUP BY subject_id, hadm_id, stay_id, hour_time
            ),
            features AS (
                SELECT
                    *,
                    AVG(spo2_mean) OVER w AS spo2_4h_mean,
                    MIN(spo2_mean) OVER w AS spo2_4h_min,
                    STDDEV_SAMP(spo2_mean) OVER w AS spo2_4h_std,
                    regr_slope(spo2_mean, epoch(hour_time) / 3600.0) OVER w AS spo2_4h_slope,
                    AVG(rr_mean) OVER w AS rr_4h_mean,
                    MAX(rr_mean) OVER w AS rr_4h_max,
                    STDDEV_SAMP(rr_mean) OVER w AS rr_4h_std,
                    regr_slope(rr_mean, epoch(hour_time) / 3600.0) OVER w AS rr_4h_slope,
                    AVG(pip_mean) OVER w AS pip_4h_mean,
                    MAX(pip_mean) OVER w AS pip_4h_max,
                    STDDEV_SAMP(pip_mean) OVER w AS pip_4h_std,
                    AVG(hr_mean) OVER w AS hr_4h_mean,
                    STDDEV_SAMP(hr_mean) OVER w AS hr_4h_std,
                    SUM(spo2_count + rr_count + pip_count + hr_count) OVER w AS observations_4h
                FROM hourly
                WINDOW w AS (
                    PARTITION BY subject_id, hadm_id, stay_id
                    ORDER BY hour_time
                    RANGE BETWEEN INTERVAL 3 HOURS PRECEDING AND CURRENT ROW
                )
            )
            SELECT
                subject_id,
                hadm_id,
                stay_id,
                hour_time AS window_end_time,
                spo2_mean,
                rr_mean,
                pip_mean,
                hr_mean,
                spo2_4h_mean,
                spo2_4h_min,
                COALESCE(spo2_4h_std, 0) AS spo2_4h_std,
                COALESCE(spo2_4h_slope, 0) AS spo2_4h_slope,
                rr_4h_mean,
                rr_4h_max,
                COALESCE(rr_4h_std, 0) AS rr_4h_std,
                COALESCE(rr_4h_slope, 0) AS rr_4h_slope,
                pip_4h_mean,
                pip_4h_max,
                COALESCE(pip_4h_std, 0) AS pip_4h_std,
                hr_4h_mean,
                COALESCE(hr_4h_std, 0) AS hr_4h_std,
                observations_4h,
                CASE WHEN spo2_4h_min < 90 THEN 1 ELSE 0 END AS spo2_drop_flag,
                CASE WHEN rr_4h_max > 30 THEN 1 ELSE 0 END AS tachypnea_flag
            FROM features
            ORDER BY subject_id, hadm_id, stay_id, window_end_time
        ) TO '{p(OUTPUT_PATH)}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        con.execute(query)
        rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p(OUTPUT_PATH)}')").fetchone()[0]
        stays = con.execute(f"SELECT COUNT(DISTINCT stay_id) FROM read_parquet('{p(OUTPUT_PATH)}')").fetchone()[0]
        print(f"Saved {OUTPUT_PATH} rows={rows:,} stays={stays:,}")
    finally:
        con.close()


if __name__ == "__main__":
    build_physiological_features()
