import logging
import duckdb
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Physiologically plausible ranges
RANGES = {
    220277: (50, 100),   # SpO2
    220045: (20, 250),   # Heart Rate
    220210: (4, 60),     # Respiratory Rate
    224695: (0, 60),     # Peak Inspiratory Pressure
}

def range_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Remove physiologically implausible values."""
    query = """
    SELECT * FROM df
    WHERE 
        (itemid = 220277 AND valuenum >= 50 AND valuenum <= 100) OR
        (itemid = 220045 AND valuenum >= 20 AND valuenum <= 250) OR
        (itemid = 220210 AND valuenum >= 4 AND valuenum <= 60) OR
        (itemid = 224695 AND valuenum >= 0 AND valuenum <= 60) OR
        (itemid NOT IN (220277, 220045, 220210, 224695))
    """
    filtered_df = duckdb.sql(query).df()
    rejected = len(df) - len(filtered_df)
    logger.info(f"range_filter: rejected {rejected} implausible rows.")
    return filtered_df

def flatline_detection(df: pd.DataFrame, window_hours: int = 4, std_threshold: float = 0.01) -> pd.DataFrame:
    """Flag windows where signal variance is near zero."""
    query = f"""
    SELECT *,
           CASE 
             WHEN COALESCE(STDDEV(valuenum) OVER w, 1.0) < {std_threshold} THEN 1
             ELSE 0
           END as is_flatline
    FROM df
    WINDOW w AS (
        PARTITION BY stay_id, itemid, 
        date_bin(INTERVAL '{window_hours} hours', CAST(charttime AS TIMESTAMP), TIMESTAMP '2000-01-01')
    )
    """
    res = duckdb.sql(query).df()
    flat_count = res['is_flatline'].sum()
    logger.info(f"flatline_detection: flagged {flat_count} flatline observations.")
    return res

def missing_rate(df: pd.DataFrame, window_hours: int = 4) -> pd.DataFrame:
    """Calculate fraction of missing observations per window (assume 1/hr expected)."""
    query = f"""
    SELECT *,
           GREATEST(0.0, {window_hours}.0 - COUNT(valuenum) OVER w) / {window_hours}.0 AS missing_rate
    FROM df
    WINDOW w AS (
        PARTITION BY stay_id, itemid, 
        date_bin(INTERVAL '{window_hours} hours', CAST(charttime AS TIMESTAMP), TIMESTAMP '2000-01-01')
    )
    """
    return duckdb.sql(query).df()

def compute_sqi_score(df: pd.DataFrame, window_hours: int = 4, std_threshold: float = 0.01) -> pd.DataFrame:
    """Composite SQI score (0-1) combining flatline and missing checks."""
    query = f"""
    WITH metrics AS (
        SELECT *,
               COALESCE(STDDEV(valuenum) OVER w, 1.0) as window_std,
               COUNT(valuenum) OVER w as obs_count
        FROM df
        WINDOW w AS (
            PARTITION BY stay_id, itemid, 
            date_bin(INTERVAL '{window_hours} hours', CAST(charttime AS TIMESTAMP), TIMESTAMP '2000-01-01')
        )
    )
    SELECT *,
           CASE 
             WHEN window_std < {std_threshold} THEN 0.0
             ELSE 1.0 - (GREATEST(0.0, {window_hours}.0 - obs_count) / {window_hours}.0)
           END AS sqi_score
    FROM metrics
    """
    res = duckdb.sql(query).df()
    logger.info("compute_sqi_score: appended SQI scores.")
    return res

def apply_sqi_filter(df: pd.DataFrame, min_sqi: float = 0.5) -> pd.DataFrame:
    """Remove windows below threshold."""
    if 'sqi_score' not in df.columns:
        df = compute_sqi_score(df)
        
    query = f"SELECT * FROM df WHERE sqi_score >= {min_sqi}"
    filtered_df = duckdb.sql(query).df()
    rejected = len(df) - len(filtered_df)
    logger.info(f"apply_sqi_filter: removed {rejected} rows (SQI < {min_sqi}).")
    return filtered_df

def main():
    data_dir = Path(r"c:\Users\rohit\MultiModal\AEGIS\data")
    input_file = data_dir / "chartevents_filtered.csv"
    output_file = data_dir / "chartevents_sqi_filtered.csv"
    
    if not input_file.exists():
        logger.error(f"Missing input: {input_file}")
        return

    logger.info("Load raw chartevents.")
    df = pd.read_csv(input_file)
    
    logger.info("Apply SQI checks.")
    df = range_filter(df)
    df = compute_sqi_score(df)
    df = apply_sqi_filter(df, min_sqi=0.5)
    
    # Drop intermediate columns if desired (sqi_score kept for records)
    df.to_csv(output_file, index=False)
    logger.info(f"Saved filtered data: {output_file}")

if __name__ == "__main__":
    main()
