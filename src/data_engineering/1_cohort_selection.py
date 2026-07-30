from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_valid_path(paths: list[Path]) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of the paths exist. Checked: {paths}")

def main():
    base_dir = Path("C:/Users/rohit/MultiModal/dataset")
    
    diagnoses_path = get_valid_path([
        base_dir / "mimic-iv-3.1-20260514T144455Z-3-005/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz",
        base_dir / "mimic-iv-3.1/hosp/diagnoses_icd.csv.gz"
    ])
    
    icustays_path = get_valid_path([
        base_dir / "mimic-iv-3.1-20260514T144455Z-3-004/mimic-iv-3.1/icu/icustays.csv.gz",
        base_dir / "mimic-iv-3.1/icu/icustays.csv.gz"
    ])
    
    admissions_path = get_valid_path([
        base_dir / "mimic-iv-3.1-20260514T144455Z-3-004/mimic-iv-3.1/hosp/admissions.csv.gz",
        base_dir / "mimic-iv-3.1-20260514T144455Z-3-005/mimic-iv-3.1/hosp/admissions.csv.gz",
        base_dir / "mimic-iv-3.1/hosp/admissions.csv.gz"
    ])
    
    patients_path = get_valid_path([
        base_dir / "mimic-iv-3.1-20260514T144455Z-3-004/mimic-iv-3.1/hosp/patients.csv.gz",
        base_dir / "mimic-iv-3.1-20260514T144455Z-3-005/mimic-iv-3.1/hosp/patients.csv.gz",
        base_dir / "mimic-iv-3.1/hosp/patients.csv.gz"
    ])
    
    out_dir = Path("C:/Users/rohit/MultiModal/AEGIS/data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "target_cohort.csv"

    logging.info("Building DuckDB query to select cohort...")
    
    conn = duckdb.connect()
    
    query = f"""
    WITH cohort_dx AS (
        SELECT subject_id, hadm_id, icd_code, icd_version
        FROM read_csv_auto('{diagnoses_path.as_posix()}')
        WHERE (icd_version = 10 AND (
            icd_code LIKE 'R13%' OR
            icd_code LIKE 'J690%' OR
            icd_code LIKE 'J95%' OR
            icd_code LIKE 'I63%' OR
            icd_code LIKE 'I46%' OR
            icd_code LIKE 'G12%' OR
            icd_code = 'G35' OR
            icd_code = 'G20' OR
            icd_code LIKE 'G70%' OR
            icd_code LIKE 'R47%'
        )) OR (icd_version = 9 AND (
            icd_code LIKE '7872%' OR
            icd_code LIKE '5070%' OR
            icd_code LIKE '5190%' OR
            icd_code LIKE '434%' OR
            icd_code = '4275' OR
            icd_code LIKE '3352%' OR
            icd_code = '340' OR
            icd_code = '3320' OR
            icd_code LIKE '3580%' OR
            icd_code LIKE '7845%'
        ))
    ),
    patients_admissions AS (
        SELECT 
            a.subject_id, a.hadm_id, a.admittime, a.admission_type, a.race as ethnicity, a.insurance,
            p.gender, p.anchor_age
        FROM read_csv_auto('{admissions_path.as_posix()}') a
        JOIN read_csv_auto('{patients_path.as_posix()}') p ON a.subject_id = p.subject_id
    ),
    icu AS (
        SELECT subject_id, hadm_id, stay_id, intime, outtime, los, first_careunit
        FROM read_csv_auto('{icustays_path.as_posix()}')
    )
    SELECT
        i.subject_id, i.hadm_id, i.stay_id,
        pa.admittime, pa.admission_type, pa.ethnicity, pa.insurance, pa.gender, pa.anchor_age,
        i.intime, i.outtime, i.los, i.first_careunit,
        string_agg(dx.icd_code, ',') as icd_codes
    FROM icu i
    JOIN patients_admissions pa ON i.subject_id = pa.subject_id AND i.hadm_id = pa.hadm_id
    JOIN cohort_dx dx ON i.subject_id = dx.subject_id AND i.hadm_id = dx.hadm_id
    GROUP BY 
        i.subject_id, i.hadm_id, i.stay_id, 
        pa.admittime, pa.admission_type, pa.ethnicity, pa.insurance, pa.gender, pa.anchor_age, 
        i.intime, i.outtime, i.los, i.first_careunit
    """
    
    logging.info("Executing query. This may take a moment...")
    df = conn.execute(query).df()
    
    logging.info(f"Saving enriched cohort to {out_file}...")
    df.to_csv(out_file, index=False)
    
    logging.info("--- Cohort Statistics ---")
    logging.info(f"Total Unique Patients: {df['subject_id'].nunique()}")
    logging.info(f"Total ICU Stays: {df['stay_id'].nunique()}")
    
    logging.info("--- Demographic Breakdown ---")
    logging.info("\\nGender:\\n" + df['gender'].value_counts().to_string())
    logging.info("\\nAdmission Type:\\n" + df['admission_type'].value_counts().to_string())
    logging.info("\\nEthnicity (Top 5):\\n" + df['ethnicity'].value_counts().head(5).to_string())
    logging.info("\\nInsurance:\\n" + df['insurance'].value_counts().to_string())
    logging.info(f"Average Age: {df['anchor_age'].mean():.2f}")
    logging.info(f"Average LOS (days): {df['los'].mean():.2f}")
    
    conn.close()

if __name__ == "__main__":
    main()
