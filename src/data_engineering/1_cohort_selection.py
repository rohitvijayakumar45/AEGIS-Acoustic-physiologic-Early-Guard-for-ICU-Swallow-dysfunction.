import duckdb
import pandas as pd
import os

def extract_cohort(data_dir: str, output_dir: str):
    """
    Extract a cohort of ICU patients who might be at risk for speech/swallowing dysfunction.
    Criteria:
    - Admitted to ICU
    - Intubated/Prolonged mechanical ventilation OR Stroke OR Cardiac Arrest
    """
    print("Connecting to DuckDB for out-of-core processing...")
    con = duckdb.connect(database=':memory:')
    
    # Define paths to MIMIC-IV files (adjust based on extraction vs gz)
    # Using the split directory paths based on our inventory
    diagnoses_path = os.path.join(data_dir, "mimic-iv-3.1-20260514T144455Z-3-005/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz")
    icustays_path = os.path.join(data_dir, "mimic-iv-3.1-20260514T144455Z-3-004/mimic-iv-3.1/icu/icustays.csv.gz")
    procedureevents_path = os.path.join(data_dir, "mimic-iv-3.1-20260514T144455Z-3-004/mimic-iv-3.1/icu/procedureevents.csv.gz")
    
    print("Finding target patients (Stroke, Cardiac Arrest, Dysphagia, Intubation)...")
    
    # Relevant ICD-10 codes (simplified for demonstration)
    # I63: Cerebral infarction (Stroke)
    # I46: Cardiac arrest
    # R13: Dysphagia
    
    query = f"""
    SELECT DISTINCT subject_id, hadm_id 
    FROM read_csv_auto('{diagnoses_path}')
    WHERE icd_code LIKE 'I63%' OR icd_code LIKE 'I46%' OR icd_code LIKE 'R13%'
    """
    
    target_admissions = con.execute(query).fetchdf()
    print(f"Found {len(target_admissions)} target admissions based on diagnoses.")
    
    # Save cohort
    os.makedirs(output_dir, exist_ok=True)
    cohort_path = os.path.join(output_dir, "target_cohort.csv")
    target_admissions.to_csv(cohort_path, index=False)
    print(f"Cohort saved to {cohort_path}")

if __name__ == "__main__":
    MIMIC_DATA_DIR = r"C:\Users\rohit\MultiModal\dataset"
    OUTPUT_DIR = r"C:\Users\rohit\MultiModal\icu_predictive_system\data\processed"
    extract_cohort(MIMIC_DATA_DIR, OUTPUT_DIR)
