"""
Data splitting module for AEGIS project.
Implements temporal validation splitting with strict subject_id grouping.
"""

import duckdb
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def get_admission_data(admissions_path: Path, features_path: Path) -> pd.DataFrame:
    """
    Get admission times from admissions file or fallback to features file.
    """
    if admissions_path.exists():
        logging.info("Loading admission times from MIMIC-IV admissions.csv.gz...")
        # DuckDB can read compressed CSVs natively
        query = f"""
            SELECT 
                subject_id, 
                hadm_id, 
                EXTRACT(YEAR FROM admittime) as admission_year
            FROM read_csv_auto('{str(admissions_path).replace("\\", "/")}')
        """
        adm_df = duckdb.query(query).df()
        
        # We need stay_id as well, so let's get it from features
        query_feat = f"""
            SELECT DISTINCT subject_id, hadm_id, stay_id
            FROM '{str(features_path).replace("\\", "/")}'
        """
        feat_df = duckdb.query(query_feat).df()
        
        df = feat_df.merge(adm_df, on=['subject_id', 'hadm_id'], how='left')
    else:
        logging.info("admissions.csv.gz not found. Extracting year from physiological_features.parquet...")
        query = f"""
            SELECT DISTINCT 
                subject_id, 
                hadm_id, 
                stay_id, 
                EXTRACT(YEAR FROM MIN(charttime)) as admission_year
            FROM '{str(features_path).replace("\\", "/")}'
            GROUP BY subject_id, hadm_id, stay_id
        """
        df = duckdb.query(query).df()
        
    return df

def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign splits (train/val/test) based on the earliest admission year for each subject_id.
    """
    # Get the earliest admission year for each subject
    first_adm = df.groupby('subject_id')['admission_year'].min().reset_index()
    first_adm.rename(columns={'admission_year': 'first_admission_year'}, inplace=True)
    
    df = df.merge(first_adm, on='subject_id', how='left')
    
    # Calculate year distribution
    years = sorted(df['first_admission_year'].dropna().unique())
    
    # Are we in the real years (<= 2025) or MIMIC shifted years (2100+)?
    if not years:
        logging.warning("No years found. Falling back to simple modulo split.")
        df['split'] = df['subject_id'] % 10
        df['split'] = df['split'].map({0: 'val', 1: 'val', 2: 'test', 3: 'test'})
        df['split'] = df['split'].fillna('train')
    elif max(years) <= 2025:
        logging.info("Using standard year splitting: <=2017 Train, 2018 Val, >=2019 Test")
        def get_split(year):
            if pd.isna(year): return 'train'
            if year <= 2017: return 'train'
            elif year == 2018: return 'val'
            else: return 'test'
            
        df['split'] = df['first_admission_year'].apply(get_split)
    else:
        logging.info("Using distribution-based splitting (70/15/15) for shifted years")
        # Find cutoffs for 70/15/15
        s = df.drop_duplicates('subject_id')['first_admission_year'].dropna()
        q70 = s.quantile(0.70)
        q85 = s.quantile(0.85)
        
        def get_split(year):
            if pd.isna(year): return 'train'
            if year <= q70: return 'train'
            elif year <= q85: return 'val'
            else: return 'test'
            
        df['split'] = df['first_admission_year'].apply(get_split)
        
    # We only need [subject_id, hadm_id, stay_id, split, admission_year]
    return df[['subject_id', 'hadm_id', 'stay_id', 'split', 'admission_year']]

def validate_split(df: pd.DataFrame):
    """
    Validate no subject_id leakage between splits and print statistics.
    """
    logging.info("\n--- Split Validation ---")
    
    # Check leakage
    split_subjects = {
        split: set(df[df['split'] == split]['subject_id'])
        for split in df['split'].unique()
    }
    
    leakage = False
    splits = list(split_subjects.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = split_subjects[splits[i]].intersection(split_subjects[splits[j]])
            if overlap:
                logging.error(f"LEAKAGE DETECTED between {splits[i]} and {splits[j]}: {len(overlap)} subjects")
                leakage = True
                
    if not leakage:
        logging.info("Success: No subject_id leakage detected between splits.")
        
    # Print stats
    logging.info("\n--- Split Statistics ---")
    for split in ['train', 'val', 'test']:
        if split not in df['split'].values:
            continue
        split_df = df[df['split'] == split]
        n_patients = split_df['subject_id'].nunique()
        n_stays = split_df['stay_id'].nunique()
        # Positive labels would require joining with targets.parquet, omitted for brevity as requested
        logging.info(f"{split.upper()}:")
        logging.info(f"  Patients: {n_patients}")
        logging.info(f"  Stays: {n_stays}")
        
def main():
    admissions_path = Path(r"C:\Users\rohit\MultiModal\dataset\mimic-iv-3.1-20260514T144455Z-3-004\mimic-iv-3.1\hosp\admissions.csv.gz")
    features_path = Path(r"C:\Users\rohit\MultiModal\AEGIS\data\processed\physiological_features.parquet")
    output_path = Path(r"C:\Users\rohit\MultiModal\AEGIS\data\processed\split_assignments.csv")
    
    # Make sure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = get_admission_data(admissions_path, features_path)
    split_df = assign_splits(df)
    
    validate_split(split_df)
    
    split_df.to_csv(output_path, index=False)
    logging.info(f"\nSaved split assignments to {output_path}")

if __name__ == "__main__":
    main()
