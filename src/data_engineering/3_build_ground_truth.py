import duckdb
import pandas as pd
import json
import logging
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# Caveman logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Start build ground truth.")

    # Paths
    CHARTEVENTS_PATH = "C:/Users/rohit/MultiModal/dataset/chartevents_filtered.csv"
    NOTES_PATH = "C:/Users/rohit/MultiModal/dataset/mimic-iv-note-deidentified-free-text-clinical-notes-2.2-20260514T144228Z-3-001/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note/discharge.csv.gz"
    PROCEDURES_PATH = "C:/Users/rohit/MultiModal/dataset/mimic-iv-3.1-20260514T144455Z-3-004/mimic-iv-3.1/icu/procedureevents.csv.gz"
    DIAGNOSES_PATH = "C:/Users/rohit/MultiModal/dataset/mimic-iv-3.1-20260514T144455Z-3-005/mimic-iv-3.1/hosp/diagnoses_icd.csv.gz"
    ICUSTAYS_PATH = "C:/Users/rohit/MultiModal/dataset/mimic-iv-3.1-20260514T144455Z-3-004/mimic-iv-3.1/icu/icustays.csv.gz"
    OUT_DIR = Path("C:/Users/rohit/MultiModal/AEGIS/data/processed/")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    logger.info("Extract chartevents swallow flags.")
    chartevents_query = f"""
    SELECT 
        subject_id, 
        hadm_id, 
        stay_id, 
        charttime AS event_time, 
        'chartevent_swallow' AS label_type, 
        CAST(valuenum AS VARCHAR) AS label_value
    FROM read_csv_auto('{CHARTEVENTS_PATH}')
    WHERE itemid = 225118
    """
    df_chart = con.execute(chartevents_query).df()

    logger.info("Extract procedure suction flags.")
    procedures_query = f"""
    SELECT 
        subject_id, 
        hadm_id, 
        stay_id, 
        starttime AS event_time, 
        'procedure_suction' AS label_type, 
        CAST(itemid AS VARCHAR) AS label_value
    FROM read_csv_auto('{PROCEDURES_PATH}')
    WHERE itemid IN (224383, 227289) -- oral, subglottic suction
    """
    df_proc = con.execute(procedures_query).df()

    logger.info("Extract notes swallow markers.")
    regex_pattern = "aspiration pneumonia|oropharyngeal dysphagia|difficulty swallowing|failed water trial|swallow assessment|failed swallow|swallowing dysfunction|speech pathology|speech therapy|swallow eval|swallow screen|wet voice|dysphagia|aspiration|tracheostomy|decannulation|\\\\bnpo\\\\b|nil by mouth|modified diet|thickened liquids|fees|vfss|videofluoroscopy|fiberoptic endoscopic"
    
    notes_query = f"""
    WITH icu AS (
        SELECT subject_id, hadm_id, stay_id FROM read_csv_auto('{ICUSTAYS_PATH}')
    )
    SELECT 
        n.subject_id, 
        n.hadm_id, 
        i.stay_id, 
        n.charttime AS event_time, 
        'note_keyword' AS label_type, 
        '1' AS label_value
    FROM read_csv_auto('{NOTES_PATH}') n
    LEFT JOIN icu i ON n.hadm_id = i.hadm_id
    WHERE regexp_matches(lower(n.text), '{regex_pattern}')
    """
    df_notes = con.execute(notes_query).df()

    logger.info("Combine labels.")
    df_all = pd.concat([df_chart, df_proc, df_notes], ignore_index=True)
    df_all['event_time'] = pd.to_datetime(df_all['event_time'])

    out_csv = OUT_DIR / "ground_truth_labels.csv"
    df_all.to_csv(out_csv, index=False)
    logger.info(f"Save labels: {out_csv}")

    logger.info("Validate labels.")
    # Unique hadm_id for agreement
    all_hadm = df_all['hadm_id'].dropna().unique()
    
    hadm_notes = df_notes['hadm_id'].dropna().unique()
    hadm_proc = df_proc['hadm_id'].dropna().unique()

    df_val = pd.DataFrame({'hadm_id': all_hadm})
    df_val['has_note'] = df_val['hadm_id'].isin(hadm_notes).astype(int)
    df_val['has_proc'] = df_val['hadm_id'].isin(hadm_proc).astype(int)

    kappa = cohen_kappa_score(df_val['has_note'], df_val['has_proc'])
    
    # Pseudo gold standard = intersection
    df_val['gold'] = df_val['has_note'] & df_val['has_proc']
    
    tp = (df_val['has_note'] & df_val['gold']).sum()
    fp = (df_val['has_note'] & ~df_val['gold']).sum()
    fn = (~df_val['has_note'] & df_val['gold']).sum()

    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0

    stats = {
        "total_labels": len(df_all),
        "source_counts": df_all['label_type'].value_counts().to_dict(),
        "unique_admissions": len(all_hadm),
        "validation": {
            "cohen_kappa_notes_vs_proc": float(kappa),
            "note_pseudo_ppv": float(ppv),
            "note_pseudo_sensitivity": float(sens),
            "note_proc_agreement_count": int(df_val['gold'].sum())
        }
    }

    report_path = OUT_DIR / "label_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Save report: {report_path}")

    logger.info("Done.")

if __name__ == "__main__":
    main()
