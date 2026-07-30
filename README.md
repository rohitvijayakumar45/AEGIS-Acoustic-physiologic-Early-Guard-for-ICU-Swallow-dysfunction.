# AEGIS — Acoustic-physiologic Early Guard for ICU Swallow-dysfunction

A publication-grade clinical decision support system for predicting speech and swallowing dysfunction in ICU patients. Built on MIMIC-IV clinical data with external validation on FEMH, SVD, and swallowing datasets.

## Project Structure

```
AEGIS/
├── conf/                          # Configuration (Hydra/OmegaConf)
│   └── config.yaml
├── src/
│   ├── data_engineering/
│   │   ├── 1_cohort_selection.py      # ICU cohort extraction with demographics
│   │   ├── 2_feature_extraction.py    # 4h rolling physiological features + SQI
│   │   ├── 3_build_ground_truth.py    # Label construction + validation (Kappa)
│   │   ├── 4_acoustic_feature_extraction.py  # COUGHVID/TORGO features
│   │   ├── signal_quality.py          # Signal Quality Index module
│   │   └── data_splitting.py          # Temporal validation + subject grouping
│   └── modeling/
│       ├── baseline_models.py         # Benchmarking ladder (LR→RF→XGB→LGBM→LSTM→TCN)
│       ├── clinical_evaluation.py     # Calibration, CI, DeLong, DCA, fairness
│       ├── explainability.py          # TreeSHAP + Integrated Gradients (Captum)
│       ├── ablation_and_error.py      # Feature ablation + FP/FN error analysis
│       ├── external_validation.py     # Zero-shot eval on FEMH/SVD/Swallowing
│       └── multimodal_fusion.py       # LSTM-MLP fusion (Phase G, future)
├── data/
│   ├── processed/                 # Generated features, labels, splits
│   └── models/                    # Trained models, metrics JSON
├── Dockerfile
├── environment.yml
├── requirements.txt
└── README.md
```

## Pipeline Execution Order

```
Phase A: Data Validity
  1_cohort_selection.py → 2_feature_extraction.py → 3_build_ground_truth.py → data_splitting.py

Phase B: Baselines
  baseline_models.py (trains all models on temporal train split, evaluates on test)

Phase C: Explainability
  explainability.py + ablation_and_error.py

Phase D: Clinical Evaluation
  clinical_evaluation.py (calibration, CIs, DeLong, DCA, fairness, prospective sim)

Phase E: External Validation
  external_validation.py (FEMH, SVD, Swallowing data)

Phase F: Deployment
  ONNX export, INT8 quantization, Docker
```

## Prediction Target

- **Primary outcome**: Reactive suctioning events / dysphagia indicators
- **Prediction horizon**: Multi-horizon (1h, 2h, 4h, 6h, 8h, 12h)
- **Gold-standard labels**: Heuristic matching from clinical notes + procedure logs

## Datasets

| Dataset | Type | Role |
|---------|------|------|
| MIMIC-IV Clinical | ICU vitals, labs, procedures | Primary training |
| MIMIC-IV Notes | Discharge summaries | Label extraction |
| FEMH | Clinical speech WAV | External validation |
| SVD | Voice disorder recordings | External validation |
| Swallowing data | Swallow signals + labels | External validation |
| COUGHVID | Cough audio + metadata | Acoustic features |
| TORGO | Dysarthric speech | Acoustic features |

## Reproducibility

```bash
# Option 1: Conda
conda env create -f environment.yml
conda activate aegis

# Option 2: pip
pip install -r requirements.txt

# Option 3: Docker
docker build -t aegis .
docker run aegis
```

## License

Research use only. MIMIC-IV requires PhysioNet credentialed access.
