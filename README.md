# Bradycardia prediction from pre-administration physiological signals

This release contains only the analysis steps reported in the manuscript. It is organised in execution order and contains no study data, fitted models, local drive paths, or original working-file names.

## Directory order

1. `01_preprocessing/preprocess.py` derives the outcome and signal features from a five-minute pre-administration window.
2. `02_training/train_models.py` performs feature selection, model development, and internal held-out prediction.
3. `03_evaluation/evaluate_models.py` applies the fixed models to an evaluation cohort without retraining and creates the reported performance analyses.

## Data contract

Use a metadata CSV with one row per procedure. The required columns are:

| Column | Purpose |
| --- | --- |
| `case_id` | De-identified procedure identifier |
| `record_path` | Path to the locally available physiological recording |
| `administration_time` | Administration timestamp or `HH:MM:SS` time |
| `age`, `sex`, `height_cm`, `weight_kg` | Demographic predictors |

Supply a plain-text track list with one recording track per line. The record-path column and track names are intentionally configurable; do not commit data locations or locally identifying labels. The code reads recordings with the `vitaldb` Python package.

## Run

Create an environment with the packages in `requirements.txt`, then run the three stages with relative paths. Example paths below are placeholders only.

```bash
python 01_preprocessing/preprocess.py \
  --metadata data/metadata.csv \
  --tracks config/tracks.txt \
  --outcome-track device/heart_rate \
  --output output/features.csv

python 02_training/train_models.py \
  --data output/features.csv \
  --output output/training

python 03_evaluation/evaluate_models.py \
  --data data/evaluation_features.csv \
  --artifacts output/training/model_artifacts \
  --thresholds output/training/validation_thresholds.csv \
  --output output/evaluation
```

For internal held-out evaluation, use the generated `output/training/internal_holdout_features.csv` as `--data`; for external validation, provide the independently prepared feature table with the same generic schema. In both cases the evaluator never retrains models or refits preprocessing.

`preprocess.py` accepts `--horizon-minutes 0`, `1`, `2`, or `3`. Re-running the three stages for each value reproduces the prediction-horizon design, with a five-minute feature window ending at the requested number of minutes before administration.

## Methods represented in code

- Outcome: heart-rate decrease of at least 20% from the administration-time value, occurring 30 seconds to 10 minutes afterward.
- Features: ten summary statistics per retained signal track, ECG-derived HRV, photoplethysmography-derived PRV, and age, sex, height, weight, and BMI.
- Screening: normality-aware two-group testing at P < 0.05, followed by Spearman correlation/VIF filtering and RFECV using a Random Forest classifier.
- Development: stratified five-fold development with training-fold-only mode imputation, standardisation from the training fold, random oversampling, random-search tuning, five tree-based classifiers, and equal-weight soft voting.
- Evaluation: AUROC, accuracy, precision, recall, F1 score, Youden J and high-recall thresholds, calibration, SHAP, 2,000 case-level stratified bootstrap confidence intervals, and paired DeLong AUROC comparisons.

The random-search spaces are defined in `train_models.py`. Update them only from a prespecified analysis plan; they are not inferred from an evaluation cohort.

## Outputs and privacy

Generated tables, plots, models, raw recordings, and feature CSVs are excluded by `.gitignore`. Review every exported column before publication: a de-identified `case_id` is still study data and should not be committed by default.
