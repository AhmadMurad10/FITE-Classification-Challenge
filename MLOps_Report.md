# MLOps Report - FITE Classification Challenge

## Project Summary

This project solves a multi-class classification task on anonymized tabular data. The pipeline trains multiple models, compares them with stratified validation, logs experiments with MLflow, tracks data with DVC, and generates `submission.csv`.

The final selection uses a conservative probability ensemble. The current final artifacts report:

- OOF Macro F1: `0.991702`
- OOF Accuracy: `0.997812`
- Final weight strategy: `stable_holdout_average`
- Final prediction distribution: `class1=81`, `class2=188`, `class3=2931`

## MLflow

Experiments are logged in `mlflow.db`. To open the dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

The final report should include a screenshot of the MLflow experiment table.

## Validation

The workflow uses:

- `StratifiedKFold`
- `Macro F1`
- `Balanced Accuracy`
- confusion matrix
- robust validation across multiple random states
- adversarial train/test distribution check
- test-like slice audit
- duplicate-policy audit
- holdout and nested ensemble audits

## Main Artifacts

Important generated files:

- `classification_artifacts/cv_results.csv`
- `classification_artifacts/robust_validation_summary.csv`
- `classification_artifacts/duplicate_policy_summary.csv`
- `classification_artifacts/test_like_slice_summary.csv`
- `classification_artifacts/nested_greedy_summary.json`
- `classification_artifacts/master_comparison.csv`
- `classification_artifacts/ensemble_info.json`
- `submission.csv`

## DVC

Raw CSV files are tracked with DVC instead of being committed directly to Git:

```text
train_data.csv.dvc
test_data.csv.dvc
sample_submission.csv.dvc
```

To restore data on another machine:

```bash
dvc pull
```

If the Google Drive path differs, update the DVC remote locally, then run `dvc pull`.

## Deliverables

Submit:

- `classification_pipeline.ipynb`
- `classification_pipeline.py`
- `MLOps_Report.docx`

