# FITE Classification Challenge

This repository contains the reproducible notebook and training pipeline for the
FITE Classification Challenge.

## Project Summary

The task is a multi-class classification problem on anonymized tabular data.
The final notebook trains several models, compares them using cross-validation,
logs experiments with MLflow, and generates the Kaggle submission file:

```text
submission.csv
```

Both submitted code files are complete:

- `classification_pipeline.ipynb` is self-contained and includes the full workflow split into readable notebook cells.
- `classification_pipeline.py` is a standalone Python version of the same pipeline for direct execution.

The notebook also includes two validation-audit sections:

- `course_baseline_results.csv`: simpler models inspired by the lectures, such as KNN, Logistic Regression, Decision Tree, Bagging, and AdaBoost.
- `robust_validation_summary.csv`: repeated cross-validation checks across multiple random seeds to verify that the strongest models are stable and not just lucky on one split.

## How To Run

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Restore the data with DVC:

```bash
dvc pull
```

3. Run the notebook or the Python pipeline:

```bash
jupyter notebook classification_pipeline.ipynb
```

or:

```bash
python classification_pipeline.py
```

4. Submit the generated file:

```text
submission.csv
```

## DVC Data Versioning

The raw CSV files are tracked with DVC instead of Git:

```text
train_data.csv.dvc
test_data.csv.dvc
sample_submission.csv.dvc
```

The DVC remote is currently configured to a Google Drive for Desktop folder:

```text
G:\My Drive\FITE_Classification_Challenge_DVC
```

If a teammate has Google Drive mounted at a different path, they should update
the remote locally before running `dvc pull`:

```bash
dvc remote modify storage url "PATH_TO_TEAMMATE_GOOGLE_DRIVE_FOLDER"
dvc pull
```

Example:

```bash
dvc remote modify storage url "G:\My Drive\FITE_Classification_Challenge_DVC"
dvc pull
```

## MLflow Experiment Tracking

Experiments are tracked in `mlflow.db`. To open the dashboard, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the local MLflow URL, go to the `FITE_Classification_Challenge`
experiment, and take a screenshot of the runs comparison table for the final
report.

## Arabic Team Notes

المشروع عبارة عن تصنيف متعدد الكلاسات. الداتا مجهولة المعنى، لذلك اعتمدنا على
تحليل إحصائي عام، معالجة عدم توازن الكلاسات، وتجربة عدة موديلات بدل الاعتماد
على معنى الأعمدة. كل تجربة مهمة يتم تسجيلها في MLflow، وملفات الداتا نفسها
مدارة عبر DVC حتى لا نرفع ملفات CSV الثقيلة مباشرة على GitHub.
