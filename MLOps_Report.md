# MLOps Report - FITE Classification Challenge

## 1. Project Overview

This project solves a multi-class classification task using anonymized tabular data. The main target is to predict the `target` class for every row in `test_data.csv` and generate a valid Kaggle submission file named `submission.csv`.

The solution was designed to be reproducible, easy to review, and traceable through MLflow and DVC.

## 2. Reproducibility Setup

We fixed random seeds and used a consistent validation strategy to make the results reproducible. The main training code is contained in:

```text
classification_pipeline.py
```

The notebook:

```text
classification_pipeline.ipynb
```

documents the full workflow, displays the data analysis, shows the model comparisons, and calls the pipeline code to regenerate the final outputs.

## 3. Experiment Tracking With MLflow

MLflow was integrated into the training code. Each experiment logs:

- model name
- number of folds
- random state
- accuracy mean
- accuracy standard deviation
- balanced accuracy mean
- macro F1 mean
- fold metrics

The final ensemble run also logs:

- ensemble model names
- ensemble weights
- OOF accuracy
- OOF macro F1
- generated `submission.csv`
- evaluation artifacts

The MLflow tracking database is:

```text
mlflow.db
```

To open the MLflow dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

### MLflow Screenshot

Insert the MLflow UI screenshot here. The screenshot should show the experiment runs comparison table with several models and metrics.

## 4. Experiments Run

We ran more than three experiments. The main model families included:

- LightGBM
- Random Forest
- Extra Trees
- HistGradientBoosting
- XGBoost
- Gradient Boosting
- Logistic Regression

We also added course-inspired baseline experiments:

- KNN
- Logistic Regression
- Decision Tree
- Bagging
- AdaBoost

These baselines document the model-selection path and show that the final model was not chosen without comparison.

## 5. Validation Strategy

The data is strongly imbalanced, so accuracy alone is not enough. We used:

- `StratifiedKFold`
- `Macro F1`
- `Balanced Accuracy`
- confusion matrix
- classification report

Macro F1 was emphasized because it gives each class a more balanced contribution, instead of allowing the majority class to dominate the score.

## 6. Robust Validation Check

To reduce the risk of depending on one lucky validation split, we added a robust validation audit across multiple `random_state` values:

```text
7, 42, 123
```

The robust validation results are saved in:

```text
classification_artifacts/robust_validation_by_seed.csv
classification_artifacts/robust_validation_summary.csv
```

This helps verify that strong models remain strong across different splits.

## 7. Final Model

The final prediction is generated using a probability ensemble. The ensemble combines several models using out-of-fold validation probabilities.

Final OOF performance:

```text
OOF Accuracy: 0.998125
OOF Macro F1: 0.992639
```

The final Kaggle file is:

```text
submission.csv
```

## 8. Data Versioning With DVC

DVC was initialized in the project folder. The raw CSV files are not committed directly to GitHub. Instead, the repository contains small `.dvc` tracking files:

```text
train_data.csv.dvc
test_data.csv.dvc
sample_submission.csv.dvc
```

The actual data is stored in a Google Drive for Desktop DVC remote:

```text
G:\My Drive\FITE_Classification_Challenge_DVC
```

The data was pushed using:

```bash
dvc push
```

A teammate can restore the data with:

```bash
dvc pull
```

If their Google Drive path is different, they can update the remote path first:

```bash
dvc remote modify storage url "PATH_TO_TEAMMATE_GOOGLE_DRIVE_FOLDER"
dvc pull
```

## 9. GitHub Repository

The GitHub repository contains:

- notebook
- Python pipeline
- README
- requirements
- DVC tracking files
- MLflow database
- evaluation artifacts

The raw CSV data files are intentionally ignored by Git and are managed through DVC.

Repository:

```text
https://github.com/AhmadMurad10/FITE-Classification-Challenge.git
```

## 10. Final Deliverables

The final deliverables are:

- `classification_pipeline.ipynb`
- `classification_pipeline.py`
- `MLOps_Report.md` or exported PDF version
- `submission.csv`
- GitHub repository with DVC tracking files
- MLflow UI screenshot
