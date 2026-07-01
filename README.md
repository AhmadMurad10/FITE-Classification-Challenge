# FITE Classification Challenge

This repository contains the reproducible notebook, Python pipeline, MLOps report, MLflow tracking database, and DVC data-versioning setup for the FITE Classification Challenge.

## Project Summary

The task is a multi-class classification problem on anonymized tabular data. The pipeline trains several models, compares them with validation, logs experiments with MLflow, and generates the Kaggle submission file:

```text
submission.csv
```

Main submitted files:

- `classification_pipeline.ipynb`: self-contained notebook with analysis, training, evaluation, and submission generation.
- `classification_pipeline.py`: standalone Python version of the same pipeline.
- `MLOps_Report.docx`: final MLOps report.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Restore the data with DVC:

```bash
dvc pull
```

Run the notebook:

```bash
jupyter notebook classification_pipeline.ipynb
```

Or run the Python pipeline:

```bash
python classification_pipeline.py
```

The output submission file is:

```text
submission.csv
```

## DVC Data Versioning

Raw CSV files are tracked with DVC instead of Git:

```text
train_data.csv.dvc
test_data.csv.dvc
sample_submission.csv.dvc
```

The DVC remote is configured as a Google Drive API remote:

```text
gdrive://12AfjjB_qMloHxg0oZsQ4fbWs3zYJCeAo
```

Team members should be added to the Google OAuth test users list and should have access to the shared Google Drive folder. After that, they can run:

```bash
dvc pull
```

If Google asks for authorization, sign in with the Google account that has access to the shared Drive folder. Do not commit `.dvc/config.local` or any `client_secret*.json` file.

## MLflow Experiment Tracking

Experiments are tracked in:

```text
mlflow.db
```

Open the MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open:

```text
http://127.0.0.1:5000
```

Take a screenshot of the runs comparison table for the final report.

## Validation Artifacts

Important generated reports:

- `classification_artifacts/cv_results.csv`
- `classification_artifacts/robust_validation_summary.csv`
- `classification_artifacts/duplicate_policy_summary.csv`
- `classification_artifacts/test_like_slice_summary.csv`
- `classification_artifacts/nested_greedy_summary.json`
- `classification_artifacts/master_comparison.csv`
- `classification_artifacts/ensemble_info.json`

## Arabic Team Notes

المشروع عبارة عن تصنيف متعدد الكلاسات على بيانات مجهولة المعنى. لذلك اعتمدنا على تحليل إحصائي، تحقق داخلي، وتجربة عدة نماذج بدل الاعتماد على تفسير أسماء الأعمدة.

استخدمنا `Macro F1` لأن توزيع الكلاسات غير متوازن، وسجلنا التجارب باستخدام `MLflow`، وتتبعنا ملفات الداتا باستخدام `DVC` حتى لا نرفع ملفات CSV الخام على GitHub.

ملف الرفع النهائي `submission.csv` يتم توليده من النماذج فقط، بدون تعديل يدوي وبدون استخدام أي labels من ملف الاختبار.

