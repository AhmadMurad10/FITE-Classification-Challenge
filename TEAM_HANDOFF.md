# ملخص تسليم المشروع للفريق

## ما هو المشروع؟

المشروع هو `FITE Classification Challenge`. المطلوب توقع `target` لكل صف في `test_data.csv`، والنتيجة النهائية تخرج في:

```text
submission.csv
```

## الملفات المهمة

- `classification_pipeline.ipynb`: النوتبوك الأساسي للشرح والتشغيل.
- `classification_pipeline.py`: كود التدريب والتقييم وتوليد `submission.csv`.
- `submission.csv`: ملف الرفع على Kaggle.
- `README.md`: تعليمات التشغيل و DVC و MLflow.
- `requirements.txt`: المكتبات المطلوبة.
- `mlflow.db`: قاعدة MLflow المحلية التي فيها التجارب.
- `train_data.csv.dvc`, `test_data.csv.dvc`, `sample_submission.csv.dvc`: ملفات DVC الصغيرة التي تتبع الداتا.
- `classification_artifacts/`: نتائج التقييم والتقارير.

## كيف نشغل المشروع؟

```bash
pip install -r requirements.txt
dvc pull
python classification_pipeline.py
```

أو افتح:

```text
classification_pipeline.ipynb
```

وشغل الخلايا بالترتيب.

## كيف نفتح MLflow؟

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

بعدها نفتح الرابط المحلي ونأخذ screenshot من جدول التجارب.

## ماذا فعلنا في الحل؟

- حللنا شكل الداتا وتوزيع الكلاسات.
- استخدمنا `Macro F1` لأن الداتا غير متوازنة.
- استخدمنا `StratifiedKFold` للتقييم.
- جربنا نماذج مقارنة بسيطة مثل KNN و Logistic Regression و Decision Tree و Bagging و AdaBoost.
- جربنا موديلات أقوى مثل LightGBM و RandomForest و HistGradientBoosting و XGBoost.
- أضفنا Robust Validation على أكثر من `random_state`.
- استخدمنا Ensemble لأنه أعطى أفضل OOF Macro F1.
- سجلنا التجارب في MLflow.
- استخدمنا DVC حتى لا نرفع ملفات CSV الخام على GitHub.

## ماذا لا نغير؟

- لا نستخدم أي test labels.
- لا نستخدم `true_values.csv` إن وجد.
- لا نعدل `submission.csv` يدويا.
- لا نرفع ملفات CSV الخام على GitHub.
- لا نحذف فولدر Google Drive الخاص بـ DVC:

```text
FITE_Classification_Challenge_DVC
```

## قبل التسليم

- تأكد أن `submission.csv` فيه عمودين فقط: `ID`, `target`.
- تأكد أن Google Drive انتهى من مزامنة فولدر DVC.
- خذ screenshot من MLflow.
- شغل النوتبوك أو `classification_pipeline.py` مرة أخيرة للتأكد أن كل شيء قابل للإعادة.
