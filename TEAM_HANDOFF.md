# ملخص تسليم المشروع للفريق

## ما هو المشروع؟

المشروع هو `FITE Classification Challenge`. المطلوب توقع قيمة `target` لكل صف في `test_data.csv`. الملف النهائي للرفع على Kaggle هو:

```text
submission.csv
```

## الملفات المهمة

- `classification_pipeline.ipynb`: النوتبوك الأساسي للشرح والتشغيل.
- `classification_pipeline.py`: نسخة Python مستقلة من نفس الـ pipeline.
- `submission.csv`: ملف الرفع على Kaggle.
- `MLOps_Report.docx`: تقرير MLOps النهائي.
- `README.md`: تعليمات التشغيل وDVC وMLflow.
- `requirements.txt`: المكتبات المطلوبة.
- `mlflow.db`: قاعدة MLflow المحلية التي تحتوي التجارب.
- `train_data.csv.dvc`, `test_data.csv.dvc`, `sample_submission.csv.dvc`: ملفات DVC التي تتبع الداتا.
- `classification_artifacts/`: ملفات التقييم والتحليلات.

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

بعدها نفتح الرابط المحلي ونأخذ screenshot من جدول التجارب:

```text
http://127.0.0.1:5000
```

## ماذا فعلنا في الحل؟

- حللنا شكل الداتا وتوزيع الكلاسات.
- استخدمنا `Macro F1` لأن الداتا غير متوازنة.
- استخدمنا `StratifiedKFold` للتقييم.
- جربنا نماذج بسيطة مثل KNN وLogistic Regression وDecision Tree وBagging وAdaBoost.
- جربنا نماذج أقوى مثل LightGBM وRandomForest وHistGradientBoosting وXGBoost.
- أضفنا Robust Validation على أكثر من `random_state`.
- أضفنا Test-like slice audit لفحص أداء النماذج على صفوف التدريب الأقرب لتوزيع الاختبار.
- أضفنا Duplicate policy audit لتوثيق قرار التعامل مع الصفوف المتكررة.
- أضفنا Nested greedy ensemble audit لفحص خطر overfitting في أوزان الدمج.
- استخدمنا Ensemble محافظ لتوليد `submission.csv`.
- سجلنا التجارب في MLflow.
- استخدمنا DVC لتتبع ملفات الداتا بدون رفع CSV الخام على GitHub.

## DVC للفريق

الـ remote مضبوط على Google Drive API:

```text
gdrive://12AfjjB_qMloHxg0oZsQ4fbWs3zYJCeAo
```

أي عضو يريد استخدامه يحتاج:

- أن يكون مضافاً إلى Google OAuth test users.
- أن يكون لديه صلاحية على فولدر Google Drive.
- أن يشغل:

```bash
dvc pull
```

إذا طلب Google تسجيل دخول، يستخدم الحساب الذي لديه صلاحية على الفولدر.

لا نرفع أبداً:

- `.dvc/config.local`
- أي ملف `client_secret*.json`
- ملفات CSV الخام

## قبل التسليم

- تأكد أن `submission.csv` يحتوي عمودين فقط: `ID`, `target`.
- تأكد أن `classification_pipeline.ipynb` يعمل من البداية للنهاية.
- تأكد أن `classification_pipeline.py` يعمل مباشرة.
- خذ screenshot من MLflow UI.
- ارفع الملفات المطلوبة منفصلة بدون ضغط إذا كان نموذج التسليم يطلب ذلك.

