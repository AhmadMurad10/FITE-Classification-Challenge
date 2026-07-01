<div dir="rtl" align="right">

# تقرير MLOps - FITE Classification Challenge

## 1. ملخص المشروع

المشروع عبارة عن مسألة تصنيف متعدد الكلاسات على بيانات جدولية مجهولة المعنى. الهدف هو تدريب نموذج يتنبأ بقيمة `target` لكل صف في `test_data.csv`، ثم توليد ملف `submission.csv` مطابق لصيغة Kaggle.

اعتمدنا في التقييم الداخلي على `Macro F1` لأنه يعطي وزناً متوازناً لكل كلاس، وهذا مهم لأن توزيع الكلاسات غير متساو. لذلك لم نكتفِ بالدقة العامة، بل تابعنا أيضاً `Balanced Accuracy` و`Confusion Matrix` لفهم أين يخطئ النموذج.

## 2. سير العمل المختصر

| المرحلة | ماذا فعلنا؟ | لماذا؟ |
|---|---|---|
| قراءة البيانات | تحميل `train_data.csv` و`test_data.csv` و`sample_submission.csv` | التأكد من تطابق الأعمدة وصيغة الرفع |
| تحليل أولي | فحص توزيع الكلاسات والفروقات بين train وtest | معرفة طبيعة البيانات قبل النمذجة |
| بناء الخصائص | إضافة خصائص رقمية عامة مثل التفاعلات والنسب والملخصات الصفية | استخراج إشارات مفيدة من أعمدة مجهولة المعنى |
| تدريب النماذج | تجربة عدة نماذج مثل Decision Tree وBagging وRandom Forest وExtra Trees وLightGBM وXGBoost وHistGradientBoosting | مقارنة أكثر من عائلة نماذج بدل الاعتماد على نموذج واحد |
| التحقق الداخلي | استخدام `StratifiedKFold` وقياس `Macro F1` | تقييم عادل يحافظ على نسب الكلاسات داخل كل fold |
| تحليل الأخطاء | حفظ `OOF probabilities` و`Confusion Matrices` لكل موديل | معرفة أي الكلاسات أصعب، وليس فقط مشاهدة رقم نهائي |
| اختيار المرشحين | توليد ملف رفع لكل موديل قوي + ملف ensemble مرجعي | تجربة مرشحين واضحين وقابلين للتتبع |

## 3. تتبع التجارب باستخدام MLflow

تم دمج `MLflow` داخل كود التدريب. كل تجربة تسجل:

- اسم النموذج.
- أهم الإعدادات مثل نوع الموديل و`random_state`.
- مقاييس التقييم: `Accuracy` و`Balanced Accuracy` و`Macro F1`.
- ملفات artifacts مثل جداول النتائج، مصفوفات الالتباس، وملف الرفع النهائي.

لتشغيل واجهة MLflow محلياً:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

ثم فتح الرابط:

```text
http://127.0.0.1:5000
```

يجب وضع Screenshot من جدول تجارب MLflow داخل التقرير النهائي أو ضمن ملف التسليم، بحيث يظهر أكثر من run والمقاييس المسجلة لكل تجربة.

## 4. التجارب التي تم توثيقها

تم تدريب ومقارنة عدة اتجاهات:

| نوع التجربة | الهدف منها |
|---|---|
| Baseline models | معرفة أداء نماذج بسيطة قبل التعقيد |
| Decision Tree / Bagging | اختبار قدرة أشجار القرار على التقاط الأنماط الواضحة |
| Random Forest / Extra Trees | تقليل تذبذب الشجرة الواحدة عبر تجميع عدة أشجار |
| LightGBM / XGBoost | تجربة نماذج boosting قوية على البيانات الجدولية |
| HistGradientBoosting | بديل قوي من scikit-learn لبيانات tabular |
| Unweighted LightGBM | مقارنة النموذج مع وبدون class weights |
| Core 4 Features Model | اختبار نموذج صغير يعتمد على أقوى أربع خصائص فقط: `f10`, `f12`, `f14`, `f9` |
| ADASYN / BorderlineSMOTE audit | فحص هل توليد أمثلة إضافية للفئات الأقل عدداً يحسن `Macro F1` أم يضيف ضجيجاً |
| Reference Soft Voting Candidate | مرشح ensemble مبني على دمج احتمالات عدة نماذج |

## 5. الملفات التحليلية الناتجة

أهم الملفات التي ينتجها التدريب:

| الملف | فائدته |
|---|---|
| `classification_artifacts/cv_results.csv` | ترتيب النماذج حسب نتائج cross-validation |
| `classification_artifacts/model_submission_portfolio.csv` | قائمة ملفات الرفع المرشحة لكل موديل وترتيبها |
| `classification_artifacts/model_oof_confusion_matrices.csv` | مصفوفة الالتباس لكل موديل على OOF predictions |
| `classification_artifacts/ensemble_oof_confusion_matrix.csv` | مصفوفة الالتباس للـ ensemble النهائي |
| `classification_artifacts/oof_probability_audit.csv` | احتمالات OOF لكل صف لتحليل سلوك النموذج |
| `classification_artifacts/oof_hard_examples.csv` | الصفوف التي أخطأ بها النموذج أو كان غير واثق منها |
| `classification_artifacts/hard_sampling_adasyn_results.csv` | نتائج تجربة ADASYN وBorderlineSMOTE |
| `classification_artifacts/final_artifact_smoke_test.json` | فحص أن الـ artifact النهائي يعيد توليد نفس submission |
| `submission.csv` | ملف الرفع الأساسي |

## 6. ملفات الرفع المرشحة

بالإضافة إلى `submission.csv`، يولد الكود ملفات رفع منفصلة داخل:

```text
classification_artifacts/model_submissions/
```

هذه الملفات مفيدة للمقارنة بين المرشحين على Kaggle، مثل:

- `00_reference_soft_voting_candidate.csv`
- `01_bagging_tree_original.csv`
- ملفات LightGBM وXGBoost وRandomForest وغيرها.

وجود هذه الملفات يساعدنا على معرفة أي عائلة نماذج تعطي نتيجة أفضل، مع بقاء كل ملف مربوطاً بتجربته ونتيجته الداخلية.

## 7. تتبع البيانات باستخدام DVC

تم استخدام `DVC` لتتبع ملفات البيانات بدون رفع ملفات CSV الخام إلى GitHub.

الملفات الموجودة على GitHub هي ملفات تتبع صغيرة:

```text
train_data.csv.dvc
test_data.csv.dvc
sample_submission.csv.dvc
```

رابط مستودع GitHub:

```text
https://github.com/AhmadMurad10/FITE-Classification-Challenge
```

الـ DVC remote مضبوط على Google Drive:

```text
gdrive://12AfjjB_qMloHxg0oZsQ4fbWs3zYJCeAo
```

لاسترجاع البيانات على جهاز جديد:

```bash
pip install -r requirements.txt
dvc pull
```

إذا كان المستخدم جديداً على فولدر Google Drive، يجب إعطاؤه صلاحية الوصول إلى الفولدر قبل تشغيل `dvc pull`.

## 8. الملفات المطلوبة للتسليم

| الملف | الغرض |
|---|---|
| `classification_pipeline.ipynb` | النوتبوك الرئيسي وفيه التحليل والتجارب والنتائج |
| `classification_pipeline.py` | نسخة Python قابلة للتشغيل المباشر وتعيد إنتاج نفس المخرجات |
| `MLOps_Report.docx` | تقرير MLOps وفيه MLflow وDVC والتجارب والروابط |

## 9. ملاحظة عن اختيار النموذج النهائي

بما أن نتائج الفرق متقاربة، لم نعتمد على رقم واحد فقط. قارنا النماذج باستخدام `Macro F1`، وفحصنا الثبات عبر أكثر من تقسيم، وحفظنا ملفات رفع متعددة للموديلات القوية. بهذه الطريقة يبقى الاختيار النهائي مبنياً على تقييم داخلي واضح وليس على تجربة عشوائية واحدة.

</div>
