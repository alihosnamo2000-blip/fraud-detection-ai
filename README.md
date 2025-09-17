# Fraud Detection AI System

# Project Overview
This project aims to build an intelligent system that detects fraudulent financial transactions using machine learning techniques. It leverages real-world datasets and applies classification algorithms to identify suspicious patterns.

## Team Members
| AC.NO | Name | Role | Contributions |
|----|------|------|---------------|
| 1 | Hasan | Project Lead | Documentation & Integration |
| 2 | Talal | Data Analyst | Data Analysis & Preprocessing |
| 3 | Shihab | Develober | Model Development |
| 3 | Ahmen | Develober | Model  Evaluation |

## Objectives
- Analyze and clean transaction data
- Train classification models to detect fraud
- Evaluate model performance using precision, recall, and F1-score
- Provide a simple interface for testing predictions

## Installation and Setup

### Technologies Used
- Python 3.10.11
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter, tensorflow, keras, torch, flask, gradio
- Environment: UV-based virtual environment (`.venv`)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone 
   cd 
   ```

2. Install dependencies using UV:
   ```bash
   uv sync
   ```

3. Run the project:
   ```bash
   uv run python main.py
   uv run python app.py

## 📁 Project Structure
fraud-detection-ai/
├── README.md                      # توثيق شامل للمشروع
├── pyproject.toml                 # إعدادات UV والمكتبات
├── .python-version                # إصدار بايثون المستخدم (3.10.11)
├── main.py                        # نقطة تشغيل المشروع
├── app.py                         # نقطة تشغيل واجهة المستخدم
├── models/                        # نتائج التدريب
│   └── evaluation_report.txt      # تقرير التقييم
│   └── random_forest_model.joblib # النماذج المحفوظة
├── src/                           # الأكواد المصدرية
│   ├── data/                      # معالجة البيانات وتنظيفها
│   │   └── preprocess.py          # دوال تنظيف وتحضير البيانات
│   ├── models/                    # النماذج الذكية
│   │   └── classifier.py          # بناء وتدريب النموذج
│   └── utils/                     # الوظائف المساعدة
│       └── metrics.py             # حساب الأداء والتقييم
├── notebooks/                     # دفاتر Jupyter
│   └── EDA.ipynb                  # تحليل استكشافي للبيانات
├── data/                          # ملفات البيانات (CSV أو غيرها)
│   └── creditcard.csv             # مجموعة البيانات المستخدمة
├── docs/                          # التوثيق الإضافي (صور، مخططات، تقارير)
│   └── architecture.png           # مخطط هيكل المشروع
│   └── EDA-summary.md             # توثيق كل مخرجات EDA.ipynb
└── .gitignore                     # استثناء الملفات غير الضرورية من Git


## Usage

### Basic Usage
```python
import gradio as gr
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import io

 # 1. تحميل البيانات
    data_path = 'data/creditcard.csv'
    df = load_data(data_path)

    # 2. تنظيف البيانات
    df_clean = clean_data(df)

    # 3. فصل الميزات عن الهدف
    X, y = split_features(df_clean, target_column='Class')

    # 4. تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = split_data(X, y)

    # طباعة عدد العينات قبل التوازن
    print(f" قبل التوازن: عدد عينات التدريب = {len(X_train)}")
    print(f" - عدد العينات غير الاحتيالية: {(y_train == 0).sum()}")
    print(f" - عدد العينات الاحتيالية: {(y_train == 1).sum()}")

    # 5. معالجة التوازن باستخدام undersampling
    train_df = X_train.copy()
    train_df['Class'] = y_train
    fraud = train_df[train_df['Class'] == 1]
    non_fraud = train_df[train_df['Class'] == 0].sample(n=len(fraud), random_state=42)
    balanced_df = pd.concat([fraud, non_fraud])

    X_train = balanced_df.drop(columns=['Class'])
    y_train = balanced_df['Class']

    # طباعة عدد العينات بعد التوازن
    print(f"\n بعد التوازن: عدد عينات التدريب = {len(X_train)}")
    print(f" - عدد العينات غير الاحتيالية: {(y_train == 0).sum()}")
    print(f" - عدد العينات الاحتيالية: {(y_train == 1).sum()}")

    # 6. تدريب النموذج
    model = train_model(X_train, y_train)

    # 7. تقييم النموذج وحفظ التقرير
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = compute_metrics(y_test, y_pred)
    auc = compute_auc(y_test, y_scores)
   ```


## Results

- **Model Accuracy**  : 93%
- **Recall**          : 0.9286
- **F1 Score**        : 0.0849
- **Confusion Matrix**: [[54909, 1955], [7, 91]]
- **AUC Score**       : 0.9747
- **Algorithm**       : Random Forest after data processes and balance using undersampling

## User Interface
Buit with Gradio, the interface includes two tabs

### Manual Input
 - **Input fields** : V14, V17, V10, Amount
- **Output**        : Fraudulent or Legitimate
- **Probability**   

### CSV Upload
- Upload full transaction files with all original columns
- Batch prediction and analysis
- Tabular results
- Pie chart showing fraud ratio
- Bar chart showing fraud rate by transaction amount

## Feature Explanation
- **V1 to V28**: PCA-transformed features derived from transaction behavior
- **Amount**   : Transaction value
- **Class**    : Ground truth label (1 = Fraudulent, 0 = Legitimate)

### Who Benefits From This Project?

| Stakeholder       | Benefit                                      |
|-------------------|----------------------------------------------|
| Banks             | Reduce financial losses due to fraud         |
| Payment platforms | Improve security and transaction monitoring  |
| Regulators        | Analyze suspicious patterns and behaviors    |
| Researchers       | Apply real-world AI to financial data        |

---

### How to Run

1. Train the model:

`bash
uv run python main.py
`

2. Launch the interface:

`bash
uv run python app.py
`

Then open the local link:
`
http://127.0.0.1:7860
