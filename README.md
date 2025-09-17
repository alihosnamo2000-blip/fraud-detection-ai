# Fraud Detection AI System

# Project Overview
This project aims to build an intelligent system that detects fraudulent financial transactions using machine learning techniques. It leverages real-world datasets and applies classification algorithms to identify suspicious patterns.


## Team Members
| AC.NO | Name   | Role         | Contributions                  |
|-------|--------|--------------|--------------------------------|
| 1     | Hasan  | Project Lead | Documentation & Integration    |
| 2     | Talal  | Data Analyst | Data Analysis & Preprocessing  |
| 3     | Shihab | Develober    | Model Development & Evaluation |

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
├── models                         # نتائج التدريب
├── src/                           # الأكواد المصدرية
│   ├── data/                      # معالجة البيانات وتنظيفها
│   │   └── preprocess.py          # دوال تنظيف وتحضير البيانات
│   ├── models/                    # النماذج الذكية
│   │   └── classifier.py          # بناء وتدريب النموذج
│   └── utils/                     # الوظائف المساعدة
│       └── metrics.py             # حساب الأداء والتقييم
├── notebooks/                     # دفاتر Jupyter
├── data                           # ملفات البيانات (CSV أو غيرها)
├── docs                           # التوثيق الإضافي (صور، مخططات، تقارير)
└── .gitignore                     # استثناء الملفات غير الضرورية من Git



## Usage

### Basic Usage




## Results

- **Model Accuracy**  :
