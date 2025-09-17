# 📊 Exploratory Data Analysis Summary

This document summarizes the key findings and visual insights obtained during the exploratory data analysis (EDA) phase of the fraud detection project.

---

## 📁 Dataset Overview

- **Source**: Credit card transactions dataset (Kaggle)
- **Total Records**: 284,807
- **Features**: 30 columns including anonymized features (V1–V28), `Time`, `Amount`, and target `Class`
- **Target Distribution**:
  - Non-fraudulent: 284,315 (≈ 99.83%)
  - Fraudulent: 492 (≈ 0.17%)

> ⚠️ The dataset is highly imbalanced, 

which will require special handling during model training.

---

## 🧼 Data Quality

- **Missing Values**: No missing values detected across all columns.
- **Outliers**: Detected in `Amount` and some anonymized features (e.g., V14, V17).
- **Skewness**: `Amount` and `Time` features show skewed distributions.

---

## 📈 Feature Insights

### 🔹 Time
- Represents the seconds elapsed between each transaction and the first transaction.
- No strong correlation with fraud, but may 

help in temporal pattern analysis.

### 🔹 Amount
- Fraudulent transactions tend to have lower amounts on average.
- Boxplot analysis shows clear separation between fraud and non-fraud cases.

### 🔹 Anonymized Features (V1–V28)
- Features like V14, V17, and V10 show stronger correlation with the target `Class`.
- Heatmap analysis reveals clusters of correlated features, which may be useful for dimensionality reduction.

---

## 📉 Correlation Analysis

- A correlation matrix was generated to identify relationships between features.
- Most features are weakly correlated with 

each other, except for a few pairs.
- Strongest positive correlation with fraud: V17
- Strongest negative correlation with fraud: V14

---

## 📊 Class Imbalance

- The extreme imbalance (≈ 1:577) poses a challenge for standard classifiers.
- Will consider techniques such as:
  - Oversampling (e.g., SMOTE)
  - Undersampling
  - Class weighting in model training

---

## 📌 Next Steps

- Normalize or scale `Amount` and `Time` 

features.
- Select top features based on correlation and visual relevance.
- Proceed to model development using classification algorithms.

---
