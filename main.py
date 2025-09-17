# main.py

import os
import pandas as pd
import joblib
from src.data.preprocess import load_data, clean_data, split_features, split_data
from src.models.classifier import train_model, evaluate_model, save_model
from src.utils.metrics import compute_metrics, compute_auc

def main():
    # 1. تحميل البيانات
    data_path = 'data/creditcard.csv'
    df = load_data(data_path)

    # 2. تنظيف البيانات
    df_clean = clean_data(df)

    # 3. فصل الميزات عن الهدف
    X, y = split_features(df_clean, target_column='Class')

    # 4. تقسيم البيانات إلى تدريب واختبار
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ✅ طباعة عدد العينات قبل التوازن
    print(f"🔢 قبل التوازن: عدد عينات التدريب = {len(X_train)}")
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

    # ✅ طباعة عدد العينات بعد التوازن
    print(f"\n⚖️ بعد التوازن: عدد عينات التدريب = {len(X_train)}")
    print(f" - عدد العينات غير الاحتيالية: {(y_train == 0).sum()}")
    print(f" - عدد العينات الاحتيالية: {(y_train == 1).sum()}")

    # 6. تدريب النموذج
    model = train_model(X_train, y_train)

    # 7. تقييم النموذج وحفظ التقرير
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = compute_metrics(y_test, y_pred)
    auc = compute_auc(y_test, y_scores)

    # عرض النتائج في الطرفية
    print("\n📊 Evaluation Metrics:")
    for key, value in metrics.items():
        if key == "Confusion Matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"AUC Score: {auc:.4f}")

    # حفظ التقرير في ملف نصي
    report_path = 'models/evaluation_report.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("📊 Evaluation Metrics\n\n")
        for key, value in metrics.items():
            if key == "Confusion Matrix":

                f.write(f"{key}:\n{value}\n\n")
            else:
                f.write(f"{key}: {value:.4f}\n")
        f.write(f"AUC Score: {auc:.4f}\n")

    # 8. حفظ النموذج
    save_model(model)
    joblib.dump(model, 'models/random_forest_model.joblib')
    print("\n✅ All steps completed successfully.")

if __name__ == "__main__":
    main()
