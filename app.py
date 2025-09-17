# app.py

import gradio as gr
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import io

# تحميل النموذج المدرب
model = load("models/random_forest_model.joblib")

# الميزات المطلوبة في الإدخال اليدوي
FEATURES = ['V14', 'V17', 'V10', 'Amount']

def predict_single(V14, V17, V10, Amount):
    """
    توقع الاحتيال لعملية واحدة بناءً على القيم المدخلة.
    """
    input_data = pd.DataFrame([[V14, V17, V10, Amount]], columns=FEATURES)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    result = "🚨 عملية احتيالية" if prediction == 1 else "✅ عملية سليمة"
    return f"{result}\nنسبة الاحتمال: {probability:.2%}"

def predict_csv(file):
    """
    توقع الاحتيال لعدة عمليات من ملف CSV.
    """
    df = pd.read_csv(file)
    # تحقق من وجود الأعمدة التي درب عليها النموذج 
    expected_features = model.feature_names_in_
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        return f"❌ الملف ينقصه الأعمدة التالية: {missing}", None, None, None

    #التنبؤ
    preds = model.predict(df[expected_features])
    probs = model.predict_proba(df[expected_features])[:, 1]

    df['Prediction'] = ['احتيالية' if p == 1 else 'سليمة' for p in preds]

    df['Probability'] = [f"{prob:.2%}" for prob in probs]

    fraud_count = sum(preds)
    total = len(preds)
    summary = f"📊 عدد العمليات: {total}\n🚨 عمليات احتيالية مكتشفة: {fraud_count}\n✅ عمليات سليمة: {total - fraud_count}"

    #📊 رسم بياني دائري
    fig1, ax1 = plt.subplots()
    ax1.pie([fraud_count, total - fraud_count], labels=['سليمة', 'احتيالية'], autopct='%1.1f%%', colors=['red', 'green'])
    ax1.set_title("نسبة العمليات الاحتيالية")

    #📈 رسم بياني شريطي حسب المبلغ
    df['Amount_bin'] = pd.cut(df['Amount'], bins=5)
    amount_group = df.groupby('Amount_bin')['Prediction'].apply(lambda x: (x == 'احتيالية').mean())
                              
    fig2, ax2 = plt.subplots()
    amount_group.plot(kind='bar', color='orange', ax=ax2)
    ax2.set_ylabel("نسبة الاحتيال")
    ax2.set_title("نسبة الاحتيال حسب المبلغ")

    return summary, df[['V14', 'V17', 'V10', 'Amount', 'Prediction', 'Probability']] , fig1, fig2

# واجهة الإدخال اليدوي
manual_interface = gr.Interface(
    fn=predict_single,
    inputs=[
        gr.Number(label="V14"),
        gr.Number(label="V17"),
        gr.Number(label="V10"),
        gr.Number(label="Amount")
    ],
    outputs=gr.Textbox(label="نتيجة التنبؤ"),

    title="🔍 كشف الاحتيال - إدخال يدوي",
    description="""
    أدخل خصائص العملية المالية للحصول على التنبؤ.

    شرح الاعمدة:
    - V14, V17, V10: مكونات مشتقة من بيانات المعاملة باستخدام تحليل مكونات (PCA)، وتستخدم لاكتشاف الأنماط غير الطبيعية.
    - Amount: قيمة المعاملة المالية، وقد تكون مؤشراً على الاحتيال في بعض الحالات.
    """
)

# واجهة رفع ملف CSV
csv_interface = gr.Interface(
    fn=predict_csv,
    inputs=gr.File(label="📁 رفع ملف CSV يحتوي على العمليات"),
    outputs=[
        gr.Textbox(label="ملخص النتائج"),
        gr.Dataframe(label="تفاصيل العمليات"),
        gr.Plot(label="📊 نسبة الاحتيال"),
        gr.Plot(label="📈 نسبة الاحتيال حسب المبلغ")
    ],
    title="📂 كشف الاحتيال - تحليل ملف CSV",
    description="""
    ارفع ملف يحتوي على الأعمدة V14 وV17 وV10 وAmount لتحليل العمليات دفعة واحدة
    شرح الاعمدة:
    - V14, V17, V10: تمثل انماطاً رقمية مشتقة من بيانات المعاملة، ونستخدم لتحديد السلوك المشبوه.
    - Amount: قيمة المعاملة المالية، وقد تكون مؤشراً على الاحتيال في بعض الحالات.
    """
)

# دمج الواجهتين في تبويب واحد
demo = gr.TabbedInterface(

    interface_list=[manual_interface, csv_interface],
    tab_names=["🔢 إدخال يدوي", "📁 رفع ملف CSV"]
)

if __name__ == "__main__":
    demo.launch()

