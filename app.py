import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

xgb_model = xgb.Booster()
xgb_model.load_model("xgb_model.json")

with open("dv.pkl", "rb") as f:
    dv = pickle.load(f)

st.set_page_config(page_title="جدير", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(to bottom, #f3e5f5, #ede7f6);
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            font-size: 3rem;
            color: #4a148c;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            text-align: center;
            color: #6a1b9a;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        .stButton > button {
            background-color: #6a1b9a;
            color: white;
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
            border: none;
        }
        .stButton > button:hover {
            background-color: #4a148c;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 class='main-title'>🟪 جدير</h1>
<p class='subtitle'>عَبِّ البيانات، وخلنا نقول لك إذا الشخص يستاهل القرض أو لا</p>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("العمر", value=30, min_value=18, step=1)
    income = st.number_input("الدخل الشهري", value=3000)
    expenses = st.number_input("المصروفات الشهرية", value=1000)
    assets = st.number_input("إجمالي الأصول", value=5000)
    home = st.selectbox("ملكية السكن", ['اختر', 'إيجار', 'مالك', 'أخرى'], index=0)
    marital = st.selectbox("الحالة الاجتماعية", ['اختر', 'أعزب', 'متزوج', 'مطلق'], index=0)

with col2:
    debt = st.number_input("إجمالي الديون", value=1000)
    amount = st.number_input("مبلغ القرض", value=1500)
    price = st.number_input("قيمة السلعة", value=2000)
    experience = st.number_input("عدد سنوات الخبرة", value=5)
    records = st.selectbox("سجل ائتماني سابق", ['اختر', 'لا', 'نعم'], index=0)
    job = st.selectbox("نوع الوظيفة", ['اختر', 'دوام كامل', 'دوام جزئي'], index=0)

term = st.slider("مدة القرض (بالأشهر)", 6, 72, 36)

required_fields = [home, marital, records, job]
if "اختر" in required_fields:
    st.warning("يرجى تعبئة جميع الحقول قبل التنبؤ")
else:
    if st.button("🔍 احسب التوقع"):
        client = {
            'age': age,
            'income': income,
            'expenses': expenses,
            'assets': assets,
            'debt': debt,
            'amount': amount,
            'price': price,
            'home': home,
            'marital': marital,
            'records': records,
            'job': job,
            'experience': experience,
            'term': term
        }

        X = dv.transform([client])
        dmatrix = xgb.DMatrix(X, feature_names=dv.feature_names_)
        prediction = float(xgb_model.predict(dmatrix)[0])
        default_chance = prediction * 100
        repay_chance = (1 - prediction) * 100

        st.subheader("نتيجة التوقع")
        st.progress(int(default_chance))

        if prediction >= 0.5:
            st.error(f"❌ مخاطرة عالية بالتعثر ({default_chance:.2f}%) | فرصة السداد: {repay_chance:.2f}%")
        else:
            st.success(f"✅ على الأرجح سيتم سداد القرض ({repay_chance:.2f}%) | نسبة التعثر: {default_chance:.2f}%")
