# 📌 Jadeer – AI-Based Credit Risk Assessment

A simple and interactive web app built with Streamlit that helps predict whether a person is likely to default on a loan using a pre-trained machine learning model (XGBoost).

---

## 🎯 Project Objective

Jadeer helps individuals or financial institutions make smarter loan decisions by analyzing customer data and estimating the probability of repayment or default.

---

## ⚙️ How to Use

1. Fill in the customer's information (age, income, job type, etc.)
2. Click the "🔍 Predict" button
3. The app displays a clear probability of default or repayment

---

## 🧠 Model Overview

This app uses the **XGBoost** algorithm, a powerful and efficient classification model well-suited for structured/tabular data.

---

## 🧰 Tech Stack

- Python
- Streamlit
- XGBoost
- scikit-learn
- pandas
- numpy

---

## 🚀 Try It Online

Once deployed on Streamlit Cloud, you can access it directly at:

[Launch the App](https://share.streamlit.io/your-username/jadeer/main/app.py)

---

## 📁 Repository Contents

- `app.py` – Streamlit application
- `dv.pkl` – DictVectorizer (data transformer)
- `xgb_model.json` – Trained XGBoost model
- `requirements.txt` – Project dependencies
