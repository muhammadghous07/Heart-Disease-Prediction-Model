import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model, scaler, columns
model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('columns.pkl')

#Using Streamlit 
st.title("Heart Disease Prediction App")

age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 70, 210, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }])

    # Encode like training
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease Detected")
