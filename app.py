import streamlit as st
import joblib
import pandas as pd

kbest = joblib.load('kbest.plk')
scale = joblib.load('scaler.plk')
model = joblib.load('model.plk')

st.set_page_config(page_title="cardio attack predition",page_icon="❤️")
st.title("Predict Cardio disease app ❤️")
st.markdown("Enter the patient details to predit the disease")

with st.form("user_input_form"):
    col1, col2 = st.columns(2)
    with col1:
        male = st.selectbox("Gender (Male=1, Female=0)", [1, 0])
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        currentSmoker = st.selectbox("Current Smoker (Yes=1, No=0)", [1, 0])
        cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=60, step=1)
        BPMeds = st.selectbox("On BP Medication (Yes=1, No=0)", [1, 0])
        prevalentStroke = st.selectbox("Prevalent Stroke (Yes=1, No=0)", [1, 0])
        prevalentHyp = st.selectbox("Prevalent Hypertension (Yes=1, No=0)", [1, 0])


    with col2:  
        diabetes = st.selectbox("Diabetes (Yes=1, No=0)", [1, 0])
        totChol = st.number_input("Total Cholesterol", min_value=50, max_value=700, step=1)
        sysBP = st.number_input("Systolic BP", min_value=50, max_value=300, step=1)
        diaBP = st.number_input("Diastolic BP", min_value=30, max_value=200, step=1)
        BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
        heartRate = st.number_input("Heart Rate", min_value=30, max_value=200, step=1)
        glucose = st.number_input("Glucose", min_value=40, max_value=400, step=1)


        submit = st.form_submit_button("predict")
        if submit:
            input_data = pd.DataFrame([[male, age, currentSmoker, cigsPerDay, BPMeds,
prevalentStroke, prevalentHyp, diabetes, totChol,
sysBP, diaBP, BMI, heartRate, glucose]])
            
            scaler = scale.transform(input_data)
            kbest_f=kbest.transform(scaler)
            prediction = model.predict(kbest_f)
            if(prediction[0]==1):
                st.success(f"Prediction: Positive")
            else:
                st.success(f"Prediction: Negative")