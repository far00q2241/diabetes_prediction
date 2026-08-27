import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("diabetes_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes risk.")

# -----------------------------
# User Inputs
# -----------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=30)

hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0)

hba1c = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)

blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=120)

gender = st.selectbox("Gender", ["Female", "Male", "Other"])

smoking = st.selectbox(
    "Smoking History",
    ["No Info", "never", "former", "current", "ever", "not current"]
)

# -----------------------------
# Create Input Data
# -----------------------------
input_data = {
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": blood_glucose,

    "gender_Male": 1 if gender == "Male" else 0,
    "gender_Other": 1 if gender == "Other" else 0,

    "smoking_history_current": 1 if smoking == "current" else 0,
    "smoking_history_ever": 1 if smoking == "ever" else 0,
    "smoking_history_former": 1 if smoking == "former" else 0,
    "smoking_history_never": 1 if smoking == "never" else 0,
    "smoking_history_not current": 1 if smoking == "not current" else 0
}

input_df = pd.DataFrame([input_data])

# -----------------------------
# Scale Numerical Features
# -----------------------------
num_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]

input_df[num_cols] = scaler.transform(input_df[num_cols])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Diabetes"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ The patient is likely to have Diabetes.")
    else:
        st.success("✅ The patient is unlikely to have Diabetes.")

    st.subheader("Prediction Probability")

    st.write(f"**No Diabetes:** {probability[0]*100:.2f}%")
    st.write(f"**Diabetes:** {probability[1]*100:.2f}%")
