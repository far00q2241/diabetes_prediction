import streamlit as st
import pandas as pd
import joblib


# Load model and scaler
model = joblib.load("diabetes_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")


# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)


# Title
st.title("🩺 Diabetes Prediction App")
st.write("Enter the patient details below to predict diabetes.")


# -----------------------------
# Patient Details
# -----------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

hypertension = st.selectbox(
    "Hypertension",
    ["No", "Yes"]
)

heart_disease = st.selectbox(
    "Heart Disease",
    ["No", "Yes"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=70.0,
    value=25.0
)

hba1c = st.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=15.0,
    value=5.5
)

blood_glucose = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=300,
    value=120
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

smoking_history = st.selectbox(
    "Smoking History",
    [
        "No Info",
        "never",
        "former",
        "current",
        "ever",
        "not current"
    ]
)


# -----------------------------
# Create Input Data
# -----------------------------

input_data = {
    "age": age,
    "hypertension": 1 if hypertension == "Yes" else 0,
    "heart_disease": 1 if heart_disease == "Yes" else 0,
    "bmi": bmi,
    "HbA1c_level": hba1c,
    "blood_glucose_level": blood_glucose,

    "gender_Male": 1 if gender == "Male" else 0,
    "gender_Other": 1 if gender == "Other" else 0,

    "smoking_history_current": 1 if smoking_history == "current" else 0,
    "smoking_history_ever": 1 if smoking_history == "ever" else 0,
    "smoking_history_former": 1 if smoking_history == "former" else 0,
    "smoking_history_never": 1 if smoking_history == "never" else 0,
    "smoking_history_not current": 1 if smoking_history == "not current" else 0
}


input_df = pd.DataFrame([input_data])


# Make sure columns are in the same order as training
input_df = input_df[
    [
        "age",
        "hypertension",
        "heart_disease",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "gender_Male",
        "gender_Other",
        "smoking_history_current",
        "smoking_history_ever",
        "smoking_history_former",
        "smoking_history_never",
        "smoking_history_not current"
    ]
]


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict Diabetes"):

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Prediction probability
    probability = model.predict_proba(input_scaled)[0]


    # -----------------------------
    # Display Result
    # -----------------------------

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Patient is likely to have Diabetes.")
    else:
        st.success("✅ Patient is unlikely to have Diabetes.")


    st.subheader("Prediction Probability")

    st.write(
        f"**No Diabetes:** {probability[0] * 100:.2f}%"
    )

    st.write(
        f"**Diabetes:** {probability[1] * 100:.2f}%"
    )
