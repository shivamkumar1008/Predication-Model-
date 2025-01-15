import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title="Health Prediction App", page_icon="🏦", layout="centered")

# App title
st.title("Health Prediction App")
st.write("This app predicts the likelihood of **Diabetes** and **Heart Diseases** based on the input data.")

# Load models
# def load_model(file_path):
#     with open(file_path, 'wb') as file:
#         return pickle.load(file)

# diabetes_model_path = "classifier.pkl"
# heart_model_path = "model.pkl"

# diabetes_model = load_model('classifier.pkl')
# heart_model = load_model('model.pkl')

pickle.load(open('classifier.sav','rb'))
pickle.load(open('model.sav','rb'))
# Tabs for different predictions
diabetes_tab, heart_tab = st.tabs(["Diabetes Prediction", "Heart Disease Prediction"])

with diabetes_tab:
    st.header("Diabetes Prediction")
    
    # Inputs for diabetes prediction
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI (kg/m^2)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.2f")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    # Make prediction
    if st.button("Predict Diabetes"):
        diabetes_input = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
        diabetes_prediction = diabetes_model.predict(diabetes_input)[0]
        result = "Positive" if diabetes_prediction == 1 else "Negative"
        st.success(f"The prediction for Diabetes is: {result}")

with heart_tab:
    st.header("Heart Disease Prediction")

    # Inputs for heart disease prediction
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    cp = st.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3], index=1)
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)", options=[0, 1], index=0)
    restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", options=[0, 1, 2], index=1)
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes; 0 = No)", options=[0, 1], index=0)
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (0-2)", options=[0, 1, 2], index=1)
    ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-4)", options=[0, 1, 2, 3, 4], index=0)
    thal = st.selectbox("Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)", options=[1, 2, 3], index=1)

    # Make prediction
    if st.button("Predict Heart Disease"):
        heart_input = np.array([age, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
        heart_prediction = heart_model.predict(heart_input)[0]
        result = "Positive" if heart_prediction == 1 else "Negative"
        st.success(f"The prediction for Heart Disease is: {result}")

# Notes section
st.sidebar.title("About")
st.sidebar.info("This app uses machine learning models to predict the likelihood of Diabetes and Heart Diseases based on user input. Ensure accurate data entry for reliable predictions.")
