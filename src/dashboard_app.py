import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('data/rf_model.pkl')

st.title("Patient Readmission Risk Predictor")

# Sidebar inputs
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Age (years)", min_value=0, max_value=120, value=50)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
# Add more input fields matching your features if needed

# Prepare input dataframe for prediction
input_data = pd.DataFrame({
    'age': [age],
    'gender_Male': [1 if gender=="Male" else 0],
    # Add remaining features with default values if necessary
})

# Prediction
if st.button("Predict Readmission Risk"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    st.write(f"Predicted Readmission: {'Yes' if prediction[0]==1 else 'No'}")
    st.write(f"Probability: {probability*100:.2f}%")