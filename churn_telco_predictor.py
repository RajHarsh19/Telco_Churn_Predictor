import streamlit as st
import numpy as np
import h5py
import pickle
import os
import pandas as pd

# Load the trained model
model_path = "churn_model.h5"

st.title("Customer Churn Predictor 📉")

# Check if model exists
if not os.path.exists(model_path):
    st.error("⚠️ Model file 'churn_model.h5' not found! Please train and save the model.")
else:
    # Load the model from .h5 file using h5py and pickle
    with h5py.File(model_path, "r") as f:
        model_bytes = f["model"][()]  # Retrieve model bytes
        model = pickle.loads(model_bytes.tobytes())  # Deserialize model

    # User input for essential customer features
    st.sidebar.header("Enter Customer Information")
    tenure = st.sidebar.number_input("Tenure (months):", min_value=0, step=1)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($):", min_value=0.0, step=0.1)
    total_charges = st.sidebar.number_input("Total Charges ($):", min_value=0.0, step=0.1)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    # Convert categorical inputs to numerical format
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

    input_data = np.array([[
        tenure, monthly_charges, total_charges,
        contract_map[contract], internet_map[internet_service]
    ]])

    st.write("Input data shape:", input_data.shape)

    if st.sidebar.button("Predict Churn ⚡"):
        try:
            prediction = model.predict(input_data)
            churn_label = "Yes" if prediction[0] == 1 else "No"
            if churn_label == "Yes":
                st.error(f"🚨 Churn Prediction: {churn_label}")
            else:
                st.success(f"✅ Churn Prediction: {churn_label}")
        except Exception as e:
            st.error(f"⚠️ Prediction error: {str(e)}")
