import streamlit as st
import pandas as pd
import numpy as np
import pickle
from minisom import MiniSom  # Add this line!
from tensorflow.keras.models import load_model

# 1. Load the trained components
@st.cache_resource
def load_fraud_model():
    model = load_model('fraud_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_fraud_model()

# 2. App Header
st.title("🛡️ Credit Card Fraud Detection System")
st.markdown("""
This application uses a **Hybrid Deep Learning Model (SOM + ANN)** to predict the probability 
that a credit card application is fraudulent based on 14 anonymized features.
""")

# 3. User Input Form
st.sidebar.header("Application Details")
def user_input_features():
    # We create 14 inputs to match the Australian Credit Dataset
    inputs = []
    for i in range(14):
        val = st.sidebar.number_input(f"Feature {i+1}", value=0.5, step=0.1)
        inputs.append(val)
    return np.array(inputs).reshape(1, -1)

input_df = user_input_features()

# 4. Prediction Logic
if st.button("Analyze Application"):
    # Scale the input just like the training data
    scaled_input = scaler.transform(input_df)
    
    # Get probability
    prediction = model.predict(scaled_input)
    probability = float(prediction[0][0])
    
    # Display Results
    st.subheader("Analysis Result")
    if probability > 0.5:
        st.error(f"⚠️ High Risk of Fraud: {probability*100:.2f}% probability")
    else:
        st.success(f"✅ Low Risk / Legitimate: {(1 - probability)*100:.2f}% confidence")
    
    st.progress(probability)