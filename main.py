import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open("best_model.pickle", "rb") as file:
    model = pickle.load(file)

st.title("Crop Prediction Demo")

# Input fields
nitrogen = st.text_input("Nitrogen", "50")
phosphorus = st.text_input("Phosphorus", "50")
potassium = st.text_input("Potassium", "50")
temperature = st.text_input("Temperature", "25")
humidity = st.text_input("Humidity", "50")
ph = st.text_input("pH", "7.0")
rainfall = st.text_input("Rainfall (mm)", "100")

if st.button("Predict Crop"):
    features = [
        float(nitrogen),
        float(phosphorus),
        float(potassium),
        float(temperature),
        float(humidity),
        float(ph),
        float(rainfall),
    ]
    prediction = model.predict(np.array([features]))
    st.write(f"The predicted crop is: {prediction}")
