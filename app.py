import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Model Predictor", layout="centered")
st.title("📈 Predict with the Best Model (Pickle Version)")
st.write("Enter feature values to predict using the trained best model.")

# Load model safely using pickle
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("Model file not found or failed to load. Ensure 'best_model.pkl' is in the same folder.")
    st.stop()

# Determine input feature count dynamically
try:
    num_features = model.n_features_in_
except AttributeError:
    num_features = 3  # fallback if not available

inputs = []
for i in range(num_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    inputs.append(val)

features = np.array([inputs])

if st.button("Predict"):
    try:
        prediction = model.predict(features)
        st.success(f"Predicted Value: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
