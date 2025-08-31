import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

# --- Background Image ---
# Ensure the image URL is accessible.
# You can adjust opacity or other properties as needed.
BACKGROUND_IMAGE_URL = "https://static.vecteezy.com/system/resources/thumbnails/020/567/738/small/glass-of-wine-on-red-background-photo.jpeg"
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url({BACKGROUND_IMAGE_URL});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    /* Make the main content area more transparent for background to show */
    .stApp > header, .stApp > div {{
        background-color: rgba(30, 30, 30, 0); /* Darker background with more transparency (0.4 vs 0.7) */
        padding: 1rem;
        border-radius: 10px;
    }}
    .stAlert {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #A52A2A; /* Wine-like color */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1rem;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: #8B0000; /* Darker red on hover */
        border-color: #8B0000;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load the Trained Model ---
model_path = 'trained_model.sav'

if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found.")
    st.warning("Please run 'train_model.py' first to create the model file.")
    st.stop()

try:
    loaded_model = pickle.load(open(model_path, 'rb'))
    st.success("Model loaded successfully! Ready for predictions.")
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.warning("Please verify the model file's integrity and ensure it was saved correctly.")
    st.stop()

# --- Prediction Function ---
def predict_quality(input_features):
    # The loaded_model is a Pipeline (scaler + classifier).
    # It expects a DataFrame-like input for consistency.
    feature_names = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    input_df = pd.DataFrame([input_features], columns=feature_names)

    # The pipeline's scaler will automatically scale these features before prediction
    prediction = loaded_model.predict(input_df)[0]
    return prediction

# --- Streamlit App Main Function ---
def main():
    st.set_page_config(page_title="Wine Quality Predictor", layout="centered")

    st.title("🍷 Wine Quality Prediction Web App")
    st.markdown("Enter the wine's chemical properties to predict its quality.")

    st.header("Wine Properties")

    # Input fields for wine features
    fixed_acidity = st.slider('Fixed Acidity', min_value=4.0, max_value=16.0, value=7.8, step=0.1)
    volatile_acidity = st.slider('Volatile Acidity', min_value=0.1, max_value=1.5, value=0.580, step=0.001, format="%.3f")
    citric_acid = st.slider('Citric Acid', min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    residual_sugar = st.slider('Residual Sugar', min_value=0.5, max_value=15.0, value=2.0, step=0.1)
    chlorides = st.slider('Chlorides', min_value=0.01, max_value=0.6, value=0.073, step=0.001, format="%.3f")
    free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', min_value=1.0, max_value=70.0, value=9.0, step=1.0)
    total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', min_value=6.0, max_value=300.0, value=18.0, step=1.0)
    density = st.slider('Density', min_value=0.99, max_value=1.01, value=0.9968, step=0.0001, format="%.4f")
    pH = st.slider('pH', min_value=2.7, max_value=4.0, value=3.36, step=0.01)
    sulphates = st.slider('Sulphates', min_value=0.3, max_value=2.0, value=0.57, step=0.01)
    alcohol = st.slider('Alcohol', min_value=8.0, max_value=15.0, value=9.5, step=0.1)

    # Button to trigger prediction
    if st.button('Predict Wine Quality'):
        # Prepare input data as a list
        input_data = [
            fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
            pH, sulphates, alcohol
        ]

        try:
            quality_prediction = predict_quality(input_data)

            if quality_prediction == 0:
                st.error("📉 This is a **Bad Quality Wine**")
            else:
                st.success("🥂 This is a **Good Quality Wine**")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.warning("Please check the input values and model compatibility.")

    st.markdown("---")
    st.caption("Powered by Machine Learning")

# --- Run the Streamlit App ---
if __name__ == '__main__':
    main()
