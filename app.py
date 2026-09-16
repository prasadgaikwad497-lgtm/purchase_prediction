import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="Gradient Boosting Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    
""", unsafe_allow_html=True)

# Load Model Function
@st.cache_resource
def load_model():
    model_path = "gradient.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in working directory.")
        return None
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        return None

model = load_model()

# Header Section
st.markdown('
