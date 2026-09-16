import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Gradient Boosting Classifier App",
    page_icon="🤖",
    layout="centered"
)

# Title & Description
st.title("🤖 Customer Classification App")
st.write(
    "Enter the feature values below to get a real-time prediction from the "
    "trained **GradientBoostingClassifier** model."
)

st.divider()

# Load model safely
@st.cache_resource
def load_model():
    try:
        with open("gradient.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Error: `gradient.pkl` file not found in the project root directory.")
        return None
    except Exception as e:
        st.error(f"Error loading the model file: {e}")
        return None

model = load_model()

if model is not None:
    st.subheader("📋 Input Features")

    # Inputs arrangement
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            label="Age",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
            help="Customer age in years"
        )
        gender = st.selectbox(
            label="Gender",
            options=[0, 1],
            format_func=lambda x: "Female (0)" if x == 0 else "Male (1)",
            help="Select gender code"
        )

    with col2:
        review = st.number_input(
            label="Review Score",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            help="Numeric review rating or score"
        )
        education = st.selectbox(
            label="Education Level",
            options=[0, 1, 2, 3],
            help="Select education level code"
        )

    st.markdown("---")

    # Construct input dataframe maintaining exact feature order
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "review": review,
        "education": education
    }])

    # Input Data Preview
    with st.expander("🔍 View Formatted Model Input Data"):
        st.dataframe(input_data)

    # Predict Button
    if st.button("🚀 Predict", type="primary", use_container_width=True):
        try:
            # Generate Prediction
            prediction = model.predict(input_data)[0]
            
            # Prediction Probabilities if available
            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]

            st.success("Prediction complete!")

            # Output Section
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric(
                    label="Predicted Class",
                    value=f"Class {prediction}"
                )

            with res_col2:
                if probabilities is not None:
                    confidence = probabilities[int(prediction)] * 100
                    st.metric(
                        label="Model Confidence",
                        value=f"{confidence:.2f}%"
                    )

            if probabilities is not None:
                st.subheader("📊 Class Probabilities")
                prob_df = pd.DataFrame({
                    "Class": model.classes_,
                    "Probability": probabilities
                })
                st.bar_chart(prob_df.set_index("Class"))

        except Exception as e:
            st.error(f"Failed to generate prediction. Error: {e}")
