import streamlit as st
import pickle
import pandas as pd
import os
from pathlib import Path

st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("📉 Comcast Churn Prediction App")

# Load model and preprocessor from saved files
@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer"""
    try:
        # Load the model
        with open("Data/processed/model.pkl", "rb") as f:
            model = pickle.load(f)
        
        # Load the TF-IDF vectorizer
        with open("Data/processed/tfidf.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the training pipeline first.")
        st.stop()
        return None, None

model, vectorizer = load_model_and_vectorizer()

st.subheader("Enter Customer Complaint Text")

# Text input for prediction
complaint_text = st.text_area(
    "Customer Complaint",
    placeholder="Enter the customer complaint text here...",
    height=150,
    value="I used to love Comcast. Until all these constant price increases. I pay $190 a month for cable and internet."
)

if st.button("Predict Sentiment"):
    if complaint_text.strip():
        # Preprocess text
        clean_text = complaint_text.lower().strip()
        
        # Transform using TF-IDF vectorizer
        text_features = vectorizer.transform([clean_text])
        
        # Make prediction
        prediction = model.predict(text_features)[0]
        probability = model.predict_proba(text_features)[0]
        
        st.write("---")
        st.write("### Prediction Result:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("**Sentiment:** POSITIVE (Rating ≥ 4)")
                st.metric("Confidence", f"{probability[1]:.1%}")
            else:
                st.error("**Sentiment:** NEGATIVE (Rating < 4)")
                st.metric("Confidence", f"{probability[0]:.1%}")
        
        with col2:
            st.write("**Probability Distribution:**")
            st.write(f"- Negative: {probability[0]:.1%}")
            st.write(f"- Positive: {probability[1]:.1%}")
        
        st.write("---")
        st.info("This model predicts customer sentiment based on complaint text. Positive sentiment indicates a rating of 4 or higher.")
    else:
        st.warning("Please enter some complaint text.")
