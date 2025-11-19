import streamlit as st
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Comcast Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-positive {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .prediction-negative {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Environment variables with defaults
MODEL_PATH = os.getenv("MODEL_PATH", "Data/processed/model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "Data/processed/tfidf.pkl")
DATA_PATH = os.getenv("DATA_PATH", "Data/processed/reviews_raw.parquet")

@st.cache_resource
def load_model_and_vectorizer():
    """Load the trained model and TF-IDF vectorizer"""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        
        logger.info("Model and vectorizer loaded successfully")
        return model, vectorizer
    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        st.error(f"Model files not found. Please run the training pipeline first.")
        st.stop()
        return None, None
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Error loading model: {str(e)}")
        st.stop()
        return None, None

@st.cache_data
def load_historical_data():
    """Load historical data for analytics"""
    try:
        if Path(DATA_PATH).exists():
            df = pd.read_parquet(DATA_PATH)
            logger.info(f"Loaded {len(df)} historical reviews")
            return df
        return None
    except Exception as e:
        logger.warning(f"Could not load historical data: {e}")
        return None

def predict_sentiment(text, model, vectorizer):
    """Make sentiment prediction"""
    try:
        clean_text = text.lower().strip()
        text_features = vectorizer.transform([clean_text])
        prediction = model.predict(text_features)[0]
        probability = model.predict_proba(text_features)[0]
        
        logger.info(f"Prediction made: {prediction}, confidence: {max(probability):.2f}")
        return prediction, probability
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

def batch_predict(df, model, vectorizer):
    """Batch prediction for multiple texts"""
    try:
        df['clean_text'] = df['text'].str.lower().str.strip()
        X = vectorizer.transform(df['clean_text'])
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        df['predicted_sentiment'] = predictions
        df['confidence'] = probabilities.max(axis=1)
        df['negative_prob'] = probabilities[:, 0]
        df['positive_prob'] = probabilities[:, 1]
        
        return df
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise

# Load model and data
model, vectorizer = load_model_and_vectorizer()
historical_data = load_historical_data()

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/Comcast_Logo.svg/320px-Comcast_Logo.svg.png", width=200)
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🔮 Single Prediction", "📊 Batch Analysis", "📈 Model Analytics", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### Model Info")
    st.info(f"""
    **Model Type:** Logistic Regression  
    **Features:** TF-IDF (5000 max)  
    **Classes:** Positive (≥4★) / Negative (<4★)
    """)
    
    if historical_data is not None:
        st.success(f"📦 {len(historical_data):,} reviews loaded")

# Main content
if page == "🔮 Single Prediction":
    st.markdown('<h1 class="main-header">📉 Comcast Sentiment Analyzer</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Customer Review")
        
        # Sample reviews
        sample_reviews = {
            "Select a sample...": "",
            "Positive Review": "Comcast has been great! The technician was professional and the internet speed is exactly as promised. Very happy with the service.",
            "Negative Review": "Worst customer service ever. Been on hold for 2 hours just to cancel my service. Overpriced and unreliable internet.",
            "Mixed Review": "The internet speed is good but the customer support needs improvement. Billing issues every month."
        }
        
        selected_sample = st.selectbox("Try a sample review:", list(sample_reviews.keys()))
        
        default_text = sample_reviews[selected_sample] if selected_sample != "Select a sample..." else ""
        
        complaint_text = st.text_area(
            "Customer Review Text",
            value=default_text,
            height=150,
            placeholder="Enter the customer review text here...",
            help="Enter any customer review or complaint text to analyze sentiment"
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            predict_button = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🔄 Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if predict_button and complaint_text.strip():
            with st.spinner("Analyzing sentiment..."):
                try:
                    prediction, probability = predict_sentiment(complaint_text, model, vectorizer)
                    
                    st.markdown("---")
                    
                    # Results
                    if prediction == 1:
                        st.markdown('<div class="prediction-positive">', unsafe_allow_html=True)
                        st.success("### ✅ POSITIVE SENTIMENT")
                        st.markdown("This review indicates customer satisfaction (Rating ≥ 4 stars)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="prediction-negative">', unsafe_allow_html=True)
                        st.error("### ⚠️ NEGATIVE SENTIMENT")
                        st.markdown("This review indicates customer dissatisfaction (Rating < 4 stars)")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Confidence metrics
                    st.markdown("### 📊 Confidence Scores")
                    col_m1, col_m2, col_m3 = st.columns(3)
                    
                    with col_m1:
                        st.metric("Overall Confidence", f"{max(probability):.1%}")
                    with col_m2:
                        st.metric("Negative Probability", f"{probability[0]:.1%}")
                    with col_m3:
                        st.metric("Positive Probability", f"{probability[1]:.1%}")
                    
                    # Probability chart
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Negative', 'Positive'],
                            y=[probability[0], probability[1]],
                            marker_color=['#dc3545', '#28a745'],
                            text=[f"{probability[0]:.1%}", f"{probability[1]:.1%}"],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Sentiment Probability Distribution",
                        yaxis_title="Probability",
                        showlegend=False,
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f" Error during prediction: {str(e)}")
                    logger.error(f"Prediction error: {e}")
        
        elif predict_button:
            st.warning("⚠️ Please enter some review text to analyze")
    
    with col2:
        st.subheader("💡 Quick Tips")
        st.info("""
        **How to use:**
        1. Enter or select a customer review
        2. Click "Analyze Sentiment"
        3. View the results and confidence scores
        
        **What the model predicts:**
        - **Positive:** Customer is likely satisfied (4-5 stars)
        - **Negative:** Customer is likely unsatisfied (1-3 stars)
        """)
        
        st.markdown("### Analysis Guidelines")
        st.success("""
        - Longer reviews provide better accuracy
        - The model analyzes overall sentiment tone
        - Confidence >80% indicates high reliability
        """)

elif page == "Batch Analysis":
    st.markdown('<h1 class="main-header">📊 Batch Sentiment Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Upload a CSV file with a `text` column containing customer reviews for batch prediction.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV file must contain a 'text' column with review text"
    )
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            if 'text' not in df_upload.columns:
                st.error("CSV file must contain a 'text' column")
            else:
                st.success(f"Loaded {len(df_upload)} reviews")
                
                st.subheader("Preview")
                st.dataframe(df_upload.head(10), use_container_width=True)
                
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner("Processing batch predictions..."):
                        results_df = batch_predict(df_upload, model, vectorizer)
                        
                        st.success("Batch prediction completed!")
                        
                        # Summary metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Reviews", len(results_df))
                        with col2:
                            positive_count = (results_df['predicted_sentiment'] == 1).sum()
                            st.metric("Positive", positive_count)
                        with col3:
                            negative_count = (results_df['predicted_sentiment'] == 0).sum()
                            st.metric("Negative", negative_count)
                        with col4:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                        
                        # Visualizations
                        col_v1, col_v2 = st.columns(2)
                        
                        with col_v1:
                            # Sentiment distribution
                            sentiment_counts = results_df['predicted_sentiment'].value_counts()
                            fig_pie = px.pie(
                                values=sentiment_counts.values,
                                names=['Negative', 'Positive'],
                                title="Sentiment Distribution",
                                color_discrete_sequence=['#dc3545', '#28a745']
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with col_v2:
                            # Confidence distribution
                            fig_hist = px.histogram(
                                results_df,
                                x='confidence',
                                title="Confidence Score Distribution",
                                nbins=20
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Results table
                        st.subheader("Detailed Results")
                        results_display = results_df.copy()
                        results_display['sentiment'] = results_display['predicted_sentiment'].map({
                            0: '😞 Negative',
                            1: '😊 Positive'
                        })
                        results_display['confidence'] = results_display['confidence'].apply(lambda x: f"{x:.1%}")
                        
                        st.dataframe(
                            results_display[['text', 'sentiment', 'confidence', 'negative_prob', 'positive_prob']],
                            use_container_width=True
                        )
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name=f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Batch processing error: {e}")

elif page == "📈 Model Analytics":
    st.markdown('<h1 class="main-header">📈 Model Performance Analytics</h1>', unsafe_allow_html=True)
    
    if historical_data is not None:
        # Dataset stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Reviews", f"{len(historical_data):,}")
        with col2:
            avg_rating = historical_data['rating'].mean()
            st.metric("Avg Rating", f"{avg_rating:.2f}⭐")
        with col3:
            positive_pct = (historical_data['rating'] >= 4).sum() / len(historical_data) * 100
            st.metric("Positive %", f"{positive_pct:.1f}%")
        with col4:
            negative_pct = (historical_data['rating'] < 4).sum() / len(historical_data) * 100
            st.metric("Negative %", f"{negative_pct:.1f}%")
        
        st.markdown("---")
        
        # Charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            # Rating distribution
            fig_rating = px.histogram(
                historical_data,
                x='rating',
                title="Rating Distribution",
                nbins=5,
                color_discrete_sequence=['#1f77b4']
            )
            fig_rating.update_layout(
                xaxis_title="Rating",
                yaxis_title="Count",
                showlegend=False
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        
        with col_c2:
            # Sentiment over time (if date available)
            if 'posted_on' in historical_data.columns:
                historical_data['month'] = pd.to_datetime(historical_data['posted_on']).dt.to_period('M')
                monthly_sentiment = historical_data.groupby('month')['rating'].mean().reset_index()
                monthly_sentiment['month'] = monthly_sentiment['month'].astype(str)
                
                fig_trend = px.line(
                    monthly_sentiment,
                    x='month',
                    y='rating',
                    title="Average Rating Trend Over Time",
                    markers=True
                )
                fig_trend.update_layout(
                    xaxis_title="Month",
                    yaxis_title="Average Rating"
                )
                st.plotly_chart(fig_trend, use_container_width=True)
        
        # Sample reviews
        st.subheader("📝 Sample Reviews by Sentiment")
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("**Negative Reviews (Rating < 4)**")
            negative_samples = historical_data[historical_data['rating'] < 4].sample(min(5, len(historical_data[historical_data['rating'] < 4])))
            for idx, row in negative_samples.iterrows():
                with st.expander(f"⭐ {row['rating']} - {row.get('author', 'Anonymous')}"):
                    st.write(row['text'][:300] + "..." if len(row['text']) > 300 else row['text'])
        
        with col_s2:
            st.markdown("** Positive Reviews (Rating ≥ 4)**")
            positive_samples = historical_data[historical_data['rating'] >= 4].sample(min(5, len(historical_data[historical_data['rating'] >= 4])))
            for idx, row in positive_samples.iterrows():
                with st.expander(f"{row['rating']} - {row.get('author', 'Anonymous')}"):
                    st.write(row['text'][:300] + "..." if len(row['text']) > 300 else row['text'])
    
    else:
        st.warning("📊 Historical data not available for analytics")

else:  # About page
    st.markdown('<h1 class="main-header"> About This Application</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Purpose
    This application analyzes customer sentiment from Comcast reviews using machine learning.
    It helps identify customer satisfaction levels and potential churn risks.
    
    ## 🔧 Technical Stack
    - **Framework:** Streamlit
    - **ML Model:** Logistic Regression with TF-IDF
    - **ML Tracking:** MLflow
    - **Data Processing:** Pandas, Scikit-learn
    - **Visualization:** Plotly
    - **Deployment:** Docker
    
    ## 📊 Model Details
    - **Training Data:** ~5,600 Comcast customer reviews
    - **Features:** TF-IDF vectorization (5,000 max features)
    - **Model Type:** Logistic Regression
    - **Accuracy:** ~98.5%
    - **Classes:** 
      - Positive: Rating ≥ 4 stars
      - Negative: Rating < 4 stars
    
    ## 🚀 Features
    1. **Single Prediction:** Analyze individual customer reviews
    2. **Batch Analysis:** Process multiple reviews at once
    3. **Analytics Dashboard:** View model performance and data insights
    4. **Export Results:** Download predictions as CSV
    
    ## 📝 Usage Instructions
    
    ### Single Prediction
    1. Navigate to "🔮 Single Prediction"
    2. Enter or select a customer review
    3. Click "Analyze Sentiment"
    4. View results and confidence scores
    
    ### Batch Analysis
    1. Navigate to "📊 Batch Analysis"
    2. Upload CSV file with 'text' column
    3. Click "Run Batch Prediction"
    4. Download results
    
    ## 🔒 Production Ready Features
    - ✅ Error handling and logging
    - ✅ Input validation
    - ✅ Model caching for performance
    - ✅ Environment variable configuration
    - ✅ Responsive UI design
    - ✅ Batch processing capability
    - ✅ Export functionality
    
    ## 👨‍💻 Development
    Built with ❤️ using Streamlit and Scikit-learn
    
    **Version:** 2.0  
    **Last Updated:** November 2025
    """)
    
    st.markdown("---")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        st.info("### 📚 Documentation\nAccess full documentation and API reference")
    
    with col_a2:
        st.success("### 🐛 Report Issues\nSubmit bugs and feature requests")
    
    with col_a3:
        st.warning("### 🔄 Version History\nView changelog and updates")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Comcast Sentiment Analyzer v2.0 | Built with Streamlit | © 2025"
    "</div>",
    unsafe_allow_html=True
)
