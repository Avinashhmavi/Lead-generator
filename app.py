import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from openai import OpenAI
import json
import time
import joblib
import os

# --- Configuration ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "mixtral-8x7b-32768"
SALES_MODEL_PATH = "sales_model.joblib"

# --- Sales Prediction Model ---
def load_or_create_sales_model():
    try:
        if os.path.exists(SALES_MODEL_PATH):
            return joblib.load(SALES_MODEL_PATH)
        
        # Create sample training data
        np.random.seed(42)
        data = {
            'TV': np.random.uniform(100, 1000, 1000),
            'Radio': np.random.uniform(50, 500, 1000),
            'Newspaper': np.random.uniform(20, 200, 1000),
            'Sales': np.random.normal(100, 20, 1000) + 0.5*np.random.uniform(100, 1000, 1000)
        }
        df = pd.DataFrame(data)
        
        X = df[['TV', 'Radio', 'Newspaper']]
        y = df['Sales']
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        joblib.dump(model, SALES_MODEL_PATH)
        return model
        
    except Exception as e:
        st.error(f"Model Error: {str(e)}")
        return None

# --- Groq API Functions ---
def get_groq_client():
    try:
        return OpenAI(
            api_key=st.secrets["GROQ_API_KEY"],
            base_url=GROQ_BASE_URL
        )
    except Exception as e:
        st.error(f"API Configuration Error: {str(e)}")
        return None

def safe_groq_request(client, prompt):
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# --- Lead Processing ---
def analyze_leads(df, client):
    try:
        batches = [df[i:i+5] for i in range(0, len(df), 5)]
        results = []
        
        for batch in batches:
            prompt = f"""Analyze these marketing leads:
            {batch.to_markdown()}
            
            Score each lead (0-100) considering:
            - Demographic match
            - Engagement history
            - Company size
            - Industry relevance
            
            Format response as markdown table with columns:
            | Name | Email | Lead Score | Top Factor |"""
            
            response = safe_groq_request(client, prompt)
            if response:
                results.append(response)
                time.sleep(1)  # Rate limit buffer
                
        return "\n\n".join(results)
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        return None

def generate_emails(analysis, client):
    try:
        prompt = f"""Generate personalized sales emails based on this lead analysis:
        {analysis}
        
        Requirements:
        - Use recipient's name
        - Reference key scoring factors
        - Include clear call-to-action
        - Keep under 150 words
        - Friendly but professional tone"""
        
        return safe_groq_request(client, prompt)
    except Exception as e:
        st.error(f"Email Generation Error: {str(e)}")
        return None

# --- Main App ---
def main():
    st.set_page_config(
        page_title="AI Sales Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load components
    sales_model = load_or_create_sales_model()
    groq_client = get_groq_client()
    
    # Sidebar
    with st.sidebar:
        st.title("AI Sales Assistant")
        st.markdown("""
        **Features:**
        - 📈 Sales Prediction (Random Forest)
        - 📊 Lead Scoring (AI Analysis)
        - 📧 Personalized Email Generation
        """)
        
        if st.button("🔄 Retrain Sales Model"):
            if os.path.exists(SALES_MODEL_PATH):
                os.remove(SALES_MODEL_PATH)
            sales_model = load_or_create_sales_model()
            st.rerun()
    
    # Main Tabs
    tab1, tab2 = st.tabs(["💰 Sales Predictor", "📈 Lead Manager"])
    
    with tab1:
        st.header("Sales Prediction Engine")
        col1, col2 = st.columns(2)
        
        with col1:
            tv = st.number_input("TV Ad Budget ($)", min_value=0, value=500)
            radio = st.number_input("Radio Ad Budget ($)", min_value=0, value=200)
        
        with col2:
            newspaper = st.number_input("Newspaper Ad Budget ($)", min_value=0, value=100)
            if st.button("Predict Sales"):
                if sales_model:
                    try:
                        prediction = sales_model.predict([[tv, radio, newspaper]])
                        st.success(f"Predicted Sales: ${prediction[0]:.2f}")
                    except Exception as e:
                        st.error(f"Prediction Error: {str(e)}")
                else:
                    st.error("Model not loaded")

    with tab2:
        st.header("Lead Management System")
        sub1, sub2 = st.tabs(["🔍 Analyze Leads", "📧 Generate Emails"])
        
        with sub1:
            uploaded_file = st.file_uploader("Upload Leads CSV", type=["csv"])
            if uploaded_file and groq_client:
                try:
                    df = pd.read_csv(uploaded_file)
                    required_cols = ['Name', 'Email', 'Company', 'Industry']
                    
                    if not all(col in df.columns for col in required_cols):
                        st.error("Missing required columns in CSV")
                        return
                    
                    st.success(f"Loaded {len(df)} leads")
                    
                    if st.button("Analyze Leads"):
                        with st.spinner("Analyzing..."):
                            analysis = analyze_leads(df, groq_client)
                            if analysis:
                                st.session_state.lead_analysis = analysis
                                st.markdown(analysis)
                
                except Exception as e:
                    st.error(f"File Error: {str(e)}")
        
        with sub2:
            if 'lead_analysis' in st.session_state:
                if st.button("Generate Emails"):
                    with st.spinner("Creating emails..."):
                        emails = generate_emails(st.session_state.lead_analysis, groq_client)
                        if emails:
                            st.session_state.generated_emails = emails
                            st.markdown(emails)
                            
                            st.download_button(
                                "Download Emails",
                                emails,
                                file_name="personalized_emails.md"
                            )

if __name__ == "__main__":
    main()