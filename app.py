import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from openai import OpenAI
import json
import time
import os

# --- Configuration ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 3000
BATCH_SIZE = 3  # Reduced for reliability
RETRY_DELAY = 15  # Increased for safety
MAX_RETRIES = 3

# --- XGBoost Model Handling ---
def load_xgb_model():
    try:
        model = XGBRegressor()
        model.load_model("xgb_model.json")  # Using native XGBoost format
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# --- Groq API Functions ---
def get_groq_client(api_key):
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

def safe_groq_call(client, messages):
    for _ in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                time.sleep(RETRY_DELAY)
                continue
            st.error(f"API Error: {str(e)}")
            return None
    return None

# --- Core Processing Functions ---
def process_leads_safely(df):
    try:
        batches = [df[i:i+BATCH_SIZE].to_dict('records') 
                 for i in range(0, len(df), BATCH_SIZE)]
        return batches
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return []

def generate_analysis_prompt(batch):
    return f"""Analyze these marketing leads. For each lead, provide:
    - Lead Score (0-100)
    - Primary Strength
    - Recommended Action
    - Confidence Level (High/Medium/Low)

    Input Data:
    {json.dumps(batch, indent=2)}

    Format response as markdown table with columns:
    | Name | Email | Score | Strength | Action | Confidence |"""

def parse_analysis_response(response):
    try:
        lines = [line.split('|') for line in response.split('\n') if '|' in line]
        headers = [h.strip() for h in lines[0] if h]
        return [dict(zip(headers, [cell.strip() for cell in row])) 
               for row in lines[2:]]  # Skip header separator
    except Exception as e:
        st.error(f"Analysis parsing failed: {str(e)}")
        return []

# --- UI Configuration ---
def configure_sidebar():
    with st.sidebar:
        st.title("📈 Business Analytics Suite")
        st.markdown("---")
        st.subheader("Key Features")
        st.markdown("""
        - 🚀 AI-Powered Sales Predictions
        - 📊 Intelligent Lead Scoring
        - 📧 Personalized Email Generation
        - 🔒 Secure Cloud Processing
        """)
        
        if 'groq' not in st.secrets:
            st.error("Groq API key missing!")
            st.markdown("Add to `.streamlit/secrets.toml`:")
            st.code("[groq]\napi_key = 'your-key-here'")

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Business Analytics Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load components
    configure_sidebar()
    xgb_model = load_xgb_model()
    
    # Main tabs
    tab1, tab2 = st.tabs(["💰 Sales Predictor", "📈 Lead Analyzer"])
    
    # Sales Prediction Tab
    with tab1:
        st.header("Sales Prediction Engine")
        col1, col2 = st.columns(2)
        
        with col1:
            tv = st.number_input("TV Budget ($)", min_value=0.0, value=1000.0)
            radio = st.number_input("Radio Budget ($)", min_value=0.0, value=500.0)
        
        with col2:
            newspaper = st.number_input("Newspaper Budget ($)", min_value=0.0, value=200.0)
            if st.button("Predict Sales", type="primary"):
                if xgb_model:
                    try:
                        prediction = xgb_model.predict([[tv, radio, newspaper]])
                        st.success(f"Projected Sales: ${prediction[0]:.2f}")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                else:
                    st.error("Model not available")

    # Lead Analysis Tab
    with tab2:
        st.header("AI Lead Analysis")
        sub1, sub2 = st.tabs(["🔍 Analyze Leads", "📧 Generate Emails"])
        
        with sub1:
            uploaded_file = st.file_uploader("Upload Leads CSV", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} leads")
                    
                    if st.button("Analyze Leads", type="primary"):
                        if 'groq' not in st.secrets:
                            st.error("Groq API key required")
                            return
                            
                        client = get_groq_client(st.secrets.groq.api_key)
                        batches = process_leads_safely(df)
                        
                        with st.spinner("Analyzing..."):
                            results = []
                            for batch in batches:
                                prompt = generate_analysis_prompt(batch)
                                response = safe_groq_call(client, [
                                    {"role": "system", "content": "Expert lead analyst"},
                                    {"role": "user", "content": prompt}
                                ])
                                if response:
                                    results.extend(parse_analysis_response(response))
                            
                            if results:
                                st.subheader("Analysis Results")
                                st.write(pd.DataFrame(results))
                                st.download_button(
                                    label="Download Results",
                                    data=pd.DataFrame(results).to_csv(index=False),
                                    file_name="lead_analysis.csv"
                                )
                except Exception as e:
                    st.error(f"File error: {str(e)}")
        
        with sub2:
            if 'analysis_results' in st.session_state:
                emails = generate_personalized_emails(st.session_state.analysis_results)
                st.write(emails)

if __name__ == "__main__":
    main()