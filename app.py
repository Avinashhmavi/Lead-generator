import streamlit as st
import pandas as pd
import json
import time
import os
from xgboost import XGBRegressor
from openai import OpenAI

# --- Configuration ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 3000
BATCH_SIZE = 3
RETRY_DELAY = 15
MAX_RETRIES = 3

# --- Secrets Management ---
def get_secrets():
    return {
        'groq_api_key': st.secrets.get("GROQ_API_KEY"),
        'model_path': st.secrets.get("MODEL_PATH", "xgb_model.json")
    }

# --- Model Handling ---
def load_sales_model():
    secrets = get_secrets()
    try:
        if not os.path.exists(secrets['model_path']):
            raise FileNotFoundError(f"Model file not found at {secrets['model_path']}")
        
        model = XGBRegressor()
        model.load_model(secrets['model_path'])
        return model
    except Exception as e:
        st.error(f"Model Error: {str(e)}")
        return None

# --- Groq API Client ---
def get_groq_client():
    secrets = get_secrets()
    if not secrets['groq_api_key']:
        st.error("Missing Groq API key in secrets")
        return None
    return OpenAI(api_key=secrets['groq_api_key'], base_url=GROQ_BASE_URL)

# --- Core Processing Functions ---
def safe_groq_request(client, messages):
    for attempt in range(MAX_RETRIES):
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
                time.sleep(RETRY_DELAY * (attempt + 1))
                continue
            st.error(f"API Error: {str(e)}")
            return None
    return None

def process_leads(df):
    try:
        return [df[i:i+BATCH_SIZE].to_dict('records') 
               for i in range(0, len(df), BATCH_SIZE]
    except Exception as e:
        st.error(f"Data Processing Error: {str(e)}")
        return []

def generate_analysis_prompt(batch):
    return f"""Analyze these marketing leads. For each entry:
- Calculate lead score (0-100)
- Identify primary strength
- Suggest engagement strategy
- Estimate conversion potential (Low/Medium/High)

Input Data:
{json.dumps(batch, indent=2)}

Format response as markdown table with columns:
| Name | Email | Score | Strength | Strategy | Potential |"""

# --- UI Components ---
def sidebar_status():
    with st.sidebar:
        st.title("🔐 System Status")
        
        # Secrets status
        secrets = get_secrets()
        status_text = [
            f"**Groq API**: {'✅ Configured' if secrets['groq_api_key'] else '❌ Missing'}",
            f"**Model File**: {'✅ Found' if os.path.exists(secrets['model_path']) else '❌ Missing'}"
        ]
        st.markdown("\n".join(status_text))
        
        st.markdown("---")
        st.markdown("""
        **Configuration Guide**
        1. Create `.streamlit/secrets.toml`
        2. Add:
           ```toml
           GROQ_API_KEY = "your_api_key_here"
           MODEL_PATH = "xgb_model.json"
           ```
        3. Place model file in specified path
        """)

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="Business Analytics Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load components
    sidebar_status()
    sales_model = load_sales_model()
    groq_client = get_groq_client()
    
    # Main interface
    tab1, tab2 = st.tabs(["💰 Sales Predictor", "📈 Lead Analyzer"])
    
    # Sales Prediction Tab
    with tab1:
        st.header("Sales Prediction Engine")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            tv = st.number_input("TV Budget ($)", min_value=0.0, value=1000.0, step=100.0)
            radio = st.number_input("Radio Budget ($)", min_value=0.0, value=500.0, step=50.0)
        
        with col2:
            newspaper = st.number_input("Newspaper Budget ($)", min_value=0.0, value=200.0, step=20.0)
            if st.button("Predict Sales", type="primary"):
                if sales_model:
                    try:
                        prediction = sales_model.predict([[tv, radio, newspaper]])
                        st.success(f"**Projected Sales**: ${prediction[0]:.2f}")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                else:
                    st.error("Sales prediction unavailable - check model configuration")

    # Lead Analysis Tab
    with tab2:
        st.header("AI-Powered Lead Analysis")
        sub1, sub2 = st.tabs(["🔍 Analyze Leads", "📧 Generate Emails"])
        
        with sub1:
            uploaded_file = st.file_uploader("Upload leads (CSV)", type=["csv"])
            if uploaded_file and groq_client:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df)} leads")
                    
                    if st.button("Analyze Leads", type="primary"):
                        with st.spinner("Processing..."):
                            results = []
                            batches = process_leads(df)
                            
                            for batch in batches:
                                prompt = generate_analysis_prompt(batch)
                                response = safe_groq_request(groq_client, [
                                    {"role": "system", "content": "Expert marketing analyst"},
                                    {"role": "user", "content": prompt}
                                ])
                                
                                if response:
                                    results.append(response)
                                    time.sleep(1)  # Rate limit buffer
                            
                            if results:
                                st.session_state.analysis_results = "\n\n".join(results)
                                st.subheader("Analysis Results")
                                st.markdown(st.session_state.analysis_results)
                                st.download_button(
                                    label="Download Report",
                                    data=st.session_state.analysis_results,
                                    file_name="lead_analysis.md"
                                )
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")
        
        with sub2:
            if 'analysis_results' in st.session_state:
                if st.button("Generate Emails", type="primary"):
                    with st.spinner("Creating personalized emails..."):
                        email_response = safe_groq_request(groq_client, [
                            {"role": "system", "content": "Expert email copywriter"},
                            {"role": "user", "content": f"Generate emails based on:\n{st.session_state.analysis_results}"}
                        ])
                        
                        if email_response:
                            st.subheader("Generated Emails")
                            st.markdown(email_response)
                            st.download_button(
                                label="Download Emails",
                                data=email_response,
                                file_name="personalized_emails.md"
                            )

if __name__ == "__main__":
    main()