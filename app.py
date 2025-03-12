import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
import pickle
from openai import OpenAI
import json
import time

# Load the trained XGBoost model
with open('xgb_model.pkl', 'rb') as file:
    xgb_model = pickle.load(file)

# Groq Configuration
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 3000
BATCH_SIZE = 5
RETRY_DELAY = 10
MAX_RETRIES = 3

# --- Groq API Functions ---
def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "API test"}],
            max_tokens=5,
        )
        return True
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        return False

def get_groq_response(messages, api_key, retry_count=0):
    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content
    except Exception as e:
        if "rate_limit_exceeded" in str(e) and retry_count < MAX_RETRIES:
            st.warning(f"Rate limit exceeded - waiting {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
            return get_groq_response(messages, api_key, retry_count + 1)
        st.error(f"Groq API Error: {str(e)}")
        return None

# --- Lead Processing Functions ---
def process_leads(df):
    leads_data = df.to_dict("records")
    batches = [leads_data[i:i + BATCH_SIZE] 
              for i in range(0, len(leads_data), BATCH_SIZE)]

    prompts = []
    for batch in batches:
        formatted_leads = json.dumps(batch, indent=2)
        prompt = f"""Analyze these {len(batch)} leads using:
        - Channel Presence (0-50)
        - Professional Experience (0-50)
        - Career Motivation (0-30)
        - Geography (0-30)

        Output table with columns:
        | Full Name | Preferred Name | Email | Lead Score | Reason | LinkedIn | Motivation |

        Lead Data:
        {formatted_leads}"""
        prompts.append(prompt)
    return prompts

def process_in_batches(prompts, api_key):
    results = []
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": "Expert lead scoring analyst."},
            {"role": "user", "content": prompt}
        ]
        response = get_groq_response(messages, api_key)
        if response:
            results.append(response)
            if i < len(prompts) - 1:
                time.sleep(1)
    return "\n\n".join(results)

def parse_lead_data(input_text):
    lines = input_text.strip().split('\n')
    leads = []
    columns = ['Full Name', 'Preferred Name', 'Email', 'Lead Score', 
              'Reason', 'LinkedIn', 'Motivation']
    for line in lines:
        parts = line.split('\t') if '\t' in line else line.split(',')
        if len(parts) == len(columns):
            lead_dict = dict(zip(columns, [p.strip() for p in parts]))
            leads.append(lead_dict)
    return leads

def generate_personalized_emails(leads_data, api_key):
    messages = [
        {
            "role": "system",
            "content": "Expert email writer for professional courses. Create: "
                       "- Personalized emails using provided lead data "
                       "- Focus on individual motivations "
                       "- Professional but friendly tone "
                       "- Include email address and preferred name"
        },
        {
            "role": "user",
            "content": f"Generate emails for these leads:\n"
                       f"{json.dumps(leads_data, indent=2)}\n"
                       f"Close with warm regards from Karan"
        }
    ]
    response = get_groq_response(messages, api_key)
    return response if response else "Error generating emails"

# --- UI Components ---
def show_features():
    st.markdown("""
    ## 🌟 Lead Analysis Features
    - **🚀 Real-time Lead Scoring** with multi-dimensional analysis
    - **📧 Personalized Email Generation** based on lead motivations
    - **📊 Batch Processing** with automatic rate limiting
    - **📥 Easy CSV/Excel Import** for lead data
    - **📤 Downloadable Reports** in markdown format
    """)

# --- Main App ---
def main():
    st.set_page_config(
        page_title="Business Analytics Suite",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .main {
            max-width: 1200px;
            margin: auto;
            padding: 20px;
        }
        h1 {
            color: #0066cc;
        }
        .btn-primary {
            background-color: #0066cc;
            color: #ffffff;
        }
        .btn-primary:hover {
            background-color: #0050a5;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 0 25px;
            background-color: #F0F2F6;
            border-radius: 5px 5px 0px 0px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar Configuration
    with st.sidebar:
        st.title("🚀 Analytics Tools")
        st.markdown("---")
        show_features()
        if 'groq' not in st.secrets or not st.secrets.groq.get('api_key'):
            st.error("⛔ Groq API Key Missing")
            st.markdown("1. Go to [Groq Console](https://console.groq.com/keys)")
            st.markdown("2. Create an API key")
            st.markdown("3. Add to `.streamlit/secrets.toml`:")
            st.code("""
            [groq]
            api_key = "your_api_key_here"
            """)
        else:
            st.success("✅ Groq API Key Configured")
        st.markdown("---")
        st.warning(f"API Limits:\n- Max {BATCH_SIZE} leads/batch\n- {MAX_TOKENS} tokens/request")

    # Main Tabs
    tab1, tab2 = st.tabs(["📈 Sales Predictor", "📊 Lead Analyzer"])

    # Sales Prediction Tab
    with tab1:
        st.header("Sales Prediction Engine")
        tv = st.text_input("TV Ad Spend ($)", "")
        radio = st.text_input("Radio Ad Spend ($)", "")
        newspaper = st.text_input("Newspaper Ad Spend ($)", "")

        if st.button("Predict Sales", type="primary"):
            try:
                prediction = xgb_model.predict([[float(tv), float(radio), float(newspaper)]])
                st.success(f"Predicted Sales: ${prediction[0]:.2f}")
            except ValueError:
                st.error("Please enter valid numerical values")

    # Lead Analysis Tab
    with tab2:
        st.header("Lead Analysis Platform")
        subtab1, subtab2 = st.tabs(["📋 Lead Scoring", "✉️ Email Generator"])

        with subtab1:
            uploaded_file = st.file_uploader(
                "Upload Lead Data (CSV/Excel)", 
                type=["csv", "xlsx"],
                help="File should contain lead contact info and professional details"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.type == "text/csv" else pd.read_excel(uploaded_file)
                    st.success(f"Loaded {len(df)} leads")
                    
                    if st.button("Start Analysis", type="primary"):
                        if 'groq' not in st.secrets or not st.secrets.groq.get('api_key'):
                            st.error("Groq API key required for analysis")
                            return
                        
                        with st.spinner("Analyzing leads..."):
                            prompts = process_leads(df)
                            response = process_in_batches(prompts, st.secrets.groq.api_key)
                            
                            if response:
                                st.subheader("Analysis Results")
                                st.markdown(response)
                                st.download_button(
                                    label="Download Report",
                                    data=response,
                                    file_name="lead_scores.md",
                                    mime="text/markdown"
                                )
                except Exception as e:
                    st.error(f"File processing error: {str(e)}")

        with subtab2:
            lead_input = st.text_area(
                "Paste Lead Analysis Results", 
                height=300,
                placeholder="Paste tab/comma separated data with columns: Full Name, Preferred Name, Email, Lead Score, Reason, LinkedIn, Motivation"
            )
            
            if st.button("Generate Emails", type="primary"):
                if not lead_input.strip():
                    st.warning("Please input lead data first")
                    return
                
                if 'groq' not in st.secrets or not st.secrets.groq.get('api_key'):
                    st.error("Groq API key required for email generation")
                    return
                
                with st.spinner("Generating personalized emails..."):
                    leads_data = parse_lead_data(lead_input)
                    
                    if leads_data:
                        emails = generate_personalized_emails(leads_data, st.secrets.groq.api_key)
                        
                        if emails:
                            st.subheader("Generated Emails")
                            st.markdown(emails)
                            st.download_button(
                                label="Download Emails",
                                data=emails,
                                file_name="personalized_emails.md",
                                mime="text/markdown"
                            )
                    else:
                        st.error("Invalid input format")

if __name__ == "__main__":
    main()