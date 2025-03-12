import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import time
import csv
from typing import List, Dict, Optional
import re

# --- Custom Exceptions ---
class APIError(Exception):
    pass

class ProcessingError(Exception):
    pass

# --- Configuration ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 3000
BATCH_SIZE = 5
RETRY_DELAY = 10
MAX_RETRIES = 3
REQUIRED_COLUMNS = ['Full Name', 'Email', 'LinkedIn', 'Motivation']

# --- API Functions ---
def validate_api_key(api_key: str) -> bool:
    """Validate the Groq API key."""
    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "API test"}],
            max_tokens=5,
        )
        return True
    except Exception as e:
        raise APIError(f"API validation failed: {str(e)}") from e

def get_groq_response(messages: List[Dict], api_key: str, retry_count: int = 0) -> Optional[str]:
    """Get response from Groq API with error handling."""
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
            time.sleep(RETRY_DELAY)
            return get_groq_response(messages, api_key, retry_count + 1)
        raise APIError(f"API request failed: {str(e)}") from e

# --- Data Processing ---
@st.cache_data(ttl=3600)
def process_leads(df: pd.DataFrame) -> List[str]:
    """Process leads into analysis prompts."""
    leads_data = df.to_dict("records")
    return [leads_data[i:i + BATCH_SIZE] for i in range(0, len(leads_data), BATCH_SIZE)]

def validate_lead_data(df: pd.DataFrame) -> bool:
    """Validate required columns in lead data."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return True

def validate_uploaded_file(file) -> bool:
    """Validate uploaded file type and content."""
    if file.type not in ["text/csv", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        raise ValueError("Unsupported file type. Please upload CSV or Excel file.")
    return True

# --- Security Functions ---
def sanitize_input(text: str) -> str:
    """Basic input sanitization to prevent XSS."""
    return re.sub(r'[<>]', lambda x: '&lt;' if x.group() == '<' else '&gt;', text)

# --- Template Management ---
def load_email_templates() -> Dict:
    """Load different email templates."""
    return {
        'default': {
            'system': "Expert email writer for professional courses.",
            'user': "Create emails using lead data with professional tone."
        },
        'technical': {
            'system': "Technical course specialist focusing on skills development.",
            'user': "Generate emails emphasizing technical skill requirements."
        }
    }

# --- Processing Functions ---
def process_in_batches(prompts: List[str], api_key: str) -> str:
    """Process prompts in batches with progress tracking."""
    results = []
    progress_bar = st.progress(0)
    
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": "Expert lead scoring analyst."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = get_groq_response(messages, api_key)
            if response:
                results.append(response)
                time.sleep(1)  # Rate limit buffer
        except APIError as e:
            st.error(str(e))
            break
            
        progress_bar.progress((i+1)/len(prompts))
    
    progress_bar.empty()
    return "\n\n".join(results)

def parse_lead_data(input_text: str) -> List[Dict]:
    """Parse lead data from text input with sanitization."""
    sanitized = sanitize_input(input_text)
    reader = csv.DictReader(sanitized.splitlines())
    return [row for row in reader]

def generate_personalized_emails(leads_data: List[Dict], api_key: str, template: str = 'default') -> str:
    """Generate emails using selected template."""
    templates = load_email_templates()
    selected = templates.get(template, templates['default'])
    
    messages = [
        {"role": "system", "content": selected['system']},
        {"role": "user", "content": f"{selected['user']}\n{json.dumps(leads_data, indent=2)}"}
    ]
    
    try:
        response = get_groq_response(messages, api_key)
        return response or "Error generating emails"
    except APIError as e:
        return str(e)

# --- UI Components ---
def show_features():
    """Display feature list in sidebar."""
    st.markdown("""
    ## 🌟 Key Features
    - **🚀 Real-time Lead Scoring** with AI analysis
    - **📧 Template-based Email Generation**
    - **🛡️ Input Validation & Sanitization**
    - **📊 Progress Tracking for Batch Processing**
    - **🔒 Secure API Handling**
    """)

def api_health_check(api_key: str):
    """Display API usage information."""
    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        usage = client.usage.retrieve()
        st.sidebar.metric("API Credits Remaining", usage.remaining)
    except Exception:
        st.sidebar.warning("API usage data unavailable")

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="AI Lead Manager Pro",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Authentication ---
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = None

    if not st.secrets.get("GROQ_API_KEY"):
        with st.sidebar:
            st.error("⛔ API Key Missing")
            st.markdown("1. Create API key at [Groq Console](https://console.groq.com/keys)")
            st.markdown("2. Add to `.streamlit/secrets.toml`:")
            st.code("[groq]\napi_key = 'your_key_here'")
        return

    # --- Sidebar ---
    with st.sidebar:
        st.title("🔍 Lead Manager Pro")
        show_features()
        st.markdown("---")
        api_health_check(st.secrets.groq.api_key)
        st.markdown("---")
        st.warning(f"API Limits:\n- Max {BATCH_SIZE} leads/batch\n- {MAX_TOKENS} tokens/request")

    # --- Main Content ---
    st.header("AI-Powered Lead Management Platform")
    tab1, tab2 = st.tabs(["📊 Lead Analysis", "📧 Email Studio"])

    with tab1:
        st.subheader("Lead Scoring & Analysis")
        uploaded_file = st.file_uploader(
            "Upload Lead Data",
            type=["csv", "xlsx"],
            help="Required columns: Full Name, Email, LinkedIn, Motivation"
        )

        if uploaded_file:
            try:
                validate_uploaded_file(uploaded_file)
                
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                validate_lead_data(df)
                st.success(f"✅ Validated {len(df)} leads")

                if st.button("🚀 Start Analysis", type="primary"):
                    with st.spinner(f"Analyzing {len(df)} leads..."):
                        batches = process_leads(df)
                        prompts = [
                            f"""Analyze these {len(batch)} leads using:
                            - Channel Presence (0-50)
                            - Professional Experience (0-50)
                            - Career Motivation (0-30)
                            - Geography (0-30)
                            Data: {json.dumps(batch, indent=2)}
                            Output table with columns:
                            | Full Name | Email | Lead Score | Top Factor |"""
                            for batch in batches
                        ]
                        
                        response = process_in_batches(prompts, st.secrets.groq.api_key)
                        st.session_state.analysis_results = response
                        
                        st.subheader("Analysis Results")
                        st.markdown(response)
                        
                        st.download_button(
                            "📥 Download Report",
                            data=response,
                            file_name="lead_analysis.md",
                            mime="text/markdown"
                        )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    with tab2:
        st.subheader("Email Generation")
        
        # Template Selection
        templates = load_email_templates()
        selected_template = st.selectbox(
            "Email Template",
            options=list(templates.keys()),
            index=0
        )

        # Input Section
        lead_input = st.text_area(
            "Paste Lead Data (CSV Format)",
            height=300,
            placeholder="Full Name,Email,Lead Score,Motivation,..."
        )

        if st.button("📧 Generate Emails", type="primary"):
            if lead_input.strip():
                try:
                    leads_data = parse_lead_data(lead_input)
                    if not leads_data:
                        raise ProcessingError("Invalid input format")
                    
                    with st.spinner("Generating personalized emails..."):
                        emails = generate_personalized_emails(
                            leads_data,
                            st.secrets.groq.api_key,
                            selected_template
                        )
                        
                        st.subheader("Generated Emails")
                        st.markdown(emails)
                        
                        st.download_button(
                            "📥 Download Emails",
                            data=emails,
                            file_name=f"emails_{selected_template}.md",
                            mime="text/markdown"
                        )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please input lead data first")

if __name__ == "__main__":
    main()