import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import time

# --- Configuration ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 3000
BATCH_SIZE = 5
RETRY_DELAY = 10
MAX_RETRIES = 3

# --- API Functions ---
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

# --- Core Processing ---
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
    ## 🌟 Key Features
    - **🚀 Real-time Lead Scoring** with multi-dimensional analysis
    - **📧 Personalized Email Generation** based on lead motivations
    - **📊 Batch Processing** with automatic rate limiting
    - **📥 Easy CSV/Excel Import** for lead data
    - **📤 Downloadable Reports** in markdown format
    """)

def main():
    st.set_page_config(
        page_title="Lead Scoring Dashboard", 
        page_icon="📈", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- Authentication Check ---
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = None

    if not st.secrets.get("GROQ_API_KEY"):
        with st.sidebar:
            st.error("⛔ API Key Missing")
            st.markdown("1. Go to [Groq API](https://console.groq.com/keys)")
            st.markdown("2. Create an API key")
            st.markdown("3. Add to `.streamlit/secrets.toml`:")
            st.code("""
            [groq]
            api_key = "your_api_key_here"
            """)
        return

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://groq.com/favicon.ico", width=60)
        st.title("🚀 Groq Lead Analyzer")
        st.markdown("---")
        show_features()
        st.markdown("---")
        st.warning(f"API Limits:\n- Max {BATCH_SIZE} leads/batch\n- {MAX_TOKENS} tokens/request")

    # --- Main Content ---
    st.header("📈 Lead Intelligence Platform")
    tab1, tab2 = st.tabs(["📊 Lead Scoring", "📧 Email Generation"])

    # --- Lead Scoring Tab ---
    with tab1:
        st.subheader("Upload Leads for Analysis")
        uploaded_file = st.file_uploader(
            "Upload CSV/Excel File", 
            type=["csv", "xlsx"],
            help="File should contain lead data with standard fields"
        )

        if uploaded_file:
            try:
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(f"Loaded {len(df)} leads")
                
                if st.button("🚀 Start Analysis", type="primary"):
                    with st.spinner(f"Processing {len(df)} leads..."):
                        prompts = process_leads(df)
                        response = process_in_batches(prompts, st.secrets.groq.api_key)

                        if response:
                            st.subheader("📊 Analysis Results")
                            st.markdown(response)
                            
                            st.download_button(
                                label="📥 Download Report",
                                data=response,
                                file_name="lead_scores.md",
                                mime="text/markdown"
                            )
            except Exception as e:
                st.error(f"File processing error: {str(e)}")

    # --- Email Generation Tab ---
    with tab2:
        st.subheader("Generate Personalized Emails")
        lead_input = st.text_area(
            "Paste Lead Data", 
            height=300,
            placeholder="Full Name, Preferred Name, Email, Lead Score, Reason, LinkedIn, Motivation"
        )

        if st.button("📧 Generate Emails", type="primary"):
            if lead_input.strip():
                with st.spinner("Creating emails..."):
                    leads_data = parse_lead_data(lead_input)
                    
                    if leads_data:
                        emails = generate_personalized_emails(
                            leads_data, 
                            st.secrets.groq.api_key
                        )
                        
                        if emails:
                            st.subheader("📩 Generated Emails")
                            st.markdown(emails)
                            
                            st.download_button(
                                label="📥 Download Emails",
                                data=emails,
                                file_name="personalized_emails.md",
                                mime="text/markdown"
                            )
                    else:
                        st.error("Invalid input format")
            else:
                st.warning("Please input lead data first")

if __name__ == "__main__":
    main()
