import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import time

# Groq API configuration
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 3000
BATCH_SIZE = 5  # Process 5 leads at a time
RETRY_DELAY = 10  # Seconds to wait on rate limit error
MAX_RETRIES = 3  # Max number of retry attempts

def validate_api_key(api_key):
    """Validate the Groq API key with a lightweight test"""
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
    """Get response from Groq API with retry logic"""
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

def process_leads(df):
    """Convert lead data into optimized batched prompts"""
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
    """Process prompts with rate limit handling"""
    results = []
    for i, prompt in enumerate(prompts):
        messages = [
            {"role": "system", "content": "Expert lead scoring analyst."},
            {"role": "user", "content": prompt}
        ]
        
        response = get_groq_response(messages, api_key)
        if response:
            results.append(response)
            # Add delay between batches to avoid rate limits
            if i < len(prompts) - 1:
                time.sleep(1)
    
    return "\n\n".join(results)

def parse_lead_data(input_text):
    """Parse lead data from text input"""
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
    """Generate emails using Groq with optimized prompt"""
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

def main():
    st.set_page_config(
        page_title="Lead Scoring Dashboard", 
        page_icon="📈", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("🚀 Lead Analyzer")
        api_key = st.text_input("Enter Groq API Key", type="password")

        if st.button("Validate API Key"):
            if api_key and validate_api_key(api_key):
                st.session_state["api_key_valid"] = True
                st.session_state["api_key"] = api_key
                st.success("✅ API Key validated!")
            else:
                st.session_state["api_key_valid"] = False
                st.session_state["api_key"] = None

        st.markdown("---")
        st.write("### Navigation")
        page = st.radio("Select Tool", ["Lead Scoring", "Email Generator"])
        st.markdown("---")
        st.warning(f"Groq API Limits:\n- Max {BATCH_SIZE} leads/batch\n- {MAX_TOKENS} tokens/request")

    if page == "Lead Scoring":
        st.header("📊 Lead Scoring Engine")
        
        if not st.session_state.get("api_key_valid"):
            st.warning("Please validate your Groq API key first")
        else:
            uploaded_file = st.file_uploader(
                "Upload Leads (CSV/Excel)", 
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
                    
                    if st.button("Start Analysis", type="primary"):
                        with st.spinner(f"Processing {len(df)} leads in batches..."):
                            prompts = process_leads(df)
                            response = process_in_batches(prompts, st.session_state["api_key"])

                            if response:
                                st.subheader("Analysis Results")
                                st.markdown(response)
                                
                                st.download_button(
                                    label="Download Report",
                                    data=response,
                                    file_name="lead_scores.md",
                                    mime="text/markdown"
                                )
                            else:
                                st.error("Analysis failed")

                except Exception as e:
                    st.error(f"File processing error: {str(e)}")

    elif page == "Email Generator":
        st.header("✉️ Personalized Email Generator")
        
        if not st.session_state.get("api_key_valid"):
            st.warning("Please validate your Groq API key first")
        else:
            lead_input = st.text_area(
                "Paste Lead Data", 
                height=300,
                placeholder="Format:\nFull Name, Preferred Name, Email, Lead Score, Reason, LinkedIn, Motivation"
            )

            if st.button("Generate Emails", type="primary"):
                if lead_input.strip():
                    with st.spinner("Creating personalized emails..."):
                        leads_data = parse_lead_data(lead_input)
                        
                        if leads_data:
                            emails = generate_personalized_emails(
                                leads_data, 
                                st.session_state["api_key"]
                            )
                            
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
                                st.error("Email generation failed")
                        else:
                            st.error("Invalid input format")
                else:
                    st.warning("Please input lead data first")

if __name__ == "__main__":
    if "api_key_valid" not in st.session_state:
        st.session_state["api_key_valid"] = False
    main()
