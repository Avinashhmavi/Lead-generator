import streamlit as st
from openai import OpenAI
import pandas as pd
import json
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Groq API configuration
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "mixtral-8x7b-32768"
MAX_TOKENS = 32768  # Full context window
BATCH_SIZE = 20      # Increased batch size
MAX_RETRIES = 3      # Reduced retries with exponential backoff

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(MAX_RETRIES))
def get_groq_response(messages, api_key):
    """Get response from Groq API with exponential backoff retry"""
    client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=MAX_TOKENS,
    )
    return response.choices[0].message.content

def process_leads(df):
    """Optimized lead processing with dynamic token management"""
    leads_data = df.to_dict("records")
    batches = [leads_data[i:i + BATCH_SIZE] 
              for i in range(0, len(leads_data), BATCH_SIZE)]
    
    prompts = []
    for batch in batches:
        # Simplified prompt structure to save tokens
        prompt = f"""
        Score {len(batch)} leads using criteria: 
        1. Channel Presence (0-50)
        2. Professional Experience (0-50)
        3. Career Motivation (0-30)
        4. Geography (0-30)
        
        Output JSON format:
        {{ "leads": [
            {{
                "Full Name": "",
                "Preferred Name": "",
                "Email": "",
                "Lead Score": 0,
                "Reason": "",
                "LinkedIn": "",
                "Motivation": ""
            }},
            ...
        ]}}
        
        Data: {json.dumps(batch)}
        """
        prompts.append(prompt.strip())
    
    return prompts

def process_in_batches(prompts, api_key):
    """Optimized batch processing with concurrent requests"""
    results = []
    with st.spinner(f"Processing {len(prompts)} batches..."):
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "You are an expert lead scoring analyst. Output only JSON with no markdown."},
                {"role": "user", "content": prompt}
            ]
            
            response = get_groq_response(messages, api_key)
            if response:
                results.append(response)
            # Removed inter-batch delay as we're using retry logic
    return results

def main():
    st.set_page_config(
        page_title="Lead Scoring Dashboard", 
        page_icon="📈", 
        layout="wide"
    )

    # API Key management
    if "api_key" not in st.session_state:
        if "GROQ_API_KEY" in st.secrets:
            st.session_state["api_key"] = st.secrets["GROQ_API_KEY"]
            st.session_state["api_key_valid"] = True
        else:
            st.session_state["api_key"] = None
            st.session_state["api_key_valid"] = False

    with st.sidebar:
        st.title("🚀 Groq Lead Analyzer")
        
        if not st.session_state["api_key_valid"]:
            api_key_input = st.text_input("Enter Groq API Key", type="password")
            if st.button("Validate API Key"):
                if validate_api_key(api_key_input):
                    st.session_state["api_key"] = api_key_input
                    st.session_state["api_key_valid"] = True
                    st.success("✅ API Key validated!")
                else:
                    st.session_state["api_key"] = None
                    st.session_state["api_key_valid"] = False

    if st.session_state["api_key_valid"]:
        uploaded_file = st.file_uploader(
            "Upload Leads (CSV/Excel)", 
            type=["csv", "xlsx"],
            accept_multiple_files=False
        )

        if uploaded_file:
            df = pd.read_csv(uploaded_file) if uploaded_file.type == "text/csv" else pd.read_excel(uploaded_file)
            st.success(f"Loaded {len(df)} leads")
            
            if st.button("Start Analysis", type="primary"):
                with st.spinner(f"Processing {len(df)} leads in {BATCH_SIZE}-lead batches..."):
                    prompts = process_leads(df)
                    raw_results = process_in_batches(prompts, st.session_state["api_key"])
                    
                    # Combine and parse results
                    all_leads = []
                    for result in raw_results:
                        try:
                            all_leads.extend(json.loads(result)["leads"])
                        except json.JSONDecodeError:
                            st.error("Error parsing API response")
                    
                    if all_leads:
                        results_df = pd.DataFrame(all_leads)
                        st.dataframe(results_df)
                        
                        # Download buttons
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            "Download CSV",
                            csv,
                            "lead_scores.csv",
                            "text/csv"
                        )
                    else:
                        st.error("No valid results received from API")

if __name__ == "__main__":
    main()
