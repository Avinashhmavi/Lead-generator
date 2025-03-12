```markdown
# Groq Lead Analyzer 🚀

AI-powered lead scoring and email generation platform using Groq's large language models.

## Features 🌟

- **Real-time Lead Scoring** with multi-dimensional analysis
- **Personalized Email Generation** based on lead motivations
- **Batch Processing** with automatic rate limiting
- **Secure API Key Handling** via Streamlit Secrets
- **CSV/Excel Import** for lead data
- **Downloadable Reports** in markdown format

## Installation 🛠️

1. Clone the repository:
```bash
git clone https://github.com/yourusername/groq-lead-analyzer.git
cd groq-lead-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Groq API key:
- Create `.streamlit/secrets.toml` with:
```toml
[groq]
api_key = "your_groq_api_key_here"
```

## Usage 🚀

1. Run the app:
```bash
streamlit run app.py
```

2. Open your browser at http://localhost:8501

3. Use the tabs for:
- **Lead Scoring**: Upload CSV/Excel files with lead data
- **Email Generation**: Create personalized outreach emails

## Contributing 🤝

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License 📝

[MIT](https://choosealicense.com/licenses/mit/)
```

Key improvements made:

1. **Secrets Management**: API key now read from Streamlit secrets
2. **Enhanced UI**:
   - Modern tabbed interface
   - Improved visual hierarchy
   - Feature highlights
   - Better error handling
   - Consistent branding
3. **Streamlined Workflow**:
   - Automatic API key validation
   - Clear instructions for setup
   - Better progress indicators
4. **Code Structure**:
   - Separated concerns into functions
   - Improved error handling
   - Better type checking

To use this:

1. Create `.streamlit/secrets.toml` with your Groq API key
2. Install requirements with `pip install -r requirements.txt`
3. Run with `streamlit run app.py`
