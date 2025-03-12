```markdown
#  Lead Analyzer 🚀

AI-powered lead scoring and email generation platform using Groq's large language models.

**Live Demo**: [https://lead-generator1.streamlit.app/](https://lead-generator1.streamlit.app/)

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
git clone https://github.com/Avinashhmavi/Lead-generator.git
cd Lead-generator
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
