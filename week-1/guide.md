# Week 1 – Setup and Installation Guide

This guide covers the setup required to run the following Week 1 scripts:

- `temperature.py`
- `top_p.py`
- `max_output_tokens.py`

## Linux Simple Steps

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install python-dotenv langchain-groq langchain-mistralai langchain-google-genai
```

Create `.env` in project root:

```env
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key
```

Run:

```bash
python week-1/temperature.py
python week-1/top_p.py
python week-1/max_output_tokens.py
```

## 1. Prerequisites

Make sure you have:

- Python 3.10 or higher
- `python3 -m pip` available
- Internet connection
- API keys for Groq, Mistral, and Google Gemini

Check Python and pip versions:

```bash
python3 --version
python3 -m pip --version
```

## 2. Create and Activate Virtual Environment

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

Upgrade pip:

```bash
python3 -m pip install --upgrade pip
```

Install the required libraries:

```bash
python3 -m pip install python-dotenv langchain-groq langchain-mistralai langchain-google-genai
```

### Required Libraries

| Library | Purpose |
|---|---|
| `python-dotenv` | Loads environment variables from `.env` |
| `langchain-groq` | Integrates Groq models |
| `langchain-mistralai` | Integrates Mistral models |
| `langchain-google-genai` | Integrates Google Gemini models |

## 4. Configure API Keys

Create a `.env` file in the project root.

Add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key
```

Make sure the environment variable names match those used in your Python scripts.

## 5. Run Week 1 Scripts

Execute the scripts from the project root:

```bash
python week-1/temperature.py
python week-1/top_p.py
python week-1/max_output_tokens.py
```

### What Each Script Demonstrates

| Script | Concept |
|---|---|
| `temperature.py` | How temperature affects response creativity and randomness |
| `top_p.py` | How top-p affects token selection diversity |
| `max_output_tokens.py` | How max tokens limits response length |

---

**Note:** Activate the `.venv` environment whenever you open a new terminal before running the scripts.