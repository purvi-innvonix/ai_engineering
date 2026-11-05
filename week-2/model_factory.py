import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# ============================================================
# model_factory.py
# ============================================================
#
# A single place to create LLM instances for any provider.
# Usage:
#   from model_factory import get_model
#   llm = get_model("groq")
#   llm = get_model("mistral")
#   llm = get_model("gemini")
#
# Raises ValueError if the required API key is not set.
# ============================================================

PROVIDER_CONFIG = {
    "groq": {
        "env_key":   "GROQ_API_KEY",
        "model":     "openai/gpt-oss-120b",
        "cls":       ChatGroq,
    },
    "mistral": {
        "env_key":   "MISTRAL_API_KEY",
        "model":     "codestral-2508",
        "cls":       ChatMistralAI,
    },
    "gemini": {
        "env_key":   "GOOGLE_API_KEY",
        "model":     "gemini-3.8-flash",
        "cls":       ChatGoogleGenerativeAI,
    },
}


def get_model(provider: str, temperature: float = 0):
    """
    Return a LangChain chat model for the given provider name.

    Args:
        provider    : One of "groq", "mistral", "gemini"
        temperature : Sampling temperature (default 0 for deterministic output)

    Raises:
        ValueError  : If provider is unknown or its API key env-var is not set
    """
    provider = provider.lower()

    if provider not in PROVIDER_CONFIG:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Choose from: {list(PROVIDER_CONFIG.keys())}"
        )

    config  = PROVIDER_CONFIG[provider]
    api_key = os.getenv(config["env_key"])

    if not api_key:
        raise ValueError(
            f"API key not found for provider '{provider}'. "
            f"Set the environment variable: {config['env_key']}"
        )

    return config["cls"](
        api_key=api_key,
        model=config["model"],
        temperature=temperature,
    )
