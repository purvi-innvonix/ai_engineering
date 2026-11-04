import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Max Output Tokens
# max_tokens limits how LONG the model's response can be
# Short limit → model stops mid-sentence if needed
# Higher limit → longer full responses (costs more tokens)
#
# Default varies by provider: Groq ~1024, Mistral ~2048, Gemini ~2048+

prompt = "Write a detailed explanation of how the internet works."

# Short limit 
groq_short = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3, max_tokens=20)
print("=== Groq — max_tokens=20 (very short) ===")
print(groq_short.invoke(prompt).content)

# Longer limit
groq_long = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3, max_tokens=100)
print("\n=== Groq — max_tokens=100 (more complete) ===")
print(groq_long.invoke(prompt).content)

# Mistral — default
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
print("\n=== Mistral — default max_tokens ===")
print(mistral_llm.invoke("In one sentence, what is the internet?").content)

# Google — default
google_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)
print("\n=== Google Gemini — default max_tokens ===")
print(google_llm.invoke("In one sentence, what is the internet?").content)
