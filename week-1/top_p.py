import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Top-p (Nucleus Sampling)
# top_p limits which tokens the model can pick from at each step
# top_p=1.0  → consider ALL tokens (full vocabulary)
# top_p=0.9  → only tokens whose combined probability reaches 90%
# top_p=0.1  → very narrow choice — very predictable output

# Low top_p = focused. High top_p = broad vocabulary choices.

prompt = "Complete this sentence in 5 words: 'The best way to learn Python is'"

# top_p=0.1 (narrow)
groq_narrow = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0, top_p=0.1)
print("=== Groq — top_p=0.1 (predictable) ===")
print(groq_narrow.invoke(prompt).content)
print(groq_narrow.invoke(prompt).content)

# top_p=0.95 (varied)
groq_wide = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.5, top_p=0.95)
print("\n=== Groq — top_p=0.95 (varied) ===")
print(groq_wide.invoke(prompt).content)
print(groq_wide.invoke(prompt).content)

# Mistral default
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3,top_p=0.9)
print("\n=== Mistral — top_p=0.9 ===")
print(mistral_llm.invoke(prompt).content)
