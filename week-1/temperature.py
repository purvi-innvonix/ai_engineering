import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Temperature
# Temperature controls how RANDOM/CREATIVE the model's responses are
# temperature=0.0  → very deterministic, almost same answer every time
# temperature=0.3  → slight variation (good for Q&A, coding)
# temperature=0.8  → creative, varied (good for brainstorming, writing)
# temperature=1.2+ → very random, sometimes incoherent

prompt = "Give me one creative product name for a coffee brand. One word only."

groq_llm_low  = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.0)
groq_llm_high = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=1.0)

print("=== Groq — Temperature 0.0 (deterministic) ===")
print(groq_llm_low.invoke(prompt).content)
print(groq_llm_low.invoke(prompt).content)  # same answer again
print(groq_llm_low.invoke(prompt).content)  # same again

print("\n=== Groq — Temperature 1.0 (creative/random) ===")
print(groq_llm_high.invoke(prompt).content)
print(groq_llm_high.invoke(prompt).content)  # different!
print(groq_llm_high.invoke(prompt).content)  # different again!

# Show default temperature (0.3) on Mistral and Google
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
print("\n=== Mistral — Temperature 0.3 ===")
print(mistral_llm.invoke(prompt).content)

google_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)
print("\n=== Google Gemini — Temperature 0.3 ===")
print(google_llm.invoke(prompt).content)
