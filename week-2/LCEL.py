import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Put the Pieces Together (Full LCEL Pipeline)
# User → Prompt Template → Messages → LangChain Model → Provider API → LLM → Parser → App
#
# This file demonstrates the full pipeline with all 3 providers

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Answer concisely."),
    ("human", "{question}")
])

groq_llm    = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
google_llm  = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)

parser = StrOutputParser()

# Full pipeline chains
chain_groq    = prompt | groq_llm    | parser
chain_mistral = prompt | mistral_llm | parser
chain_google  = prompt | google_llm  | parser

inputs = {
    "role": "Python programming tutor",
    "question": "What is a dictionary in Python? One sentence only."
}

print("=== Full Pipeline — Groq ===")
print(chain_groq.invoke(inputs))

print("\n=== Full Pipeline — Mistral ===")
print(chain_mistral.invoke(inputs))

print("\n=== Full Pipeline — Google Gemini ===")
print(chain_google.invoke(inputs))

print("\n--- Pipeline Flow ---")
print("Prompt Template  →  Model (Groq/Mistral/Google)  →  StrOutputParser  →  String Result")
