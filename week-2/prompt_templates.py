import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Prompt Templates
# A prompt template is a REUSABLE prompt with {placeholders}
# Instead of writing a new prompt each time, you fill in the blanks

# ─────────────────────────────────
# 1. Simple template with one variable
# ─────────────────────────────────
simple_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one sentence for a beginner."
)

messages = simple_prompt.invoke({"topic": "Machine Learning"})
print(messages)

# ─────────────────────────────────
# 2. Chat template with system + human roles
# ─────────────────────────────────
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful {role}. Keep answers short."),
    ("human", "Explain {topic}.")
])

messages = chat_prompt.invoke({"role": "Python tutor", "topic": "lists"})
print(messages)

# ─────────────────────────────────
# 3. Chain: prompt | model — using all 3 providers
# ─────────────────────────────────
chain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert tutor. Be brief."),
    ("human", "Explain {topic} in under {words} words.")
])

groq_llm    = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
google_llm  = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)

inputs = {"topic": "APIs", "words": "30"}

print("\n=== Prompt + Model Chain ===")
chain_groq    = chain_prompt | groq_llm    | StrOutputParser()
chain_mistral = chain_prompt | mistral_llm | StrOutputParser()
chain_google  = chain_prompt | google_llm  | StrOutputParser()

print("Groq:   ", chain_groq.invoke(inputs))
print("Mistral:", chain_mistral.invoke(inputs))
print("Google: ", chain_google.invoke(inputs))
