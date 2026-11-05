import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Mini Project — Employee Question Assistant
# Combines: prompt template + model + structured output
# Step 1: plain question
# Step 2: reusable prompt
# Step 3: add a model
# Step 4: structured output

# ── STEP 1: Plain question ──
question = "What is ESIC?"
print("=== Step 1: Plain Question ===")
print("Question:", question)

# ── STEP 2: Reusable Prompt ──
prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain HR concepts to employees in simple language. Keep it under 50 words."),
    ("human", "{question}")
])
print("\n=== Step 2: Prompt Template ===")
print(prompt.invoke({"question": question}))

# ── STEP 3: Add a Model — all 3 providers ──
groq_llm    = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
# google_llm  = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.8-flash", temperature=0.3)

chain_groq    = prompt | groq_llm    | StrOutputParser()
chain_mistral = prompt | mistral_llm | StrOutputParser()
# chain_google  = prompt | google_llm  | StrOutputParser()

print("\n=== Step 3: Model Answers ===")
print("Groq:   ", chain_groq.invoke({"question": question}))
print("Mistral:", chain_mistral.invoke({"question": question}))
# print("Google: ", chain_google.invoke({"question": question}))

# ── STEP 4: Structured Output ──
class Answer(BaseModel):
    topic: str = Field(description="The HR topic being asked about")
    answer: str = Field(description="Simple explanation under 40 words")
    confidence_note: str = Field(description="One sentence note about accuracy")

print("\n=== Step 4: Structured Output — Groq ===")
structured = groq_llm.with_structured_output(Answer)
result = structured.invoke(f"Explain: {question}")
print("Topic:", result.topic)
print("Answer:", result.answer)
print("Note:", result.confidence_note)

print("\n=== Step 4: Structured Output — Mistral ===")
structured = mistral_llm.with_structured_output(Answer)
result = structured.invoke(f"Explain: {question}")
print("Topic:", result.topic)
print("Answer:", result.answer)
print("Note:", result.confidence_note)
