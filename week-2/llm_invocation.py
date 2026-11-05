import os
import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

question = "Name one planet in our solar system. One word only."

groq_llm    = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0)
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0)
google_llm  = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0)

# ─────────────────────────────────
# 1. invoke() — synchronous, waits for the full response
# ─────────────────────────────────
print("=== invoke() — waits for complete response ===")
print("Groq:   ", groq_llm.invoke(question).content)
print("Mistral:", mistral_llm.invoke(question).content)
print("Google: ", google_llm.invoke(question).text)

# ─────────────────────────────────
# 2. stream() — synchronous streaming, prints word by word
# ─────────────────────────────────
print("\n=== stream() — prints chunks as they arrive ===")
print("Groq stream: ", end="", flush=True)
for chunk in groq_llm.stream("Count from 1 to 5, space separated.also explain the RAG in 10 words only"):
    print(chunk.content, end="", flush=True)
print()

print("Mistral stream: ", end="", flush=True)
for chunk in mistral_llm.stream("Count from 1 to 5, space separated."):
    print(chunk.content, end="", flush=True)
print()

print("Google stream: ", end="", flush=True)
for chunk in google_llm.stream("Count from 1 to 5, space separated."):
    print(chunk.text, end="", flush=True)
print()

# ─────────────────────────────────
# 3. ainvoke() — async version of invoke()
# ─────────────────────────────────
async def demo_ainvoke():
    print("\n=== ainvoke() — asynchronous invoke ===")
    result = await groq_llm.ainvoke(question)
    print("Groq ainvoke:   ", result.content)
    result = await mistral_llm.ainvoke(question)
    print("Mistral ainvoke:", result.content)
    result = await google_llm.ainvoke(question)
    print("Google ainvoke: ", result.text)

# ─────────────────────────────────
# 4. astream() — async streaming
# ─────────────────────────────────
async def demo_astream():
    print("\n=== astream() — async streaming ===")
    print("Groq astream: ", end="", flush=True)
    async for chunk in groq_llm.astream("Count from 1 to 5, space separated."):
        print(chunk.content, end="", flush=True)
    print()

async def main():
    await demo_ainvoke()
    await demo_astream()

asyncio.run(main())
