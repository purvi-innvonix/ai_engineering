import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

from step1_transcript import load_youtube_transcript
from step2_chunking import split_documents
from step3_vector_db import create_vector_db

DEFAULT_LLM_MODEL = "llama3.2:1b"

RAG_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a YouTube video.

Guidelines you MUST follow:
1. Answer using ONLY the retrieved transcript context below.
2. Explain the answer in simple, easy-to-understand language.
3. If the transcript does not contain the answer, state clearly: "I cannot find the answer in the video transcript."
4. Do NOT invent or extrapolate information that is not in the transcript.
5. Explain the summary or whole video is saying like this if question is "Summarize the video.": The video explains/talks about...."

Context:
{context}

Question:
{question}

Answer:"""

def format_docs(docs):
    """Combine page content of retrieved document chunks."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def get_chat_model(model_name: str = DEFAULT_LLM_MODEL, temperature: float = 0.1):
    """
    Factory function returning an LLM instance based on model name / provider.
    Supports:
      - Local Ollama (e.g. 'llama3.2:1b', 'llama3.2:latest', 'llama3.1:8b', 'mistral')
      - Google Gemini Cloud (e.g. 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash', 'gemini-2.5-flash')
      - OpenAI Cloud (e.g. 'gpt-4o-mini', 'gpt-4o')
    """
    model_lower = model_name.lower().strip()

    # 1. Google Gemini Cloud Model
    if "gemini" in model_lower:
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed. Run: pip install langchain-google-genai")
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY in your .env file or environment variables."
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key.strip(),
            temperature=temperature,
            streaming=True
        )

    # 2. OpenAI Cloud Model
    elif model_lower.startswith("gpt-"):
        if ChatOpenAI is None:
            raise ImportError("langchain-openai is not installed. Run: pip install langchain-openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or environment variables."
            )
        return ChatOpenAI(
            model=model_name,
            api_key=api_key.strip(),
            temperature=temperature,
            streaming=True
        )

    # 3. Local Ollama Model (Default)
    else:
        return ChatOllama(
            model=model_name,
            temperature=temperature
        )

def build_rag_chain(vector_db, llm_model: str = DEFAULT_LLM_MODEL, k: int = 3):
    """
    Construct an LCEL RAG chain from vector_db and chosen LLM (Local Ollama or Cloud).
    Returns (rag_chain, retriever).
    """
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    llm = get_chat_model(model_name=llm_model, temperature=0.1)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

def ask_question(rag_chain, retriever, question: str):
    """
    Ask a question through the RAG chain, printing both retrieved context and answer.
    """
    print("\n" + "=" * 60)
    print(f" QUESTION: {question}")
    print("=" * 60)
    
    # Retrieve documents to inspect context
    retrieved_docs = retriever.invoke(question)
    print("\n--- Retrieved Chunks ---")
    for i, doc in enumerate(retrieved_docs, start=1):
        snippet = doc.page_content.strip().replace("\n", " ")
        print(f"[{i}] {snippet[:150]}...")
        
    print("\n--- Generating Answer ---")
    answer = rag_chain.invoke(question)
    print(f"\nAnswer:\n{answer}")

def main():
    default_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    user_input = input(f"Enter YouTube video URL [Press Enter for default: {default_url}]: ").strip()
    url = user_input if user_input else default_url    
    print(f"Loading transcript from: {url}")
    docs = load_youtube_transcript(url)
    if not docs:
        print("[-] Could not load transcript.")
        return
        
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
    vector_db = create_vector_db(chunks)
    
    rag_chain, retriever = build_rag_chain(vector_db)
    
    # Test 1: In-context question
    ask_question(
        rag_chain,
        retriever,
        "What promises does the singer make?"
    )
    
    # Test 2: Out-of-context question (should state not found)
    ask_question(
        rag_chain,
        retriever,
        "Can you summarize this video?"
    )

if __name__ == "__main__":
    main()
