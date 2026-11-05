import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from step1_transcript import load_youtube_transcript
from step2_chunking import split_documents
from step3_vector_db import create_vector_db
from step4_rag import build_rag_chain

BANNER = """
=============================================================
           YOUTUBE VIDEO RAG INTERACTIVE CHATBOT            
=============================================================
Ask questions about any YouTube video transcript using RAG.
Type 'exit', 'quit', or 'q' to end the session.
=============================================================
"""

def run_chatbot():
    print(BANNER)
    
    default_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    url_input = input(f"Enter YouTube video URL [Press Enter for default: {default_url}]: ").strip()
    url = url_input if url_input else default_url

    try:
        docs = load_youtube_transcript(url)
        if not docs:
            print("[-] No transcript found for this video.")
            return
    except Exception as e:
        print(f"[-] Error loading transcript: {e}")
        return

    # Split document into chunks
    print("[+] Splitting transcript into chunks...")
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)
    print(f"[+] Total chunks prepared: {len(chunks)}")

    # Build vector database with embeddings
    vector_db = create_vector_db(chunks)

    # Choose LLM (Ollama or Cloud)
    print("\n[+] Available Model Options: e.g. 'llama3.2:1b' (Local), 'gemini-1.5-flash' (Cloud), 'gpt-4o-mini' (Cloud)")
    model_choice = input("Select model [Press Enter for default: llama3.2:1b]: ").strip()
    selected_model = model_choice if model_choice else "llama3.2:1b"

    # Initialize RAG chain with chosen model
    print(f"[+] Initializing RAG pipeline with model '{selected_model}'...")
    rag_chain, retriever = build_rag_chain(vector_db, llm_model=selected_model)
    print("[+] System ready! You can now ask questions about the video.\n")

    while True:
        try:
            question = input("\n👤 Your Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Exiting chat. Goodbye!")
            break

        # Display retrieved transcript sections for transparency
        retrieved_docs = retriever.invoke(question)
        print("\n--- 📄 Retrieved Context Snippets ---")
        for i, doc in enumerate(retrieved_docs, start=1):
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"[{i}] {snippet[:150]}...")

        # Generate answer
        print("\n--- 🤖 AI Answer ---")
        try:
            answer = rag_chain.invoke(question)
            print(answer)
        except Exception as e:
            print(f"[-] Error generating answer: {e}")
        print("-" * 50)

if __name__ == "__main__":
    run_chatbot()
