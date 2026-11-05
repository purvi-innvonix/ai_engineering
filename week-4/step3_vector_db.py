import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

from step1_transcript import load_youtube_transcript
from step2_chunking import split_documents

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

def create_vector_db(chunks, model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Convert text chunks into embeddings and store them in a FAISS vector database.
    """
    print(f"\n[+] Initializing OllamaEmbeddings with model: '{model_name}'...")
    embeddings = OllamaEmbeddings(model=model_name)
    
    print(f"[+] Generating embeddings and building FAISS index for {len(chunks)} chunks...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    print("[+] Vector database successfully created!")
    return vector_db

def test_similarity_search(vector_db, queries=None):
    """
    Test similarity search on the vector database with sample questions.
    Prints the retrieved chunks without using an LLM.
    """
    if queries is None:
        queries = [
            "What is the capital of india?",
            "What is corrective RAG?",
            "What does the singer want to tell and make understand?"
        ]
        
    print("\n" + "=" * 60)
    print(" TESTING SIMILARITY SEARCH (NO LLM)")
    print("=" * 60)
    
    for query in queries:
        print(f"\n🔍 Query: \"{query}\"")
        # similarity_search_with_score returns (doc, distance)
        results = vector_db.similarity_search_with_score(query, k=2)
        
        for rank, (doc, score) in enumerate(results, start=1):
            print(f"\n  [Match {rank} | L2 Distance: {score:.4f}]")
            print(f"  Source Video: {doc.metadata.get('source', 'unknown')}")
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"  Content: {snippet[:200]}...")

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
    print(f"[+] Split into {len(chunks)} chunks.")
    
    vector_db = create_vector_db(chunks)
    test_similarity_search(vector_db)

if __name__ == "__main__":
    main()
