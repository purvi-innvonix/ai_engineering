import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from step1_transcript import load_youtube_transcript

def split_documents(docs, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Split a list of Document objects into smaller chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(docs)

def experiment_with_chunking(docs):
    """
    Experiment with chunk_size=500 and chunk_size=1000 with chunk_overlap=100.
    Prints comparison and samples.
    """
    experiments = [
        {"chunk_size": 500, "chunk_overlap": 100},
        {"chunk_size": 1000, "chunk_overlap": 100},
    ]
    
    for exp in experiments:
        c_size = exp["chunk_size"]
        c_overlap = exp["chunk_overlap"]
        
        print("\n" + "=" * 60)
        print(f" EXPERIMENT: chunk_size={c_size}, chunk_overlap={c_overlap}")
        print("=" * 60)
        
        chunks = split_documents(docs, chunk_size=c_size, chunk_overlap=c_overlap)
        print(f"[+] Total Chunks Produced: {len(chunks)}")
        
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i+1} (Length: {len(chunk.page_content)} chars) ---")
            print(chunk.page_content.strip())
            print("-" * 40)
            
    print("\n" + "=" * 60)
    print(" CHUNKING INSIGHT / SUMMARY")
    print("=" * 60)
    print(
        "• Smaller chunks (500 chars): More granular semantic precision; avoids irrelevant\n"
        "  surrounding text during retrieval, but may split related sentences across boundaries.\n"
        "• Larger chunks (1000 chars): Keeps broader context together, ideal for answering\n"
        "  questions requiring multiple connected ideas from the video.\n"
        "• Chunk overlap (100 chars): Ensures continuity across chunk boundaries so\n"
        "  meaning isn't cut off mid-thought."
    )

def main():
    default_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    user_input = input(f"Enter YouTube video URL [Press Enter for default: {default_url}]: ").strip()
    url = user_input if user_input else default_url
    
    print(f"Loading transcript from: {url}")
    docs = load_youtube_transcript(url)
    if not docs:
        print("[-] Could not load transcript.")
        return
        
    print(f"[+] Original document character length: {len(docs[0].page_content)}")
    experiment_with_chunking(docs)

if __name__ == "__main__":
    main()
