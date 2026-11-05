from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq

load_dotenv()
docs_path = 'makeup_overview.txt'
loader = TextLoader(docs_path)
docs =loader.load()

splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=50,
    chunk_overlap=5,
    length_function=len,
)

chunks = splitter.split_documents(docs)

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}: {chunk}")



# Recursive character splitter
python_text = """def greet_user(name):
    return f"Hello, {name}! Welcome to Python."

result = greet_user("Purvi")
print(result)"""

recursive_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=5,
)

recursive_chunks = recursive_splitter.split_text(python_text)
print(recursive_chunks)

# ==========================================
# 3. Semantic Text Splitter & Semantic Search Example
# ==========================================
# Semantic Chunking splits text based on semantic similarity between consecutive
# sentences rather than hard character counts.
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Initialize SemanticChunker using embeddings
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",  # "percentile", "standard_deviation", or "interquartile"
    )

    # Split documents into semantic chunks
    semantic_chunks = semantic_splitter.split_documents(docs)

    print(f"\n--- Semantic Chunker (Generated {len(semantic_chunks)} chunks) ---")
    for i, chunk in enumerate(semantic_chunks, start=1):
        print(f"Semantic Chunk {i}:\n{chunk.page_content.strip()}\n")

    # Semantic Search using InMemoryVectorStore
    print("--- Semantic Search Example ---")
    vector_store = InMemoryVectorStore.from_documents(semantic_chunks, embeddings)

    query = "How should I do my eye makeup?"
    search_results = vector_store.similarity_search(query, k=2)

    print(f"Query: '{query}'")
    for rank, result in enumerate(search_results, start=1):
        print(f"Top Match {rank}: {result.page_content.strip()}\n")

except Exception as e:
    print(f"\n[Note on OpenAI Embeddings]: {e}")
    print("To execute semantic chunking and search with OpenAIEmbeddings, make sure OPENAI_API_KEY in .env is set to a valid OpenAI key (e.g. sk-proj-...).")
