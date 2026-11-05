"""Innvonix HR Policy Q&A Bot (RAG with LangChain + FAISS).

Run:
    python hr_qa_pipeline.py                  # Gemini, interactive
    python hr_qa_pipeline.py --model ollama   # local model, interactive
    python hr_qa_pipeline.py --compare        # benchmark both models
    python hr_qa_pipeline.py --no-chunks      # hide retrieved chunks
"""

import argparse
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
POLICY_FILE = Path(__file__).parent / "innvonix_hr_policy.md"
EMBEDDING_MODEL = "nomic-embed-text"
CLOUD_MODEL = "gemini-3.5-flash-lite"
LOCAL_MODEL = "llama3.2:1b"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 4

SYSTEM_PROMPT = """You are HRBot, an HR assistant for Innvonix Technologies.

Rules:
1. Answer ONLY from the policy context below.
2. You may make direct deductions supported by the context
   (e.g. a benefit starts at confirmation, so probationers are not eligible).
3. Read tables carefully. Do not include section or clause citations (such as "Section 4.2", "(Section 3.1)", or "Clause 3") in your answer.
4. Use bullet points when listing several items.
5. Never invent rules, numbers or dates.
6. If the context is not enough, reply exactly:
   "I could not find a clear answer in the Innvonix HR Policy. Please contact hr@innvonix.com."
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "HR Policy Context:\n\n{context}\n\nQuestion: {question}"),
])


# ----------------------------------------------------------------------------
# 1. Load -> 2. Split -> 3. Embed + store   (runs once at startup)
# ----------------------------------------------------------------------------
def load_policy() -> list[Document]:
    text = POLICY_FILE.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": POLICY_FILE.name})]


def split_into_chunks(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split on headings first, then paragraphs, lines, sentences, words
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Remember which heading each chunk belongs to (used for citations)
    for chunk in chunks:
        chunk.metadata["section"] = "General"
        for line in chunk.page_content.splitlines():
            if line.startswith(("## ", "### ")):
                chunk.metadata["section"] = line.lstrip("#").strip()
                break
    return chunks


def build_vector_store(chunks: list[Document]) -> FAISS:
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


# ----------------------------------------------------------------------------
# 4. Retrieve -> 5. Generate   (runs for every question)
# ----------------------------------------------------------------------------
def retrieve(store: FAISS, question: str):
    """Return the top-k (chunk, L2 distance) pairs. Lower distance = closer match."""
    return store.similarity_search_with_score(question, k=TOP_K)


def format_context(hits) -> str:
    parts = [doc.page_content.strip() for doc, _score in hits]
    return "\n\n---\n\n".join(parts)


def get_llm(model_name: str):
    if "gemini" in model_name.lower():
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Add GOOGLE_API_KEY to your .env file")
        return ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, temperature=0.1
        )
    return ChatOllama(model=model_name, temperature=0.1)


def answer_question(store: FAISS, question: str, model_name: str):
    hits = retrieve(store, question)
    chain = PROMPT | get_llm(model_name) | StrOutputParser()

    start = time.perf_counter()
    answer = chain.invoke({"context": format_context(hits), "question": question})
    seconds = time.perf_counter() - start

    return answer, seconds, hits


# ----------------------------------------------------------------------------
# Printing
# ----------------------------------------------------------------------------
LINE = "-" * 70


def show_chunks(hits) -> None:
    print("\nRetrieved chunks:")
    for i, (doc, score) in enumerate(hits, start=1):
        preview = doc.page_content.strip().replace("\n", " ")[:200]
        print(f"  {i}. distance={score:.3f} | {doc.metadata['section']}")
        print(f"     {preview}...")


def show_answer(answer: str, seconds: float, model_name: str) -> None:
    # Remove any citations like (Section 4.2), (Section 3, Clause 1), etc.
    cleaned_answer = re.sub(r"\s*\((?:Section|Clause)[^)]*\)", "", answer)
    print(f"\nAnswer ({model_name}):\n{cleaned_answer.strip()}")
    print(f"\n[{seconds:.2f}s]\n{LINE}")


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    "How many days of Earned Leave do employees get per financial year?",
    "What is the standard notice period if an employee resigns?",
    "Can an employee take earned leave during probation?",
    "What is the meal reimbursement limit if an employee stays over 12 hours?",
    "How many days of Sick Leave are employees entitled to per year?",
    "What are the office attendance timings and required daily working hours?",
    "How many days of Marriage Leave can an employee take?",
    "What is the replacement charge for a lost punch card?",
    "How much leave is provided for birthday and wedding anniversary?",
    "When will an employee receive their relieving and experience letters after resignation?",
]



def interactive_mode(store: FAISS, model_name: str, chunks_visible: bool) -> None:
    print(f"HR Bot ready ({model_name}). Commands: samples | 1-10 | chunks | quit")

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break
        if text.lower() == "samples":
            for i, q in enumerate(SAMPLE_QUESTIONS, start=1):
                print(f"  {i:2}. {q}")
            continue
        if text.lower() == "chunks":
            chunks_visible = not chunks_visible
            print(f"Chunk display: {'ON' if chunks_visible else 'OFF'}")
            continue
        if text.isdigit():
            number = int(text)
            if not 1 <= number <= len(SAMPLE_QUESTIONS):
                print("Invalid number. Type 'samples' to see the list.")
                continue
            text = SAMPLE_QUESTIONS[number - 1]
            print(f"-> {text}")

        try:
            answer, seconds, hits = answer_question(store, text, model_name)
        except Exception as error:
            print(f"[ERROR] {error}")
            continue

        if chunks_visible:
            show_chunks(hits)
        show_answer(answer, seconds, model_name)


def run_comparison(store: FAISS) -> None:
    times = {CLOUD_MODEL: [], LOCAL_MODEL: []}

    for i, question in enumerate(SAMPLE_QUESTIONS, start=1):
        print(f"\nQ{i}/{len(SAMPLE_QUESTIONS)}: {question}\n{LINE}")

        for model_name in (CLOUD_MODEL, LOCAL_MODEL):
            try:
                answer, seconds, _ = answer_question(store, question, model_name)
                times[model_name].append(seconds)
                print(f"\n[{model_name}] {seconds:.2f}s\n{answer.strip()}")
            except Exception as error:
                print(f"\n[{model_name}] ERROR: {error}")

    print(f"\n{LINE}\nAverage response time")
    for model_name, values in times.items():
        average = sum(values) / len(values) if values else 0
        print(f"  {model_name:<20} {average:.2f}s")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Innvonix HR Policy Q&A Bot")
    parser.add_argument("--model", choices=["gemini", "ollama", "both"], default="gemini")
    parser.add_argument("--compare", action="store_true", help="benchmark both models")
    parser.add_argument("--no-chunks", action="store_true", help="hide retrieved chunks")
    args = parser.parse_args()

    if not POLICY_FILE.exists():
        raise SystemExit(f"File not found: {POLICY_FILE}")

    print("Building index...")
    chunks = split_into_chunks(load_policy())
    store = build_vector_store(chunks)
    print(f"Ready: {len(chunks)} chunks, {store.index.ntotal} vectors.\n")

    if args.compare or args.model == "both":
        run_comparison(store)
    else:
        model_name = LOCAL_MODEL if args.model == "ollama" else CLOUD_MODEL
        interactive_mode(store, model_name, chunks_visible=not args.no_chunks)


if __name__ == "__main__":
    main()