# Innvonix HR Policy Q&A Bot — Architecture, Implementation & Model Comparison

> **Project:** RAG-based HR Policy Q&A System  
> **Tech Stack:** Python · LangChain · FAISS · OllamaEmbeddings · Gemini (Cloud) · Llama3.2 (Local)  
> **Document:** Architecture · Implementation · Comparison Results

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [RAG Pipeline — Step-by-Step Implementation](#2-rag-pipeline--step-by-step-implementation)
3. [Component Details](#3-component-details)
4. [Retrieval: Chunks & Similarity Scores](#4-retrieval-chunks--similarity-scores)
5. [Model Comparison: Cloud vs Local](#5-model-comparison-cloud-vs-local)
6. [Benchmark Results](#6-benchmark-results)
7. [Key Observations & Recommendations](#7-key-observations--recommendations)
8. [How to Run](#8-how-to-run)

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Innvonix HR Policy Q&A Bot                       │
│                  (Retrieval-Augmented Generation)                   │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │  OFFLINE INDEXING PHASE  (runs once at startup)              │
    │                                                              │
    │   innvonix_hr_policy.md                                     │
    │          │                                                   │
    │          ▼                                                   │
    │   [Step 1] Document Loader                                   │
    │     LangChain Document object with metadata                  │
    │          │                                                   │
    │          ▼                                                   │
    │   [Step 2] Recursive Text Chunker                            │
    │     chunk_size=600, overlap=120                              │
    │     65 chunks with section metadata                          │
    │          │                                                   │
    │          ▼                                                   │
    │   [Step 3] OllamaEmbeddings (nomic-embed-text)              │
    │     Converts each chunk → 768-dim dense vector               │
    │          │                                                   │
    │          ▼                                                   │
    │   [Step 3] FAISS Vector Index                                │
    │     In-memory L2 similarity index                            │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │  ONLINE QUERY PHASE  (per user question)                     │
    │                                                              │
    │   User Question (CLI)                                        │
    │          │                                                   │
    │          ▼                                                   │
    │   [Step 4] Retriever                                         │
    │     similarity_search_with_score(query, k=4)                 │
    │     Returns: top-4 chunks + L2 distance scores               │
    │          │                                                   │
    │          ▼                                                   │
    │   [Step 5] Context Formatter                                 │
    │     Assembles chunks into structured prompt context          │
    │          │                                                   │
    │          ▼                                                   │
    │   [Step 5] ChatPromptTemplate                                │
    │     System: HRBot identity + grounding rules                 │
    │     Human:  Context + Question                               │
    │          │                                                   │
    │          ├──────────────────────┬───────────────────────┐    │
    │          ▼                      ▼                       │    │
    │   ChatGoogleGenerativeAI   ChatOllama                   │    │
    │   (gemini-2.0-flash)       (llama3.2:1b)               │    │
    │   Cloud API (Google)       Local Inference              │    │
    │          │                      │                       │    │
    │          └──────────────────────┘                       │    │
    │                      │                                  │    │
    │                      ▼                                  │    │
    │             StrOutputParser                              │    │
    │                      │                                  │    │
    │                      ▼                                  │    │
    │          Formatted Answer + Latency                      │    │
    └──────────────────────────────────────────────────────────┘
```

---

## 2. RAG Pipeline — Step-by-Step Implementation

### Step 1 — Document Loading

```python
from langchain_core.documents import Document
from pathlib import Path

def load_hr_policy(file_path: Path) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    doc = Document(
        page_content=content,
        metadata={
            "source": str(file_path),
            "doc_type": "hr_policy",
            "company": "Innvonix Technologies",
            "version": "2.1",
        },
    )
    return [doc]
```

**What it does:**  
Reads the entire `innvonix_hr_policy.md` (23,442 characters) into a LangChain `Document` object along with structured metadata about its source, type, and version.

---

### Step 2 — Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120,
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    add_start_index=True,
)
chunks = splitter.split_documents(docs)
```

**What it does:**  
Splits the document into 65 smaller, overlapping chunks. The separators are ordered by markdown structure (H2 → H3 → paragraph → line), ensuring semantic coherence within each chunk. The 120-char overlap prevents important context from being cut at chunk boundaries.

**Why these numbers?**  
- `chunk_size=600`: Small enough to stay focused on a single policy clause; large enough to contain a full table row or sub-section.
- `chunk_overlap=120`: ~20% overlap prevents key policy details being stranded at boundaries.

---

### Step 3 — Embedding & Vector Store

```python
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_documents(chunks, embeddings)
```

**What it does:**  
- `nomic-embed-text` (Ollama) converts each chunk into a 768-dimensional dense vector that captures its semantic meaning.
- FAISS builds an in-memory L2 (Euclidean distance) index over all 65 vectors, enabling millisecond-speed nearest-neighbor search.

**Why FAISS?**  
- Zero infrastructure overhead — no database server needed.
- Ideal for moderate document sizes (< 100k chunks).
- Exact L2 similarity search (no approximation errors at this scale).

---

### Step 4 — Retrieval with Similarity Scores

```python
results = vector_store.similarity_search_with_score(query, k=4)
# Returns: List[Tuple[Document, float]]
# float = L2 distance (lower = more similar)
```

**What it does:**  
For any user question, the retriever finds the 4 most semantically similar chunks. The L2 distance score is displayed to the user alongside an approximate similarity percentage for interpretability.

**Score interpretation:**
| L2 Distance | Approx. Similarity | Interpretation |
|-------------|-------------------|----------------|
| 0.0 – 0.5   | 95–100%           | Extremely relevant match |
| 0.5 – 1.0   | 90–95%            | Highly relevant |
| 1.0 – 2.0   | 80–90%            | Relevant but broader match |
| > 2.0       | < 80%             | Marginal relevance |

---

### Step 5 — Prompt Construction

```python
HR_SYSTEM_PROMPT = """
You are HRBot, a helpful HR assistant for Innvonix Technologies.
Answer ONLY from the policy context provided.
If the context does not contain the answer, respond:
"I could not find a clear answer. Please contact hr@innvonix.com."
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", HR_SYSTEM_PROMPT),
    ("human", "HR Policy Context:\n\n{context}\n\nQuestion: {question}"),
])
```

**What it does:**  
Creates a structured two-part prompt:  
1. **System message** — Grounds the model as an HR specialist, enforces context-only answers, and defines the fallback response.
2. **Human message** — Injects the retrieved chunks as `{context}` and the user's question as `{question}`.

---

### Step 6 — Generation (LCEL Chain)

```python
chain = RAG_PROMPT | llm | StrOutputParser()
answer = chain.invoke({"context": context, "question": question})
```

**LangChain Expression Language (LCEL) Chain:**
```
ChatPromptTemplate → LLM (Cloud or Local) → StrOutputParser
```

The chain is identical for both models — only the `llm` object changes. This is a key design feature enabling easy model swapping.

---

## 3. Component Details

| Component | Choice | Reason |
|-----------|--------|--------|
| **Document Loader** | Custom markdown reader | Direct file control, preserves structure |
| **Chunker** | RecursiveCharacterTextSplitter | Markdown-aware, preserves semantic units |
| **Embedding Model** | `nomic-embed-text` (Ollama) | Fast, free, 768-dim, runs locally |
| **Vector Store** | FAISS (CPU) | Zero-infra, sub-ms retrieval |
| **Cloud LLM** | `gemini-2.0-flash` | Fast, accurate, cost-effective frontier model |
| **Local LLM** | `llama3.2:1b` | Ultra-fast inference, fully offline, 1B params |
| **Chain Framework** | LangChain LCEL | Composable, readable pipeline |
| **Prompt Strategy** | System + Human messages | Clear role separation, anti-hallucination grounding |

---

## 4. Retrieval: Chunks & Similarity Scores

### Sample Output — "How many casual leaves does an employee get per year?"

```
Retrieved Chunks (passed to LLM as context)
──────────────────────────────────────────────────────

  Chunk 1  │  L2 Distance: 0.8426  │  Approx. Similarity: 91.6%
  Section: General
    | Leave Type  | Days/Year | Notes ...
    | Casual Leave (CL) | 6  | Non-encashable; lapses at year-end ...

  Chunk 2  │  L2 Distance: 0.8695  │  Approx. Similarity: 91.3%
  Section: 4.1 Annual Leave Entitlement
    ### 4.1 Annual Leave Entitlement

  Chunk 3  │  L2 Distance: 0.9137  │  Approx. Similarity: 90.9%
  Section: 4.3 Sick Leave Rules
    Sick leave of up to 2 consecutive days can be applied without
    a medical certificate ...

  Chunk 4  │  L2 Distance: 0.9696  │  Approx. Similarity: 90.3%
  Section: 4.2 Earned Leave Rules
    EL can be accumulated up to a maximum of 45 days at any point ...
```

### Why Display Scores?

Displaying retrieved chunks and their scores serves critical purposes:
1. **Transparency** — Users can verify what context the LLM actually received.
2. **Debugging** — Low-similarity chunks reveal gaps in the knowledge base.
3. **Trust** — Grounding proof that the answer isn't hallucinated.
4. **Education** — Demonstrates how semantic similarity works in practice.

---

## 5. Model Comparison: Cloud vs Local

### Cloud Model — `gemini-2.0-flash` (Google Gemini)

**Characteristics:**
- Frontier model with 100B+ effective parameters.
- Accessed via Google's API (requires API key and internet).
- Supports streaming, structured output, and 1M token context.
- Temperature set to 0.1 for deterministic, policy-faithful answers.

**Strengths for HR Q&A:**
- Excellent at parsing complex policy tables (e.g., leave entitlement tables).
- Produces structured, bullet-pointed answers naturally.
- Reliably refuses to hallucinate when prompted to stay in context.
- Handles multi-part questions gracefully.

**Weaknesses:**
- Requires active internet connection.
- Per-token cost (small but non-zero for production usage).
- Sends HR policy text and user queries to Google's servers.
- Latency dependent on network speed and API load.

---

### Local Model — `llama3.2:1b` (Meta Llama via Ollama)

**Characteristics:**
- 1 Billion parameter model — extremely compact.
- Runs entirely on CPU; no GPU required.
- Fully private — zero data leaves the machine.
- Free to run with no per-query cost.

**Strengths for HR Q&A:**
- Complete offline operation — suitable for air-gapped enterprise environments.
- Zero marginal cost per query.
- No privacy/data residency concerns.
- Fast inference on modern CPUs (8–15s per response).

**Weaknesses:**
- Limited reasoning on complex multi-part questions.
- Can fail to extract specific values (e.g., numbers) from markdown tables in chunks.
- Sometimes falls back to the "contact HR" fallback even when the answer is in the provided context.
- Does not produce as structured or citation-rich answers as Gemini.

---

## 6. Benchmark Results

### Test Questions Used

| # | Question |
|---|----------|
| 1 | How many casual leaves does an employee get per year? |
| 2 | What is the notice period for a Manager at Innvonix? |
| 3 | Can an employee take earned leave during probation? |
| 4 | How is the variable pay structured and when is it paid? |
| 5 | What happens if an employee is rated 1 in the performance review? |
| 6 | What is the WFH entitlement per week for confirmed employees? |
| 7 | How is gratuity calculated and when is it applicable? |
| 8 | What are the POSH complaint filing procedures? |
| 9 | What is the L&D budget for an Individual Contributor? |
| 10 | What medical insurance coverage does Innvonix provide? |

### Performance Metrics

| Metric | Cloud (gemini-2.0-flash) | Local (llama3.2:1b) |
|--------|--------------------------|----------------------|
| Avg. Response Time | ~2.5s | ~10–15s |
| Min Response Time | ~1.8s | ~8s |
| Max Response Time | ~4.2s | ~18s |
| Answer Accuracy (vs policy) | ~9/10 correct | ~6/10 correct |
| Hallucination Rate | Very Low | Low-Medium |
| Table Parsing Accuracy | High | Moderate |
| Structured Output Quality | High | Medium |
| Fallback Frequency (incorrect) | ~1/10 | ~3/10 |

### Sample Answer Comparison

**Question:** "How many casual leaves does an employee get per year?"

**Cloud (Gemini) Answer:**
> According to the Innvonix HR Policy (Section 4.1 — Annual Leave Entitlement):
> - **Casual Leave (CL): 6 days per year**
> - Key points to note:
>   - CL is non-encashable.
>   - Unused CL lapses at the end of the calendar year (December 31).
>   - CL cannot be combined with sick leave to extend absences.

**Local (Llama3.2:1b) Answer:**
> I could not find a clear answer in the Innvonix HR Policy. Please contact hr@innvonix.com.

*(Note: The answer was present in the retrieved chunks — Chunk 1 contained the leave table clearly showing CL = 6 days. The 1B model failed to parse the markdown table format correctly.)*

---

**Question:** "What is the notice period for a Manager at Innvonix?"

**Cloud (Gemini) Answer:**
> Per Section 13.1 — Notice Period:
> - **Manager / Architect: 60 days notice period**
> - The notice period is effective from the date of written resignation.
> - Notice buyout is possible at 1 month's basic salary per 30 days, subject to manager and HR approval.

**Local (Llama3.2:1b) Answer:**
> According to the Innvonix HR Policy, the notice period for a Manager or Architect is **60 days**. This is effective from the date of written resignation submission.

*(Note: Llama3.2:1b answered correctly here — simpler factual extraction from a clear table.)*

---

## 7. Key Observations & Recommendations

### When to Use Cloud (Gemini)

| Scenario | Reason |
|----------|--------|
| Public HR chatbot / employee portal | Accuracy and structured output are critical |
| Complex multi-part HR questions | Better reasoning over long context |
| Questions involving policy tables | Superior markdown/table parsing |
| Rapid prototyping | Fast iteration without hardware setup |
| Low query volume | Cost per query is negligible |

### When to Use Local (Ollama)

| Scenario | Reason |
|----------|--------|
| Internal HR tool with sensitive data | No data leaves the premises |
| Air-gapped or restricted environments | No internet dependency |
| High-volume internal Q&A | Zero per-query cost |
| Developer local testing | Free, fast iteration |
| Regulated industries (BFSI, healthcare) | Data sovereignty compliance |

### Hallucination Analysis

Both models were grounded via prompt engineering with the rule:  
> "Answer ONLY from the policy context provided."

**Gemini:** Rarely hallucinated. When context was ambiguous, it correctly said "based on the provided policy" rather than inventing specifics.

**Llama3.2:1b:** Occasionally added policy details not present in the context when it did answer. However, it more commonly "over-refused" — falling back to the HR contact fallback even when the answer was in the chunks (false negative rather than hallucination).

### RAG System Advantages Observed

1. **Context grounding significantly reduces hallucination** compared to a raw LLM with no context.
2. **Similarity scores provide real-time interpretability** — low scores flag poor retrieval before the answer is generated.
3. **Section metadata** in chunks helps the LLM cite specific policy sections accurately.
4. **The same retrieval pipeline works for both models** — only the LLM node changes, making comparison and hot-swapping trivial.

---

## 8. How to Run

### Prerequisites

```bash
# Ollama must be running with required models
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2:1b        # local chat model

# API key for cloud model
echo 'GOOGLE_API_KEY="your_key_here"' >> .env
```

### Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Interactive mode — Cloud (Gemini)
python hr_qa_pipeline.py --model gemini

# Interactive mode — Local (Ollama)
python hr_qa_pipeline.py --model ollama

# Full benchmark comparison (both models, 10 questions)
python hr_qa_pipeline.py --compare

# Hide retrieved chunks (cleaner output)
python hr_qa_pipeline.py --model gemini --no-chunks
```

### Interactive Commands

| Command | Action |
|---------|--------|
| Type your question | Ask anything about Innvonix HR policy |
| `samples` | Show 10 pre-built sample questions |
| `1` – `10` | Shortcut to run a sample question by number |
| `chunks` | Toggle chunk display on/off |
| `quit` or `q` | Exit the bot |

### Project File Structure

```
RAG/
├── hr_qa_pipeline.py          ← Main RAG pipeline (this project)
├── innvonix_hr_policy.md      ← HR knowledge base document
├── hr_qa_architecture_doc.md  ← This documentation file
├── step1_transcript.py        ← YouTube transcript loader (previous)
├── step2_chunking.py          ← Chunking experiments (previous)
├── step3_vector_db.py         ← FAISS vector DB builder (previous)
├── step4_rag.py               ← General RAG chain (previous)
├── .env                       ← API keys (never commit!)
└── requirements.txt           ← Python dependencies
```

---

*Documentation generated for Innvonix Technologies HR Policy Q&A Bot v1.0*  
*RAG Pipeline built with LangChain + FAISS + Google Gemini + Ollama*
