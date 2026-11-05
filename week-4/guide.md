# Week 4 – RAG Implementation Guide

## Overview

Week 4 implements complete RAG (Retrieval-Augmented Generation) systems. You'll build a YouTube video Q&A bot and a production HR policy assistant, comparing local vs cloud models, and deploy a web interface with streaming responses.

## Scripts

### YouTube RAG Pipeline (Step-by-Step)
- `step1_transcript.py` - YouTube transcript loading with multi-language support
- `step2_chunking.py` - Text chunking strategies and experiments
- `step3_vector_db.py` - Vector database creation with FAISS
- `step4_rag.py` - Complete RAG chain with multiple LLM providers

### HR Policy Q&A Bot (Production RAG)
- `hr_qa_pipeline.py` - Production HR policy Q&A with cloud/local model comparison
- `innvonix_hr_policy.md` - HR policy knowledge base
- `hr_qa_architecture_doc.md` - Comprehensive architecture documentation

### Web Interface
- `server.py` - FastAPI server with SSE streaming for web UI
- `static/` - Frontend assets for ChatGPT-like interface

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Environment variables in `.env`:
```env
GOOGLE_API_KEY=your_google_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Ollama Setup

Install and pull models for local inference:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama serve
```

## Key Concepts

- **RAG Pipeline**: Document loading → chunking → embeddings → vector store → retrieval → generation
- **Multi-Provider Architecture**: Local (Ollama) vs cloud (Gemini, OpenAI) models
- **Vector Databases**: FAISS for efficient similarity search with scores
- **Context Grounding**: Preventing hallucination with retrieved context
- **Web Interface**: FastAPI with SSE streaming for real-time responses

## Execution Order

### YouTube RAG Pipeline (Learning)
Run in sequence: step1_transcript → step2_chunking → step3_vector_db → step4_rag

### HR Policy Q&A Bot (Production)
```bash
python hr_qa_pipeline.py --model gemini   # Cloud model
python hr_qa_pipeline.py --model ollama   # Local model
python hr_qa_pipeline.py --compare        # Benchmark both
```

### Web Interface
```bash
python server.py
# Open browser to http://localhost:8000
```
