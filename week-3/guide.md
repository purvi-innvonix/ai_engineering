# Week 3 – Prompt Engineering, RAG, and Local LLMs Guide

## Overview

Week 3 explores prompt engineering in depth, introduces vector databases and RAG concepts, and demonstrates local LLM usage with Ollama. You'll learn different prompt strategies, vector similarity search, and compare local vs cloud models.

## Scripts

- `prompt_experiment.py` - 4 prompt engineering strategies comparison
- `week2_pipeline_improvement.py` - Enhanced Week 2 prompt with detailed guidelines
- `ollama_experiment.py` - Local (Ollama) vs cloud (Groq) model comparison
- `faiss_example.py` - Vector similarity search with FAISS
- `json_grounded_experiment.py` - JSON output and context grounding
- `text_loader.py` - Document loading utilities
- `text_splitters.py` - Text chunking strategies
- `week_3.py` - Basic LLM invocation
- `RAG_Understanding.md` - Comprehensive RAG pipeline documentation

## Setup

```bash
python3 -m pip install python-dotenv langchain-groq langchain-core langchain-community langchain-ollama faiss-cpu numpy
```

Environment variables in `.env`:
```env
OPENAI_API_KEY=your_groq_api_key
GROQ_API_KEY=your_groq_api_key
```

## Ollama Setup

Install and start Ollama for local models:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama pull nomic-embed-text
ollama serve
```

## Key Concepts

- **Prompt Engineering**: Persona, structure, constraints, and format requirements
- **Vector Databases**: Embeddings, similarity search, FAISS
- **Local LLMs**: Ollama for privacy and cost savings
- **RAG Fundamentals**: Retrieval, augmentation, generation pipeline
- **Context Grounding**: Preventing hallucination with context constraints

## Execution Order

Run scripts in sequence: week_3 → prompt_experiment → week2_pipeline_improvement → json_grounded_experiment → text_loader → text_splitters → faiss_example → ollama_experiment → RAG_Understanding.md (read)
