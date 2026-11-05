# Week 2 – LangChain Fundamentals Guide

## Overview

Week 2 covers LangChain fundamentals for building LLM-powered applications. You'll learn to work with multiple model providers, create reusable prompts, chain components together, and implement structured output.

## Scripts

- `llm_invocation.py` - Basic LLM invocation with Groq, Mistral, and Gemini
- `messages.py` - System and Human message types
- `prompt_templates.py` - Reusable prompt templates with variables
- `LCEL.py` - LangChain Expression Language for chaining
- `model_factory.py` - Provider abstraction pattern
- `employee_assistant.py` - Tool-based agent implementation
- `pydantic_structured_output.py` - Type-safe structured output
- `review_analysis_pipeline.py` - Capstone: Complete review analysis pipeline

## Setup

```bash
python3 -m pip install python-dotenv langchain-groq langchain-mistralai langchain-google-genai langchain-core pydantic
```

Environment variables in `.env`:
```env
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key
```

## Key Concepts

- **LLM Integration**: Multiple providers (Groq, Mistral, Gemini)
- **Messages**: System and Human message types
- **Prompts**: Reusable templates with variable interpolation
- **LCEL**: LangChain Expression Language for chaining components
- **Structured Output**: Pydantic models for type-safe responses
- **Batch Processing**: Efficient handling of multiple inputs

## Execution Order

Run scripts in sequence: llm_invocation → messages → prompt_templates → LCEL → model_factory → employee_assistant → pydantic_structured_output → review_analysis_pipeline

## Additional Setup

The `review_analysis_pipeline.py` requires `sample.json` in the week-2 directory with sample reviews. Creates an `output/` directory for results.
