"""
FastAPI Server for YouTube RAG Interactive Chatbot.
Provides REST and Server-Sent Events (SSE) streaming endpoints for the Web UI.
Imports existing modules without modifying them.
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, parse_qs

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# Import existing project modules directly
from step1_transcript import load_youtube_transcript, get_video_id
from step2_chunking import split_documents
from step3_vector_db import create_vector_db
from step4_rag import (
    build_rag_chain,
    format_docs,
    RAG_PROMPT_TEMPLATE,
    DEFAULT_LLM_MODEL,
    get_chat_model
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

AVAILABLE_CLOUD_MODELS = [
    {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash (Fast & Cost-Effective)", "provider": "Google Gemini"},
    {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro (High Reasoning)", "provider": "Google Gemini"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini (Fast & Smart)", "provider": "OpenAI"},
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="YouTube RAG Web UI", version="1.0.0")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory session state
class SessionState:
    def __init__(self):
        self.video_url: Optional[str] = None
        self.video_id: Optional[str] = None
        self.chunks: List[Any] = []
        self.vector_db: Optional[Any] = None
        self.retriever: Optional[Any] = None
        self.rag_chain: Optional[Any] = None
        self.llm_model: str = DEFAULT_LLM_MODEL
        self.char_count: int = 0
        self.preview_snippet: str = ""
        self.is_loaded: bool = False

    def reset(self):
        self.video_url = None
        self.video_id = None
        self.chunks = []
        self.vector_db = None
        self.retriever = None
        self.rag_chain = None
        self.char_count = 0
        self.preview_snippet = ""
        self.is_loaded = False

session = SessionState()

# Request schemas
class LoadVideoRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    chunk_size: int = Field(500, description="Chunk size in characters")
    chunk_overlap: int = Field(100, description="Chunk overlap in characters")
    model: Optional[str] = Field(None, description="Ollama model to use")

class ChatRequest(BaseModel):
    question: str = Field(..., description="User question")
    model: Optional[str] = Field(None, description="Ollama model override")
    k: int = Field(3, description="Number of context chunks to retrieve")

def get_installed_ollama_models() -> List[str]:
    """Fetch installed Ollama models from the local daemon."""
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=3)
        if res.status_code == 200:
            data = res.json()
            models = [m.get("name") for m in data.get("models", [])]
            return [m for m in models if not m.startswith("nomic-embed-text")]
    except Exception:
        pass
    return [DEFAULT_LLM_MODEL]

@app.get("/api/status")
async def get_status():
    """Return current server and video index status, including local and cloud models."""
    available_models = get_installed_ollama_models()
    has_gemini = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "is_loaded": session.is_loaded,
        "video_url": session.video_url,
        "video_id": session.video_id,
        "total_chunks": len(session.chunks),
        "char_count": session.char_count,
        "preview_snippet": session.preview_snippet,
        "current_model": session.llm_model,
        "available_models": available_models,
        "cloud_models": AVAILABLE_CLOUD_MODELS,
        "api_keys": {
            "gemini": has_gemini,
            "openai": has_openai
        },
        "thumbnail_url": f"https://img.youtube.com/vi/{session.video_id}/hqdefault.jpg" if session.video_id else None
    }

@app.post("/api/load-video")
async def load_video(req: LoadVideoRequest):
    """Load transcript for a YouTube video, split into chunks, and build FAISS vector index."""
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="YouTube URL cannot be empty.")
    
    try:
        vid_id = get_video_id(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid YouTube URL format: {str(e)}")

    model_to_use = req.model or session.llm_model or DEFAULT_LLM_MODEL

    try:
        # Load transcript using existing step1_transcript
        docs = load_youtube_transcript(url)
        if not docs:
            raise HTTPException(
                status_code=404,
                detail="No transcripts found for this video. Captions might be disabled or unavailable in supported languages."
            )

        total_chars = len(docs[0].page_content) if docs else 0
        first_doc = docs[0].page_content[:300].strip().replace("\n", " ")

        # Chunk using existing step2_chunking
        chunks = split_documents(docs, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to produce chunks from transcript.")

        # Build vector database using existing step3_vector_db
        vector_db = create_vector_db(chunks)

        # Build RAG chain using existing step4_rag
        rag_chain, retriever = build_rag_chain(vector_db, llm_model=model_to_use)

        # Update global session state
        session.video_url = url
        session.video_id = vid_id
        session.chunks = chunks
        session.vector_db = vector_db
        session.retriever = retriever
        session.rag_chain = rag_chain
        session.llm_model = model_to_use
        session.char_count = total_chars
        session.preview_snippet = first_doc
        session.is_loaded = True

        return {
            "status": "success",
            "message": f"Successfully indexed {len(chunks)} chunks from video transcript.",
            "video_id": vid_id,
            "video_url": url,
            "total_chunks": len(chunks),
            "char_count": total_chars,
            "preview_snippet": first_doc,
            "thumbnail_url": f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg",
            "model": model_to_use
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing video: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming question answering using Server-Sent Events (SSE).
    Emits source chunks first, followed by live LLM tokens, matching ChatGPT's streaming UX.
    """
    if not session.is_loaded or session.retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No video transcript has been loaded. Please load a YouTube video first."
        )

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    model_to_use = req.model or session.llm_model or DEFAULT_LLM_MODEL

    async def event_generator():
        try:
            # 1. Retrieve context chunks
            retrieved_docs = session.retriever.invoke(question)
            sources = [
                {
                    "rank": i + 1,
                    "content": doc.page_content.strip(),
                    "metadata": doc.metadata
                }
                for i, doc in enumerate(retrieved_docs)
            ]

            # Send sources payload first
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            await asyncio.sleep(0.02)

            # 2. Format prompt
            context_text = format_docs(retrieved_docs)
            prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
            formatted_messages = prompt.format_messages(context=context_text, question=question)

            # 3. Stream from dynamic LLM (Local Ollama or Cloud)
            llm = get_chat_model(model_name=model_to_use, temperature=0.1)

            # LangChain streaming generator
            for chunk in llm.stream(formatted_messages):
                content = chunk.content
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'token': content})}\n\n"
                    # Small yield to let async event loop breathe
                    await asyncio.sleep(0.005)

            # 4. Done event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/api/reset")
async def reset_session():
    """Reset current loaded video session."""
    session.reset()
    return {"status": "success", "message": "Session reset successfully."}

# Mount static files directory
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
async def serve_index():
    """Serve the single page ChatGPT-like interface."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "UI under preparation. Static files will be ready shortly."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
