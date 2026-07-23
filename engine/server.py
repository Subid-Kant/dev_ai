"""
FastAPI HTTP Server for DevAI — handles all API endpoints.
Supports: full response, streaming (SSE), health check, stats.
"""

import os
import json
import time
import asyncio
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.agent_tools import search_vector_db

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Your question")
    topic: Optional[str] = Field(None, description="Topic hint: python|android|ios|windows|mac|fullstack|networks|dbms|sql|sqlite|os|java")
    max_tokens: int = Field(512, ge=50, le=1024)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

class QueryResponse(BaseModel):
    answer: str
    topic: Optional[str]
    confidence: float
    retrieval_used: bool
    retrieved_chunks: list[str]
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str
    kb_chunks: int

# ─────────────────────────────────────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(pipeline) -> FastAPI:
    app = FastAPI(
        title="DevAI — Personal Developer AI",
        description="Custom-trained LLM for Python, Android, iOS, Windows development.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Endpoints ─────────────────────────────────────────────────────────────

    @app.get("/", response_class=JSONResponse)
    async def root():
        return {
            "name": "DevAI",
            "description": "Custom-trained LLM for developer knowledge",
            "topics": ["python", "android", "ios", "windows", "mac", "fullstack", "networks", "dbms", "sql", "sqlite", "os", "java"],
            "model_ready": pipeline.is_ready,
            "endpoints": {
                "POST /query": "Ask a question (full response)",
                "POST /query/stream": "Ask a question (streaming)",
                "GET /health": "Model + system status",
                "GET /stats": "Knowledge base statistics",
                "GET /topics": "List all topics",
            }
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="ready" if pipeline.is_ready else "not_ready",
            model_loaded=pipeline.is_ready,
            model_path=pipeline.lora_path,
            device=pipeline.device,
            kb_chunks=len(pipeline.retriever.chunks),
        )

    @app.get("/stats")
    async def stats():
        chunks_by_topic: dict[str, int] = {}
        for c in pipeline.retriever.chunks:
            chunks_by_topic[c.topic] = chunks_by_topic.get(c.topic, 0) + 1
        return {
            "knowledge_base": {
                "total_chunks": len(pipeline.retriever.chunks),
                "by_topic": chunks_by_topic,
                "vocab_terms": len(pipeline.retriever.idf),
            },
            "model": {
                "loaded": pipeline.is_ready,
                "device": pipeline.device,
                "max_new_tokens": pipeline.max_new_tokens,
            }
        }

    @app.get("/topics")
    async def topics():
        seen: dict[str, list[str]] = {}
        for c in pipeline.retriever.chunks:
            seen.setdefault(c.topic, [])
            if c.title not in seen[c.topic]:
                seen[c.topic].append(c.title)
        return {t: titles[:20] for t, titles in seen.items()}

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        if not pipeline.is_ready:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Run 'python scripts/train.py' first."
            )
        # Fetch real-time background knowledge from the crawler's ChromaDB
        background_context = search_vector_db(req.query, n_results=2)
        augmented_query = f"Background Memory Context:\n{background_context}\n\nUser Question:\n{req.query}"

        # Run in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.query(
                augmented_query,
                topic_hint=req.topic,
            )
        )
        return QueryResponse(**result)

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest):
        """Server-Sent Events streaming endpoint."""
        if not pipeline.is_ready:
            async def error_gen():
                yield f"data: {json.dumps({'error': 'Model not loaded'})}\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")

        async def generate_stream() -> AsyncGenerator[str, None]:
            # For now, run full generation then stream word-by-word
            # (true token streaming requires model modification)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.query(req.query, topic_hint=req.topic)
            )
            answer = result["answer"]
            # Send metadata first
            meta = {
                "type": "meta",
                "topic": result["topic"],
                "confidence": result["confidence"],
            }
            yield f"data: {json.dumps(meta)}\n\n"
            # Stream answer word by word
            words = answer.split(" ")
            for i, word in enumerate(words):
                chunk = {"type": "token", "text": word + (" " if i < len(words)-1 else "")}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.02)  # 50 words/sec stream rate
            # Send done
            done = {
                "type": "done",
                "tokens_generated": result["tokens_generated"],
                "time_seconds": result["time_seconds"],
            }
            yield f"data: {json.dumps(done)}\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    @app.get("/list")
    async def list_chunks(topic: Optional[str] = None):
        chunks = pipeline.retriever.chunks
        if topic:
            chunks = [c for c in chunks if c.topic == topic]
        return [{"id": c.id, "topic": c.topic, "title": c.title} for c in chunks[:100]]

    return app
