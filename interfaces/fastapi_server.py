"""
ScholaRAG FastAPI Server

A REST API for querying your Vector DB. Build your own frontend!

Usage:
    uvicorn fastapi_server:app --reload

Endpoints:
    POST /query      - Query papers
    GET /papers      - List all papers
    GET /stats       - Get statistics
    GET /health      - Health check

Security (v1.2.6):
    - API key authentication via X-API-Key header
    - Configurable CORS origins via ALLOWED_ORIGINS env var
    - Localhost binding by default (use 0.0.0.0 only if explicitly needed)

Author: ScholaRAG Team
License: MIT
"""

import os
import secrets
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from anthropic import Anthropic


# =============================================================================
# SECURITY CONFIGURATION (v1.2.6)
# =============================================================================

# API Key Authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Generate a default API key if not set (for development only)
# In production, ALWAYS set SCHOLARAG_API_KEY environment variable
DEFAULT_DEV_KEY = "dev-key-change-in-production"
API_KEY = os.getenv("SCHOLARAG_API_KEY", DEFAULT_DEV_KEY)

# CORS Configuration
# Default: localhost only. Set ALLOWED_ORIGINS for production
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000"
).split(",")


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from X-API-Key header.

    Security note: In production, always use a strong, randomly generated API key.
    """
    if API_KEY == DEFAULT_DEV_KEY:
        # Development mode: warn but allow
        print("⚠️  WARNING: Using default dev API key. Set SCHOLARAG_API_KEY in production!")
        return "dev"

    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Add X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return api_key


# Models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    citations: List[str]
    retrieved_papers: List[dict]
    query_time: str


class StatsResponse(BaseModel):
    total_papers: int
    collection_name: str
    db_path: str


# Initialize FastAPI
app = FastAPI(
    title="ScholaRAG API",
    description="REST API for querying your research papers",
    version="1.2.6"  # Updated with security fixes
)

# CORS middleware (v1.2.6: Configurable, no longer allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Configurable via ALLOWED_ORIGINS env var
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Only methods we actually use
    allow_headers=["X-API-Key", "Content-Type"],  # Only headers we need
)


# Global state
DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "papers")
vector_db = None
anthropic_client = None


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global vector_db, anthropic_client

    # Load Vector DB
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        vector_db = client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Loaded Vector DB: {vector_db.count()} papers")
    except Exception as e:
        print(f"❌ Error loading Vector DB: {e}")
        raise

    # Initialize Claude API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set")
        raise ValueError("ANTHROPIC_API_KEY required")

    anthropic_client = Anthropic(api_key=api_key)
    print("✅ Connected to Claude API")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ScholaRAG API",
        "version": "1.1.4",
        "endpoints": {
            "POST /query": "Query papers",
            "GET /papers": "List papers",
            "GET /stats": "Statistics",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vector_db": "connected" if vector_db else "disconnected",
        "claude_api": "connected" if anthropic_client else "disconnected"
    }


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics"""
    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector DB not initialized")

    return StatsResponse(
        total_papers=vector_db.count(),
        collection_name=COLLECTION_NAME,
        db_path=DB_PATH
    )


@app.post("/query", response_model=QueryResponse)
async def query_papers(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key)  # v1.2.6: Require API key
):
    """Query papers and generate answer (requires API key)"""

    if not vector_db or not anthropic_client:
        raise HTTPException(status_code=500, detail="Services not initialized")

    start_time = datetime.now()

    try:
        # Search Vector DB
        results = vector_db.query(
            query_texts=[request.question],
            n_results=request.top_k
        )

        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if 'metadatas' in results else [{}] * len(documents)

        if not documents:
            return QueryResponse(
                answer="No relevant papers found for this question.",
                citations=[],
                retrieved_papers=[],
                query_time=(datetime.now() - start_time).total_seconds()
            )

        # Build context
        context_parts = []
        papers_info = []

        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            paper_id = meta.get('paper_id', f'Paper_{i+1}')
            author = meta.get('author', 'Unknown')
            year = meta.get('year', 'N/A')
            title = meta.get('title', 'Untitled')

            context_parts.append(f"[{paper_id}] {author} ({year})\n{doc}\n")
            papers_info.append({
                "paper_id": paper_id,
                "author": author,
                "year": year,
                "title": title
            })

        context = "\n---\n".join(context_parts)

        # Generate answer with Claude
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=2048,
            temperature=request.temperature,
            messages=[{
                "role": "user",
                "content": f"""You are a research assistant. Answer based ONLY on these excerpts:

{context}

Question: {request.question}

Provide a clear answer with [Paper_ID] citations for every claim."""
            }]
        )

        answer = response.content[0].text

        # Extract citations
        import re
        citations = list(set(re.findall(r'\[([^\]]+)\]', answer)))

        query_time = (datetime.now() - start_time).total_seconds()

        return QueryResponse(
            answer=answer,
            citations=citations,
            retrieved_papers=papers_info,
            query_time=f"{query_time:.2f}s"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers")
async def list_papers(
    limit: int = 10,
    offset: int = 0,
    api_key: str = Depends(verify_api_key)  # v1.2.6: Require API key
):
    """List papers in knowledge base (requires API key)"""

    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector DB not initialized")

    try:
        # Get all documents (ChromaDB doesn't have pagination, so we get all)
        results = vector_db.get(
            limit=limit,
            offset=offset
        )

        papers = []
        metadatas = results.get('metadatas', [])

        for meta in metadatas:
            papers.append({
                "paper_id": meta.get('paper_id', 'Unknown'),
                "title": meta.get('title', 'Untitled'),
                "author": meta.get('author', 'Unknown'),
                "year": meta.get('year', 'N/A')
            })

        return {
            "total": vector_db.count(),
            "limit": limit,
            "offset": offset,
            "papers": papers
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # v1.2.6 Security: Bind to localhost by default
    # Set HOST=0.0.0.0 only if you need external access (not recommended)
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    if host == "0.0.0.0":
        print("⚠️  WARNING: Binding to 0.0.0.0 exposes the server externally!")
        print("   Make sure SCHOLARAG_API_KEY is set to a strong value.")

    if API_KEY == DEFAULT_DEV_KEY:
        print("\n" + "="*60)
        print("🔐 SECURITY: Using development API key")
        print("   Set SCHOLARAG_API_KEY for production use")
        print("="*60 + "\n")

    uvicorn.run(app, host=host, port=port)
