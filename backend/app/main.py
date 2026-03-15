"""
FastAPI Main Application
"""
import asyncio
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(name)s - %(message)s'
)

# Set HuggingFace mirror for users in restricted network environments BEFORE any model imports
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app.config import settings
from app.cache.warmer import CacheWarmer
from app.market import MarketDataService


# Global cache warmer instance
cache_warmer: CacheWarmer = None
_logger = logging.getLogger(__name__)


def _build_and_warmup_rag() -> None:
    """Build knowledge index if empty, then preload embedding models."""
    try:
        from app.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline()
        count = pipeline.collection.count()
        _logger.info(f"[RAG] ChromaDB doc count: {count}")

        if count == 0:
            _logger.info("[RAG] ChromaDB empty, building index from local documents...")
            docs = pipeline._local_documents
            if not docs:
                _logger.warning("[RAG] No local documents found to index")
                return

            def chunk_text(text, chunk_size=500, overlap=50):
                chunks, start = [], 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    chunk = text[start:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    start = end - overlap if end < len(text) else len(text)
                return chunks

            all_texts, all_metas, all_ids = [], [], []
            idx = 0
            for doc in docs:
                chunks = chunk_text(doc["content"])
                for i, chunk in enumerate(chunks):
                    all_texts.append(chunk)
                    all_metas.append({"source": doc["source"], "chunk_index": i})
                    all_ids.append(f"chunk_{idx}")
                    idx += 1

            batch_size = 100
            for i in range(0, len(all_texts), batch_size):
                pipeline.add_documents(
                    all_texts[i:i + batch_size],
                    all_metas[i:i + batch_size],
                    all_ids[i:i + batch_size],
                )
            _logger.info(f"[RAG] Index built: {pipeline.collection.count()} chunks from {len(docs)} docs")
        else:
            _logger.info(f"[RAG] Index already populated with {count} docs")

        # Warm up embedding models
        from app.rag.hybrid_pipeline import HybridRAGPipeline
        hybrid = HybridRAGPipeline()
        if hybrid.collection.count() > 0:
            hybrid._ensure_models()
            _logger.info("[RAG] Embedding + reranker models warmed up")
    except Exception as e:
        _logger.warning(f"[RAG] Build/warmup failed: {e}", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global cache_warmer

    if settings.CACHE_WARM_ENABLED:
        market_service = MarketDataService()
        cache_warmer = CacheWarmer(
            market_service=market_service,
            interval_seconds=settings.CACHE_WARM_INTERVAL_SECONDS,
            limit=settings.CACHE_WARM_LIMIT,
            concurrency=settings.CACHE_WARM_CONCURRENCY,
        )
        await cache_warmer.start_background_warming()

    # Build knowledge index (if empty) and warm up RAG models in background
    asyncio.create_task(asyncio.to_thread(_build_and_warmup_rag))

    yield

    # Shutdown: Stop cache warmer
    if cache_warmer:
        await cache_warmer.stop()


app = FastAPI(
    title="Financial Asset QA System",
    description="AI-powered financial asset question answering system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Financial Asset QA System",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
