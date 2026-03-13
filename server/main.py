"""
Video Brain — FastAPI Application Entry Point
Run: uvicorn server.main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from server.config import CORS_ORIGINS, DATA_DIR
from server.routes import ingest, search, chat


# =====================================================
# LIFESPAN (startup / shutdown)
# =====================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 50)
    print("🧠  VIDEO BRAIN — Starting Up...")
    print("=" * 50)

    from ai_engine.vector_store import ensure_collections
    ensure_collections()

    print("✅ Server ready!\n")
    yield
    print("\n🛑 Video Brain shutting down...")


# =====================================================
# APP INSTANCE
# =====================================================
app = FastAPI(
    title="Video Brain API",
    description="Multi-Modal Video RAG System — Chat with your videos using AI",
    version="1.0.0",
    lifespan=lifespan
)

# =====================================================
# MIDDLEWARE
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# STATIC FILES (serve extracted frames + uploaded videos)
# =====================================================
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "uploads").mkdir(exist_ok=True)
(DATA_DIR / "frames").mkdir(exist_ok=True)

app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# =====================================================
# ROUTES
# =====================================================
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    return {
        "app": "Video Brain",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
