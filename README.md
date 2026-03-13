# 🧠 Video Brain — Multi-Modal Video RAG System

> Chat with your videos using AI. Semantic search, visual reasoning, and natural language queries — all GPU-accelerated.

## Features

- **🎬 Video Upload** — Drag-and-drop any video (MP4, MKV, AVI, WebM)
- **🎤 Auto-Transcription** — GPU-accelerated Whisper with word-level timestamps
- **🖼️ Visual Search** — CLIP cross-modal search: type text, find matching frames
- **🔊 Transcript Search** — Semantic search through spoken content
- **🔀 Hybrid Search** — Reciprocal Rank Fusion combining visual + text results
- **🧠 AI Chat** — SmolVLM multi-frame reasoning with timestamp citations
- **⏱ Deep Linking** — Click any timestamp to jump the video player instantly
- **📝 Full Transcript** — Clickable, synchronized transcript alongside the player

## Architecture

```
User → React Frontend → FastAPI Server → AI Engine → {Whisper, CLIP, SmolVLM, Qdrant}
```

| Module | Tech | Purpose |
|--------|------|---------|
| Ingestion | FFmpeg | Frame extraction (scene detection), audio extraction |
| Transcription | Whisper (GPU) | Timestamped speech-to-text |
| Visual Embedding | CLIP ViT-B/32 | Image → 512-dim vectors |
| Vector Store | Qdrant | Hybrid vector search with filtering |
| Reasoning | SmolVLM | Multi-frame visual question answering |
| Server | FastAPI | REST API with background processing |
| Frontend | React + Vite | Premium dark UI with video player |

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker (for Qdrant)
- FFmpeg (`choco install ffmpeg` on Windows)
- NVIDIA GPU with CUDA (recommended)

### 1. Start Qdrant Vector Database
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Install Python Dependencies
```bash
# From project root (Video_based_rag/)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

### 3. Start FastAPI Server
```bash
# From project root
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start React Frontend
```bash
cd frontend
npm install
npm run dev
```

### 5. Open in browser
```
http://localhost:3000
```

## Project Structure

```
Video_based_rag/
├── server/                    # 🖥️ FastAPI API Server
│   ├── main.py                # App entry point + middleware
│   ├── config.py              # GPU-optimized configuration
│   ├── models/
│   │   └── schemas.py         # Pydantic request/response models
│   └── routes/
│       ├── ingest.py          # Upload + pipeline endpoints
│       ├── search.py          # Search endpoints
│       └── chat.py            # Chat endpoints
│
├── ai_engine/                 # 🤖 AI/ML Processing Engine
│   ├── ingestion.py           # FFmpeg frame/audio extraction
│   ├── transcription.py       # Whisper ASR (GPU)
│   ├── embeddings.py          # CLIP visual embeddings
│   ├── vector_store.py        # Qdrant operations
│   ├── search.py              # Hybrid search with RRF
│   └── reasoning.py           # SmolVLM multi-frame VLM
│
├── frontend/                  # ⚛️ React Frontend
│   └── src/
│       ├── components/        # VideoPlayer, ChatWindow, SearchBar, UploadZone
│       ├── pages/             # HomePage, VideoPage
│       └── services/api.js    # Backend API client
│
├── data/                      # Runtime data (gitignored)
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables
└── docker-compose.yml         # Docker deployment
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/videos/upload` | Upload video, start processing |
| `GET` | `/api/videos/{id}/status` | Check processing progress |
| `GET` | `/api/videos` | List all videos |
| `POST` | `/api/search` | Semantic search (visual/text/hybrid) |
| `POST` | `/api/chat` | Chat with video (VLM reasoning) |
| `GET` | `/api/videos/{id}/transcript` | Get full transcript |

## Tech Stack

- **Server**: Python, FastAPI, PyTorch (CUDA)
- **AI Engine**: Whisper (ASR), CLIP ViT-B/32 (Vision), SmolVLM (Reasoning)
- **Vector DB**: Qdrant (Docker)
- **Frontend**: React 18, Vite, Video.js
- **Search**: Reciprocal Rank Fusion (RRF) hybrid algorithm
