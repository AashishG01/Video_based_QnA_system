# Video Brain — Full Implementation Plan

## Current State Assessment

You have **7 standalone Python scripts** that demonstrate individual pieces of the pipeline:

| Script | What It Does | Status |
|---|---|---|
| [ingest.py](file:///c:/Users/Hp/Desktop/Video_based_rag/ingest.py) | FFmpeg frame/audio extraction + Whisper transcription | ✅ Working |
| [indexer.py](file:///c:/Users/Hp/Desktop/Video_based_rag/indexer.py) | Text indexing with hardcoded sample data | ⚠️ Demo only |
| [index_real_audio.py](file:///c:/Users/Hp/Desktop/Video_based_rag/index_real_audio.py) | Real audio → Whisper → Qdrant indexing | ✅ Working |
| [index_image.py](file:///c:/Users/Hp/Desktop/Video_based_rag/index_image.py) | CLIP visual embedding of frames → Qdrant | ✅ Working |
| [search.py](file:///c:/Users/Hp/Desktop/Video_based_rag/search.py) | Text-based transcript search | ✅ Working |
| [search_image.py](file:///c:/Users/Hp/Desktop/Video_based_rag/search_image.py) | CLIP text-to-image visual search | ✅ Working |
| [chat.py](file:///c:/Users/Hp/Desktop/Video_based_rag/chat.py) | SmolVLM single-frame reasoning | ✅ Working |

**Key Gaps**: No project structure, no API layer, no frontend, no hybrid search, no multi-frame reasoning, hardcoded values everywhere, no error handling, no configuration.

---

## Phase 1: Project Restructuring & Configuration

> **Goal**: Transform loose scripts into a professional, modular Python package.

### 1.1 Target Directory Structure

```
Video_based_rag/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application entry point
│   │   ├── config.py                # All configuration (env vars, paths, model names)
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── schemas.py           # Pydantic models (request/response)
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── ingestion.py         # Module A: Frame/Audio extraction
│   │   │   ├── transcription.py     # Module A: Whisper ASR
│   │   │   ├── embeddings.py        # Module B: CLIP + Text embedding
│   │   │   ├── vector_store.py      # Module B: Qdrant operations
│   │   │   ├── search.py            # Hybrid search (text + visual)
│   │   │   └── reasoning.py         # Module C: SmolVLM reasoning
│   │   └── api/
│   │       ├── __init__.py
│   │       ├── routes_ingest.py     # POST /api/videos/upload
│   │       ├── routes_search.py     # POST /api/search
│   │       └── routes_chat.py       # POST /api/chat
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VideoPlayer.jsx       # Video.js player with timestamp jumping
│   │   │   ├── ChatWindow.jsx        # Chat interface
│   │   │   ├── SearchBar.jsx         # Semantic search input
│   │   │   ├── ResultCard.jsx        # Search result with thumbnail + timestamp
│   │   │   ├── UploadZone.jsx        # Drag-and-drop video upload
│   │   │   └── Sidebar.jsx           # Video library sidebar
│   │   ├── pages/
│   │   │   ├── HomePage.jsx
│   │   │   └── VideoPage.jsx         # Main video analysis page
│   │   ├── hooks/
│   │   │   └── useVideoPlayer.js     # Custom hook for player control
│   │   ├── services/
│   │   │   └── api.js                # Axios/fetch wrapper for backend
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── data/                             # Runtime data (gitignored)
│   ├── uploads/                      # Uploaded video files
│   ├── frames/                       # Extracted frames per video
│   └── audio/                        # Extracted audio per video
├── docker-compose.yml                # Qdrant + Backend + Frontend
├── .gitignore
└── README.md
```

### 1.2 Configuration System (`config.py`)

```python
# All magic numbers and paths centralized here
class Settings:
    # Paths
    UPLOAD_DIR = "data/uploads"
    FRAMES_DIR = "data/frames"
    AUDIO_DIR = "data/audio"

    # Qdrant
    QDRANT_HOST = "localhost"
    QDRANT_PORT = 6333
    FRAMES_COLLECTION = "video_frames"
    TRANSCRIPT_COLLECTION = "video_transcript"

    # Models
    WHISPER_MODEL = "base"              # Options: tiny, base, small, medium, large
    WHISPER_DEVICE = "cpu"              # Options: cpu, cuda
    WHISPER_COMPUTE = "int8"            # Options: int8, float16, float32
    CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
    CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"
    VLM_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"

    # Processing
    FRAME_RATE = 1                      # Frames per second to extract
    MAX_SEARCH_RESULTS = 5
    VLM_MAX_TOKENS = 256
```

---

## Phase 2: Backend Services (Module-by-Module)

### 2.1 Module A — Ingestion Service (`services/ingestion.py`)

**Refactor from**: [ingest.py](file:///c:/Users/Hp/Desktop/Video_based_rag/ingest.py)

**What to build**:
- `extract_frames(video_path, output_dir, fps=1)` → Returns list of frame paths with timestamps
- `extract_audio(video_path, output_path)` → Returns audio file path
- Add **scene change detection** mode (FFmpeg `select='gt(scene,0.3)'`) as an alternative to fixed FPS
- Add video metadata extraction (duration, resolution, codec) using `ffprobe`
- Create per-video subdirectories: `data/frames/{video_id}/`, `data/audio/{video_id}/`

**Key improvement over current code**: Dynamic frame extraction reduces redundant frames in static scenes.

```python
# Scene change detection command
ffmpeg -i input.mp4 -vf "select='gt(scene,0.3)',showinfo" -vsync vfr output_%04d.jpg
```

### 2.2 Module A — Transcription Service (`services/transcription.py`)

**Refactor from**: [ingest.py](file:///c:/Users/Hp/Desktop/Video_based_rag/ingest.py) (transcribe_audio function)

**What to build**:
- [transcribe(audio_path)](file:///c:/Users/Hp/Desktop/Video_based_rag/ingest.py#56-110) → Returns `List[TranscriptSegment]`
- `TranscriptSegment` dataclass: `start_time`, `end_time`, `text`, `confidence`
- Add **word-level timestamps** (`word_timestamps=True` in Whisper) for more precise linking
- Add language detection and multi-language support
- Save transcript as both JSON and SRT format

**Key improvement**: Word-level timestamps allow pinpointing exact moments, not just segment-level.

### 2.3 Module B — Embedding Service (`services/embeddings.py`)

**Refactor from**: [index_image.py](file:///c:/Users/Hp/Desktop/Video_based_rag/index_image.py), [indexer.py](file:///c:/Users/Hp/Desktop/Video_based_rag/indexer.py)

**What to build**:
- `embed_frames(frame_paths)` → Returns list of 512-dim vectors (CLIP)
- `embed_text(text_chunks)` → Returns list of text vectors
- `embed_query_for_visual(query)` → CLIP text encoder for cross-modal search
- **Batch processing** for efficiency (process N frames at a time instead of all at once)
- **Progress callbacks** for frontend progress bars

```python
# Batch embedding for memory efficiency
def embed_frames(frame_paths: list, batch_size: int = 16):
    for i in range(0, len(frame_paths), batch_size):
        batch = frame_paths[i:i + batch_size]
        yield from image_model.embed(batch)
```

### 2.4 Module B — Vector Store Service (`services/vector_store.py`)

**Refactor from**: [index_image.py](file:///c:/Users/Hp/Desktop/Video_based_rag/index_image.py), [indexer.py](file:///c:/Users/Hp/Desktop/Video_based_rag/indexer.py)

**What to build**:
- `create_collections()` → Initialize both [video_frames](file:///c:/Users/Hp/Desktop/Video_based_rag/search_image.py#4-37) and `video_transcript` collections
- [index_frames(video_id, frame_paths, embeddings, timestamps)](file:///c:/Users/Hp/Desktop/Video_based_rag/index_image.py#5-65) → Store visual vectors
- [index_transcript(video_id, segments, embeddings)](file:///c:/Users/Hp/Desktop/Video_based_rag/indexer.py#3-42) → Store text vectors
- Add **video_id filtering** — every point gets a `video_id` payload field so you can search within a specific video
- Add **collection management** — delete/recreate per video on re-upload

**Key improvement**: `video_id` payload enables multi-video support with filtered search.

```python
# Qdrant payload filter for video-specific search
from qdrant_client.models import Filter, FieldCondition, MatchValue

video_filter = Filter(
    must=[FieldCondition(key="video_id", match=MatchValue(value="abc123"))]
)
```

### 2.5 Module B — Hybrid Search Service (`services/search.py`)

**Refactor from**: [search.py](file:///c:/Users/Hp/Desktop/Video_based_rag/search.py), [search_image.py](file:///c:/Users/Hp/Desktop/Video_based_rag/search_image.py)

**What to build**:
- `search_visual(query, video_id, limit)` → CLIP text→image search
- [search_transcript(query, video_id, limit)](file:///c:/Users/Hp/Desktop/Video_based_rag/search.py#3-28) → Text semantic search
- `hybrid_search(query, video_id, limit)` → **Fuse both results using Reciprocal Rank Fusion (RRF)**
- Return unified `SearchResult` objects with: `timestamp`, `score`, `source_type` (visual/text/hybrid), `frame_path`, `transcript_text`

**Reciprocal Rank Fusion** combines rankings from visual and text search:
```python
def rrf_fuse(visual_results, text_results, k=60):
    """Combine visual + text search results by timestamp proximity"""
    scores = {}
    for rank, result in enumerate(visual_results):
        ts = result.timestamp
        scores[ts] = scores.get(ts, 0) + 1 / (k + rank + 1)
    for rank, result in enumerate(text_results):
        ts = result.timestamp
        scores[ts] = scores.get(ts, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 2.6 Module C — Reasoning Service (`services/reasoning.py`)

**Refactor from**: [chat.py](file:///c:/Users/Hp/Desktop/Video_based_rag/chat.py)

**What to build**:
- `reason_about_frames(question, frame_paths, transcript_context)` → AI answer
- **Multi-frame reasoning**: Send top 3-5 frames (not just 1) to VLM for better answers
- **Context injection**: Append relevant transcript segments to the VLM prompt
- **Conversation history**: Maintain chat context for follow-up questions
- **Streaming support**: Yield tokens as they're generated for real-time UI updates

**Multi-frame prompt template**:
```python
messages = [
    {"role": "system", "content": "You are analyzing a video. Answer based on the frames and transcript provided."},
    {"role": "user", "content": [
        {"type": "image"},  # Frame 1
        {"type": "image"},  # Frame 2
        {"type": "image"},  # Frame 3
        {"type": "text", "text": f"""
Transcript context:
{transcript_text}

Question: {user_question}
Provide the answer with specific timestamps where relevant.
"""}
    ]}
]
```

---

## Phase 3: FastAPI Backend

### 3.1 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/videos/upload` | Upload video, trigger ingestion pipeline |
| `GET` | `/api/videos/{video_id}/status` | Check processing status (SSE/polling) |
| `GET` | `/api/videos` | List all processed videos |
| `POST` | `/api/search` | Hybrid semantic search |
| `POST` | `/api/chat` | Chat with video (VLM reasoning) |
| `GET` | `/api/videos/{video_id}/transcript` | Full transcript with timestamps |
| `GET` | `/api/frames/{video_id}/{frame_name}` | Serve extracted frame images |
| `GET` | `/api/videos/{video_id}/stream` | Stream/serve video file |

### 3.2 Request/Response Schemas (`models/schemas.py`)

```python
from pydantic import BaseModel
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str
    video_id: str
    search_type: str = "hybrid"  # "visual", "text", "hybrid"
    limit: int = 5

class SearchResult(BaseModel):
    timestamp: float
    score: float
    source_type: str        # "visual" | "text" | "hybrid"
    frame_url: Optional[str]
    transcript_text: Optional[str]
    thumbnail_url: Optional[str]

class ChatRequest(BaseModel):
    question: str
    video_id: str
    chat_history: List[dict] = []

class ChatResponse(BaseModel):
    answer: str
    referenced_timestamps: List[float]
    referenced_frames: List[str]

class VideoStatus(BaseModel):
    video_id: str
    status: str  # "uploading", "extracting", "transcribing", "indexing", "ready", "error"
    progress: float  # 0.0 to 1.0
    duration: Optional[float]
    frame_count: Optional[int]
```

### 3.3 Background Processing with FastAPI

Video ingestion is a long-running task. Use `BackgroundTasks` or a task queue:

```python
from fastapi import BackgroundTasks, UploadFile

@router.post("/api/videos/upload")
async def upload_video(file: UploadFile, background_tasks: BackgroundTasks):
    video_id = str(uuid4())
    save_path = f"data/uploads/{video_id}/{file.filename}"

    # Save uploaded file
    async with aiofiles.open(save_path, 'wb') as f:
        content = await file.read()
        await f.write(content)

    # Kick off pipeline in background
    background_tasks.add_task(run_full_pipeline, video_id, save_path)

    return {"video_id": video_id, "status": "processing"}
```

### 3.4 Full Pipeline Orchestrator

```python
async def run_full_pipeline(video_id: str, video_path: str):
    """The complete Module A → B pipeline"""
    update_status(video_id, "extracting", 0.1)

    # Module A: Extract
    frames = extract_frames(video_path, f"data/frames/{video_id}")
    audio = extract_audio(video_path, f"data/audio/{video_id}/audio.wav")
    update_status(video_id, "transcribing", 0.3)

    # Module A: Transcribe
    transcript = transcribe(audio)
    update_status(video_id, "indexing", 0.6)

    # Module B: Embed & Index
    frame_embeddings = list(embed_frames([f.path for f in frames]))
    index_frames(video_id, frames, frame_embeddings)

    text_embeddings = list(embed_text([s.text for s in transcript]))
    index_transcript(video_id, transcript, text_embeddings)

    update_status(video_id, "ready", 1.0)
```

---

## Phase 4: React Frontend

### 4.1 Pages

#### Home Page (`HomePage.jsx`)
- **Upload zone**: Drag-and-drop area with progress bar
- **Video library**: Grid of previously processed videos with thumbnails
- **Processing status**: Real-time status cards for videos being processed

#### Video Analysis Page (`VideoPage.jsx`)
- **Split layout**: Video player (left/top) + Chat interface (right/bottom)
- **Video player**: Video.js with custom controls
- **Search bar**: Semantic search with result cards below
- **Chat window**: Full conversational interface with timestamp citations

### 4.2 Key Components

#### `VideoPlayer.jsx` — The "Deep Linking" Player
```jsx
// Core feature: Jump to any timestamp programmatically
import videojs from 'video.js';

function VideoPlayer({ videoUrl, onTimeUpdate }) {
    const playerRef = useRef(null);

    // Expose jumpTo function to parent
    const jumpTo = (seconds) => {
        if (playerRef.current) {
            playerRef.current.currentTime(seconds);
            playerRef.current.play();
        }
    };

    return (
        <div>
            <video ref={videoRef} className="video-js" />
            {/* Timestamp markers overlay */}
        </div>
    );
}
```

#### `ChatWindow.jsx` — Conversational Interface
- Message bubbles with AI responses
- **Clickable timestamp badges** inside AI responses (e.g., `[01:23]`)
- Clicking a badge calls `VideoPlayer.jumpTo(83)` (83 seconds)
- Typing indicator during VLM inference
- Chat history maintained in state

#### `SearchBar.jsx` + `ResultCard.jsx`
- Debounced search input
- Result cards show: thumbnail (from extracted frames), timestamp, relevance score, source type indicator (🔊 audio / 🖼️ visual / 🔀 hybrid)
- Click on result → jump video player to that timestamp

#### `UploadZone.jsx`
- Drag-and-drop with file type validation ([.mp4](file:///c:/Users/Hp/Desktop/Video_based_rag/sample_video.mp4), `.mkv`, `.avi`, `.webm`)
- Upload progress bar using `XMLHttpRequest` for progress events
- Processing status polling after upload completes

### 4.3 API Service Layer (`services/api.js`)

```javascript
const API_BASE = 'http://localhost:8000/api';

export const api = {
    uploadVideo: (file, onProgress) => { /* multipart upload with progress */ },
    getVideoStatus: (videoId) => fetch(`${API_BASE}/videos/${videoId}/status`),
    searchVideo: (query, videoId, type) => { /* POST /api/search */ },
    chatWithVideo: (question, videoId, history) => { /* POST /api/chat */ },
    getTranscript: (videoId) => fetch(`${API_BASE}/videos/${videoId}/transcript`),
    listVideos: () => fetch(`${API_BASE}/videos`),
};
```

---

## Phase 5: Advanced Features (Taking It to the Next Level)

### 5.1 Scene Change Detection
Replace fixed 1 FPS with intelligent keyframe extraction:
```bash
ffmpeg -i input.mp4 -vf "select='gt(scene,0.3)',showinfo" -vsync vfr frame_%04d.jpg
```
This only extracts frames when the visual content **actually changes**, massively reducing redundancy.

### 5.2 Video Timeline Heatmap
- Track which timestamps the AI references most
- Overlay a "relevance heatmap" on the video player scrub bar
- Color-code by search result density

### 5.3 Transcript-Frame Alignment
- Each transcript segment already has `start_time` / `end_time`
- Each frame has a `timestamp`
- Create a **linked view**: clicking a transcript line highlights the corresponding frame and vice-versa
- Display synchronized transcript alongside the video player

### 5.4 Export & Sharing
- Export search results as JSON/CSV
- Generate video clips from timestamp ranges
- Share specific timestamped moments via URL (e.g., `/video/abc123?t=45`)

### 5.5 Multi-Video Search
- Search across ALL indexed videos at once
- Results grouped by video with relevance scoring
- Useful for the "5000 hours of lectures" use case

### 5.6 Quantization Benchmarking (Research)
- Benchmark SmolVLM in full precision vs 4-bit quantized (bitsandbytes)
- Track: inference time, answer quality (human eval), memory usage
- Generate comparison charts for the research paper

---

## Phase 6: Docker & Deployment

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    depends_on:
      - qdrant
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

volumes:
  qdrant_data:
```

---

## Implementation Order (Step-by-Step)

This is the recommended order to build, each step building on the previous:

### Step 1: Project Setup & Configuration
- [ ] Create directory structure
- [ ] Create `config.py` with all settings
- [ ] Create `requirements.txt` with all dependencies
- [ ] Set up `.env.example` and `.gitignore`
- [ ] Initialize `docker-compose.yml` with Qdrant

### Step 2: Refactor Ingestion (Module A)
- [ ] Move [ingest.py](file:///c:/Users/Hp/Desktop/Video_based_rag/ingest.py) logic → `services/ingestion.py`
- [ ] Move [transcribe_audio](file:///c:/Users/Hp/Desktop/Video_based_rag/ingest.py#56-110) → `services/transcription.py`
- [ ] Add per-video directory management
- [ ] Add video metadata extraction (`ffprobe`)
- [ ] Add scene change detection mode
- [ ] Add word-level timestamps in Whisper

### Step 3: Refactor Indexing (Module B)
- [ ] Move [index_image.py](file:///c:/Users/Hp/Desktop/Video_based_rag/index_image.py) → `services/embeddings.py`
- [ ] Move indexing logic → `services/vector_store.py`
- [ ] Add `video_id` payload field to all Qdrant points
- [ ] Add batch embedding for memory efficiency
- [ ] Implement hybrid search with Reciprocal Rank Fusion

### Step 4: Refactor Reasoning (Module C)
- [ ] Move [chat.py](file:///c:/Users/Hp/Desktop/Video_based_rag/chat.py) → `services/reasoning.py`
- [ ] Implement multi-frame reasoning (send top 5 frames)
- [ ] Add transcript context injection to VLM prompts
- [ ] Add conversation history support

### Step 5: Build FastAPI Backend
- [ ] Create `main.py` with CORS, static files, lifespan events
- [ ] Implement `/api/videos/upload` with background processing
- [ ] Implement `/api/videos/{id}/status` for progress tracking
- [ ] Implement `/api/search` with hybrid search
- [ ] Implement `/api/chat` with VLM reasoning
- [ ] Implement `/api/videos` listing and transcript endpoints
- [ ] Add error handling middleware

### Step 6: Build React Frontend
- [ ] Initialize Vite + React project
- [ ] Build `UploadZone` component with drag-and-drop
- [ ] Build `VideoPlayer` component with Video.js + timestamp jumping
- [ ] Build `ChatWindow` component with clickable timestamps
- [ ] Build `SearchBar` + `ResultCard` components
- [ ] Build `HomePage` with upload + video library
- [ ] Build `VideoPage` with split-panel layout
- [ ] Style everything with modern, premium design

### Step 7: Integration & Polish
- [ ] Connect frontend to backend API
- [ ] Add real-time processing status updates
- [ ] Add transcript-frame synchronized view
- [ ] Add video timeline heatmap
- [ ] Responsive design for mobile

### Step 8: Docker & Documentation
- [ ] Create backend `Dockerfile`
- [ ] Create frontend `Dockerfile`
- [ ] Finalize `docker-compose.yml`
- [ ] Write comprehensive `README.md`
- [ ] Record demo video/screenshots

---

## Verification Plan

### Automated Testing
```bash
# Backend unit tests
cd backend
pytest tests/ -v

# Test individual modules
python -c "from app.services.ingestion import extract_frames; print('Ingestion OK')"
python -c "from app.services.transcription import transcribe; print('Transcription OK')"
python -c "from app.services.embeddings import embed_frames; print('Embeddings OK')"
```

### API Testing
```bash
# Upload a video
curl -X POST http://localhost:8000/api/videos/upload -F "file=@sample_video.mp4"

# Check processing status
curl http://localhost:8000/api/videos/{video_id}/status

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is happening", "video_id": "...", "search_type": "hybrid"}'

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the person doing?", "video_id": "..."}'
```

### Manual Verification
1. **Upload Flow**: Upload [sample_video.mp4](file:///c:/Users/Hp/Desktop/Video_based_rag/sample_video.mp4) through the UI → verify progress bar → verify video appears in library
2. **Search Flow**: Type "What is being shown?" → verify results with thumbnails and timestamps → click result → verify video jumps to correct time
3. **Chat Flow**: Ask "What is happening in this video?" → verify AI response with timestamp citations → click timestamp → verify video jumps
4. **End-to-End**: Upload a new video → wait for processing → search + chat with it

---

## Technology Versions & Dependencies

```txt
# backend/requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
aiofiles==24.1.0

# AI/ML Models
faster-whisper==1.1.0
fastembed==0.4.0
transformers==4.46.0
torch==2.4.0
Pillow==10.4.0

# Vector Database
qdrant-client==1.12.0

# Utilities
python-dotenv==1.0.1
pydantic==2.9.0
```

```json
// frontend — key dependencies
{
  "react": "^18.3.0",
  "react-dom": "^18.3.0",
  "react-router-dom": "^6.26.0",
  "video.js": "^8.10.0",
  "axios": "^1.7.0"
}
```

> [!IMPORTANT]
> **Hardware Requirements**: This project runs on CPU but benefits greatly from a GPU. SmolVLM-256M works on CPU but is slow (~10-30s per inference). For faster inference, use `device="cuda"` if you have an NVIDIA GPU with ≥4GB VRAM. Whisper `base` model needs ~1GB RAM; `large` needs ~10GB.
