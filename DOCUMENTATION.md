# 🧠 Video Brain — Complete Project Documentation

> **Everything you need to know** — every file, every function, every algorithm, every concept explained in full detail.

---

# Table of Contents
1. [What This Project Does](#1-what-this-project-does)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Concepts & Theory](#3-core-concepts--theory)
4. [Project Structure](#4-project-structure)
5. [File-by-File Documentation](#5-file-by-file-documentation)
   - [server/config.py](#serverconfpy)
   - [ai_engine/ingestion.py](#ai_engineingestionpy)
   - [ai_engine/transcription.py](#ai_enginetranscriptionpy)
   - [ai_engine/embeddings.py](#ai_engineembeddingspy)
   - [ai_engine/vector_store.py](#ai_enginevector_storepy)
   - [ai_engine/search.py](#ai_enginesearchpy)
   - [ai_engine/reasoning.py](#ai_enginereasoningpy)
   - [server/models/schemas.py](#servermodelsschemaspy)
   - [server/routes/ingest.py](#serverroutesingestpy)
   - [server/routes/search.py](#serverroutessearchpy)
   - [server/routes/chat.py](#serverrouteschatpy)
   - [server/main.py](#servermainpy)
   - [Frontend Components](#frontend)
6. [Data Flow — End to End](#6-data-flow--end-to-end)
7. [How to Run](#7-how-to-run)

---

# 1. What This Project Does

Video Brain is a **Multi-Modal Video RAG (Retrieval Augmented Generation) System**. In simple terms:

1. User **uploads a video**
2. System **extracts frames** (images) and **transcribes audio** (speech → text)
3. Both frames and transcript are converted to **vectors** (mathematical representations) and stored in a **vector database**
4. User can then **search** the video by typing natural language queries (e.g., "Show me when the car crashes")
5. User can **chat** with the video — ask questions and get AI-generated answers with exact timestamps

**The key innovation**: Unlike YouTube search (which only searches titles/tags), this searches the **actual visual content** and **actual spoken words** of the video using AI.

---

# 2. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                             │
│  Upload Video │ Search │ Chat │ Click Timestamps                  │
└──────┬───────────┬─────────┬──────────────────────────────────────┘
       │           │         │
       ▼           ▼         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND (:3000)                          │
│  UploadZone │ VideoPlayer │ SearchBar │ ChatWindow                │
└──────┬───────────┬─────────┬──────────────────────────────────────┘
       │           │         │          (Axios HTTP → /api/*)
       ▼           ▼         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FASTAPI SERVER (:8000)                          │
│  routes/ingest.py │ routes/search.py │ routes/chat.py             │
└──────┬───────────────────┬─────────┬─────────────────────────────┘
       │                   │         │
       ▼                   ▼         ▼
┌──────────────────────────────────────────────────────────────────┐
│                       AI ENGINE                                    │
│  ingestion.py → transcription.py → embeddings.py → vector_store   │
│                    search.py ← reasoning.py                        │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  QDRANT (:6333) │
              │  Vector Database │
              └─────────────────┘
```

**Two top-level Python packages**:
- `server/` — HTTP API layer (receives requests, sends responses)
- `ai_engine/` — AI/ML processing (the actual intelligence)

---

# 3. Core Concepts & Theory

## 3.1 What is an Embedding / Vector?

An **embedding** is a way to represent data (text, images, audio) as a list of numbers (a vector). For example:

```
"A cat sitting on a mat" → [0.12, -0.45, 0.78, ..., 0.33]  (512 numbers)
[image of a cat]          → [0.15, -0.42, 0.76, ..., 0.31]  (512 numbers)
```

**Why?** Because similar things have similar numbers. The cat text and cat image will have vectors that are mathematically close to each other. This is how the system answers "show me a cat" by searching images.

**Distance metric used**: **Cosine Similarity** — measures the angle between two vectors. Score of 1.0 = identical, 0.0 = completely different.

```
Cosine Similarity = (A · B) / (|A| × |B|)
```

## 3.2 What is CLIP?

**CLIP (Contrastive Language-Image Pre-Training)** is an AI model by OpenAI that learns to understand images and text **in the same mathematical space**.

- It has two halves:
  - **Vision Encoder** (ViT-B/32): Takes an image → outputs 512-dim vector
  - **Text Encoder**: Takes text → outputs 512-dim vector
- Both vectors live in the **same space**, so you can compare them directly

**How it was trained**: On 400 million image-text pairs from the internet. It learned that a photo of a dog and the text "a dog" should have similar vectors.

**In our project**:
- `CLIP Vision Encoder` → converts video frames to vectors (stored in Qdrant)
- `CLIP Text Encoder` → converts user search queries to vectors (used for search)
- We use the `Qdrant/clip-ViT-B-32-vision` and `Qdrant/clip-ViT-B-32-text` variants via FastEmbed

## 3.3 What is Whisper?

**Whisper** is OpenAI's speech recognition model. It converts audio → text with timestamps.

| Model | Parameters | RAM | Speed | Accuracy |
|-------|-----------|-----|-------|----------|
| tiny | 39M | ~1GB | Fastest | OK |
| base | 74M | ~1GB | Fast | Good |
| small | 244M | ~2GB | Medium | Better |
| medium | 769M | ~5GB | Slow | Great |
| large-v3 | 1.5B | ~10GB | Slowest | Best |

**We use `faster-whisper`** — a CTranslate2-optimized version that is 4x faster than the original.

**Key features we use**:
- `word_timestamps=True` — gives the exact time for each word (not just sentences)
- `vad_filter=True` — **Voice Activity Detection** — skips silent parts
- `beam_size=5` — searches 5 possible interpretations in parallel for better accuracy
- GPU acceleration with float16 precision

## 3.4 What is a Vector Database (Qdrant)?

A **vector database** is like a regular database, but optimized for storing and searching vectors using mathematical similarity instead of exact matching.

**Regular database**: `SELECT * FROM users WHERE name = "John"` (exact match)
**Vector database**: `Find the 5 vectors closest to this query vector` (similarity search)

**Qdrant** specifically:
- Stores vectors + metadata (payload) together
- Uses **HNSW (Hierarchical Navigable Small World)** index for fast search
- Supports **filtered search** — e.g., "find similar vectors, but only from video X"
- Runs as a Docker container on port 6333

**Our two collections**:
| Collection | Vector Size | Source | Stores |
|-----------|-------------|--------|--------|
| `video_frames` | 512-dim | CLIP Vision | Frame images with timestamps |
| `video_transcript` | 384-dim | FastEmbed (BGE) | Transcript text segments |

## 3.5 What is RAG (Retrieval Augmented Generation)?

**RAG** is a pattern where you:
1. **Retrieve** relevant information from a database (search step)
2. **Augment** a prompt with that information (context injection)
3. **Generate** an answer using an AI model (reasoning step)

**Without RAG**: AI can only use what it was trained on (often hallucinating)
**With RAG**: AI gets real data from your video, so answers are grounded in facts

**In our project**, the RAG flow is:
```
User asks: "What is happening at the beginning?"
    → RETRIEVE: Find top 5 matching frames + transcript from Qdrant
    → AUGMENT: Build a prompt with those frames + transcript text
    → GENERATE: SmolVLM looks at frames + reads transcript → writes answer
```

## 3.6 What is SmolVLM (Vision Language Model)?

**SmolVLM** is a small but capable **VLM (Vision Language Model)** by HuggingFace that can:
- **See** images (visual understanding)
- **Read** text prompts
- **Generate** text answers about what it sees

We use `SmolVLM-256M-Instruct` — only 256M parameters, optimized for edge/consumer hardware.

**Multi-frame reasoning**: Unlike basic approaches that analyze one frame, we send **5 frames at once** so the VLM can understand temporal context (what happened before/after).

## 3.7 What is Reciprocal Rank Fusion (RRF)?

When we search both visual frames AND transcript text, we get two separate ranked lists. **RRF** combines them into one:

```
RRF Score = Σ 1/(k + rank_i)   where k=60 (constant)
```

**Example**:
- Visual search: frame at 5s is rank 1 → score = 1/(60+1) = 0.0164
- Text search: transcript at 5s is rank 3 → score = 1/(60+3) = 0.0159
- **Combined score** for 5s = 0.0164 + 0.0159 = 0.0323

A timestamp that appears in **both** search results gets a higher combined score than one appearing in only one.

## 3.8 What is Scene Change Detection?

Instead of extracting a frame every second (which creates many duplicate/similar frames for static scenes), we use FFmpeg's **scene detection filter**:

```bash
ffmpeg -vf "select='gt(scene,0.3)'"
```

This computes a **scene change score (0.0-1.0)** for each frame. If the score exceeds our threshold (0.3), the scene has changed enough to warrant a new keyframe.

**Benefits**: A 60-second video might produce 60 frames at 1fps, but only 10-15 frames with scene detection — all of them visually distinct and meaningful.

## 3.9 Lazy Loading Pattern

All AI models (Whisper, CLIP, SmolVLM) use **lazy loading**:

```python
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = load_model()  # Heavy operation, only once
    return _model
```

**Why?** Loading a model into GPU memory takes 5-30 seconds. If we loaded all 3 models at server startup, the server would take minutes to start. Instead, each model loads only when first needed and stays in memory for subsequent requests.

---

# 4. Project Structure

```
Video_based_rag/
│
├── server/                         # 🖥️ FastAPI API Server
│   ├── __init__.py                 # Package marker
│   ├── config.py                   # All configuration (paths, models, GPU)
│   ├── main.py                     # FastAPI app entry point
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic request/response models
│   └── routes/
│       ├── __init__.py
│       ├── ingest.py               # /api/videos/* endpoints
│       ├── search.py               # /api/search endpoint
│       └── chat.py                 # /api/chat endpoint
│
├── ai_engine/                      # 🤖 AI/ML Processing Engine
│   ├── __init__.py                 # Package marker
│   ├── ingestion.py                # Module A: FFmpeg frame/audio extraction
│   ├── transcription.py            # Module A: Whisper speech-to-text
│   ├── embeddings.py               # Module B: CLIP visual embeddings
│   ├── vector_store.py             # Module B: Qdrant database operations
│   ├── search.py                   # Module B: Hybrid search (RRF fusion)
│   └── reasoning.py                # Module C: SmolVLM visual reasoning
│
├── frontend/                       # ⚛️ React Frontend
│   ├── index.html                  # HTML entry point
│   ├── package.json                # Node.js dependencies
│   ├── vite.config.js              # Vite build config + API proxy
│   └── src/
│       ├── main.jsx                # React entry point
│       ├── App.jsx                 # Root component + routing
│       ├── index.css               # Complete CSS design system
│       ├── services/
│       │   └── api.js              # Backend API client (Axios)
│       ├── components/
│       │   ├── UploadZone.jsx      # Drag-and-drop file upload
│       │   ├── VideoPlayer.jsx     # Video player with jumpTo()
│       │   ├── ChatWindow.jsx      # Conversational AI chat
│       │   └── SearchBar.jsx       # Semantic search with results
│       └── pages/
│           ├── HomePage.jsx        # Upload + video library
│           └── VideoPage.jsx       # Video analysis (player + chat + search)
│
├── data/                           # Runtime data (auto-created, gitignored)
│   ├── uploads/{video_id}/         # Uploaded video files
│   ├── frames/{video_id}/          # Extracted frame images
│   └── audio/{video_id}/           # Audio + transcript files
│
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
├── .gitignore                      # Git ignore rules
├── docker-compose.yml              # Docker deployment
└── README.md                       # Quick start guide
```

---

# 5. File-by-File Documentation

---

## `server/config.py`

**Purpose**: Single source of truth for ALL configuration. Every other file imports from here.

### What It Does

1. **Path Resolution** — Uses `Path(__file__).resolve().parent.parent` to find the project root, regardless of where the script is run from. Creates `data/uploads/`, `data/frames/`, `data/audio/` directories automatically.

2. **GPU Auto-Detection** — `torch.cuda.is_available()` checks if NVIDIA CUDA is available. If yes, uses `cuda` device with `float16` precision (2x faster, half memory). If not, falls back to `cpu` with `float32`.

3. **Environment Variables** — Uses `python-dotenv` to load `.env` file. Settings like `QDRANT_HOST`, `WHISPER_MODEL`, and `VLM_MODEL` can be overridden without changing code.

### Key Settings

| Setting | Value | Why |
|---------|-------|-----|
| `DEVICE` | `cuda` or `cpu` | Auto-detected GPU |
| `TORCH_DTYPE` | `float16` or `float32` | float16 = 2x faster on GPU |
| `WHISPER_COMPUTE` | `float16` or `int8` | GPU uses float16, CPU uses int8 (quantized) |
| `CLIP_VECTOR_SIZE` | 512 | CLIP ViT-B/32 output dimension |
| `TEXT_VECTOR_SIZE` | 384 | FastEmbed BGE-small output dimension |
| `SCENE_THRESHOLD` | 0.3 | Scene change sensitivity (0=all frames, 1=no frames) |
| `EMBEDDING_BATCH_SIZE` | 32 | Frames processed per CLIP batch |

---

## `ai_engine/ingestion.py`

**Purpose**: Break the "black box" of video into machine-readable data (frames + audio).

### Functions

#### `get_video_metadata(video_path) → dict`
- **Tool**: `ffprobe` (part of FFmpeg)
- **Output**: `{"duration": 30.5, "width": 1920, "height": 1080, "fps": 29.97, "codec": "h264"}`
- **How**: Runs `ffprobe -print_format json -show_format -show_streams` which outputs JSON containing all video metadata
- **Frame rate parsing**: Handles both fraction format (`"30/1"`) and decimal format (`"29.97"`)

#### `extract_frames(video_path, video_id) → list[dict]`
- **Tool**: `ffmpeg`
- **Two modes**:

| Mode | FFmpeg Filter | When to Use |
|------|--------------|-------------|
| Scene Detection | `select='gt(scene,0.3)',showinfo` | Smart — only captures when scene changes |
| Fixed FPS | `fps=1` | Simple — one frame per second |

- **Scene detection works by**: FFmpeg calculates a pixel difference score between consecutive frames. If >30% of pixels changed, it's a new scene.
- **`-vsync vfr`**: Variable frame rate — only output frames that passed the filter (not empty placeholders)
- **`showinfo`**: Prints `pts_time` (presentation timestamp) to stderr, which we parse with regex
- **Output**: List of `{"path": "data/frames/abc123/frame_0001.jpg", "timestamp": 2.45}`

#### `extract_audio(video_path, video_id) → str`
- **FFmpeg flags explained**:
  - `-vn`: No video (discard video stream)
  - `-acodec pcm_s16le`: Uncompressed 16-bit PCM audio (Whisper requires this)
  - `-ar 16000`: 16kHz sample rate (Whisper's native rate — higher rates waste processing)
  - `-ac 1`: Mono channel (Whisper doesn't use stereo information)

---

## `ai_engine/transcription.py`

**Purpose**: Convert audio to timestamped text using Whisper ASR.

### Functions

#### `_get_whisper() → WhisperModel`
- **Lazy loading**: Model loads only on first call, stays in GPU memory
- **`compute_type`**: `float16` on GPU (fast), `int8` on CPU (quantized — smaller and faster than float32 on CPU)

#### `transcribe(audio_path, video_id) → list[dict]`
- **Input**: WAV audio file
- **Process**:
  1. Loads Whisper model (lazy)
  2. Runs transcription with `beam_size=5, word_timestamps=True, vad_filter=True`
  3. Iterates over segments, collecting text + start/end times + word-level times
  4. Saves as JSON and SRT

- **Key parameters**:
  - **`beam_size=5`**: Beam search — instead of taking the single best prediction at each step, keeps 5 candidates and picks the overall best sequence. Higher = more accurate but slower.
  - **`word_timestamps=True`**: Returns timing for each individual word, not just sentence-level. E.g., "The" at 1.20s, "cat" at 1.45s, "sat" at 1.67s.
  - **`vad_filter=True`**: **Voice Activity Detection** — uses an algorithm (Silero VAD) to detect which parts of the audio contain speech and skips silent segments. Reduces processing time significantly for videos with music/silence.

- **Output format**:
```json
{
  "start_time": 0.0,
  "end_time": 4.5,
  "text": "Welcome to the lecture on machine learning",
  "words": [
    {"word": "Welcome", "start": 0.0, "end": 0.42, "probability": 0.98},
    {"word": "to", "start": 0.42, "end": 0.55, "probability": 0.99}
  ]
}
```

#### `_save_srt(transcript, path)`
- Saves transcript in **SRT subtitle format** (standard format for subtitles)
- SRT timestamps: `HH:MM:SS,mmm` (e.g., `00:01:23,456`)
- Can be loaded into any video player (VLC, MPV, etc.)

---

## `ai_engine/embeddings.py`

**Purpose**: Convert images and text into mathematical vectors using CLIP.

### Functions

#### `embed_frames(frame_paths, batch_size=32) → list[list[float]]`
- **Model**: CLIP ViT-B/32 Vision Encoder
- **Input**: List of image file paths
- **Output**: List of 512-dimensional vectors (each vector = list of 512 floats)
- **Batch processing**: Instead of loading all images at once (OOM risk), processes 32 at a time
- **Each image goes through**:
  1. Resize to 224×224 pixels
  2. Normalize pixel values
  3. Pass through Vision Transformer (ViT) with 12 transformer layers
  4. Output: 512 numbers representing "what this image looks like"

#### `embed_query_visual(query) → list[float]`
- **Model**: CLIP ViT-B/32 Text Encoder
- **Input**: Text string (e.g., "person writing on whiteboard")
- **Output**: 512-dimensional vector
- **This is the magic of CLIP**: This text vector lives in the **same space** as the image vectors, so you can compute cosine similarity between them to find matching images

**Cross-modal search explained**:
```
Query: "red car" → [0.12, -0.34, ...]  (512-dim text vector)

Frame 1 (shows red car):  [0.11, -0.33, ...]  → Cosine similarity = 0.95 ✅
Frame 2 (shows blue sky):  [0.78, 0.22, ...]   → Cosine similarity = 0.12 ❌
```

---

## `ai_engine/vector_store.py`

**Purpose**: All database operations — create collections, store vectors, search vectors.

### Functions

#### `ensure_collections()`
- Creates two Qdrant collections if they don't exist:
  - `video_frames`: 512-dim vectors, Cosine distance
  - `video_transcript`: 384-dim vectors, Cosine distance
- Called at server startup (in `main.py` lifespan)

#### `delete_video_data(video_id)`
- Deletes all vectors for a specific video using **payload filtering**
- Uses `FieldCondition(key="video_id", match=MatchValue(value=video_id))`
- Called before re-indexing to avoid duplicates

#### `index_frames(video_id, frames, embeddings)`
- **What goes into Qdrant for each frame**:
  - `id`: Deterministic hash of `{video_id}_frame_{i}` (ensures no duplicates)
  - `vector`: 512-dim CLIP embedding
  - `payload`: `{video_id, timestamp, file_path, source, frame_index}`
- **Batch upsert**: Sends 100 points at a time to Qdrant (API limit efficiency)

#### `index_transcript(video_id, transcript)`
- Uses `client.add()` which **auto-embeds text** using Qdrant's built-in FastEmbed
- FastEmbed uses **BAAI/bge-small-en** model → 384-dim vectors
- Stores: document text, metadata (video_id, start_time, end_time, source)

**Why different vector sizes?** CLIP and BGE are different models optimized for different tasks:
- CLIP (512-dim): Trained on image-text pairs → best for cross-modal search
- BGE (384-dim): Trained on text pairs → best for text-to-text semantic search

#### `search_frames(query_vector, video_id, limit) → points`
- **Input**: CLIP text vector (from `embed_query_visual`)
- **Filter**: Only search within specific video using `video_id` payload filter
- **Returns**: Qdrant points sorted by cosine similarity (highest first)

#### `search_transcript(query_text, video_id, limit) → results`
- **Input**: Raw text (auto-embedded by Qdrant's FastEmbed)
- **Filter**: Video-specific filter
- **Returns**: Matching transcript segments with similarity scores

---

## `ai_engine/search.py`

**Purpose**: Combine visual and text search results using Reciprocal Rank Fusion.

### Functions

#### `visual_search(query, video_id, limit) → list[dict]`
1. Converts query text → CLIP 512-dim vector (via `embed_query_visual`)
2. Searches `video_frames` collection in Qdrant
3. Returns results with: timestamp, score, source_type="visual", frame_path

#### `text_search(query, video_id, limit) → list[dict]`
1. Searches `video_transcript` collection using Qdrant's auto-embedding
2. Returns results with: timestamp, score, source_type="text", transcript_text

#### `hybrid_search(query, video_id, limit) → list[dict]`
This is the most important function — combines both search types.

**Step-by-step**:
1. Fetch `2×limit` results from both visual and text search
2. Calculate RRF score for each unique timestamp:
   ```
   For each result at rank r: rrf_score += 1/(60 + r + 1)
   ```
3. If a timestamp appears in BOTH search results, its scores add up → ranked higher
4. Attach nearest frame to text-only results (so every result has a thumbnail)
5. Sort by combined score, return top `limit`

**Example walkthrough**:
```
Visual results:  [5.0s (rank 0), 12.0s (rank 1), 8.0s (rank 2)]
Text results:    [5.0s (rank 0), 3.0s (rank 1), 12.0s (rank 2)]

RRF Scores:
  5.0s  = 1/61 + 1/61  = 0.0328  ← HIGHEST (appears in both!)
  12.0s = 1/62 + 1/63  = 0.0320
  8.0s  = 1/63          = 0.0159
  3.0s  = 1/62          = 0.0161

Final ranking: [5.0s, 12.0s, 3.0s, 8.0s]
```

#### `_attach_nearest_frames(fused, visual_results)`
- For text-only results that don't have a `frame_path`, finds the nearest frame by timestamp
- Uses `min(visual_results, key=lambda v: abs(v["timestamp"] - ts))`

---

## `ai_engine/reasoning.py`

**Purpose**: Answer complex questions about video content using a Vision Language Model.

### Functions

#### `_get_vlm() → (processor, model)`
- Loads **SmolVLM-256M-Instruct** from HuggingFace
- `AutoProcessor`: Handles image preprocessing + text tokenization
- `AutoModelForImageTextToText`: The actual neural network
- **`torch_dtype=TORCH_DTYPE`**: float16 on GPU (halves memory usage)
- **`_attn_implementation="eager"`**: Uses standard attention (compatible with all hardware)

#### `reason_about_video(question, frame_paths, frame_timestamps, transcript_context) → dict`

**The full RAG pipeline**:

1. **Load images**: Opens each frame with PIL, converts to RGB
2. **Build multi-modal prompt**:
   ```
   [IMAGE_1] [IMAGE_2] [IMAGE_3] [IMAGE_4] [IMAGE_5]
   
   You are analyzing a video. You are shown 5 frames at timestamps:
   Frame 1 at 2.5s, Frame 2 at 5.0s, Frame 3 at 8.3s, ...
   
   Relevant transcript:
   [2.5s] The professor explains the formula...
   [5.0s] Now let's look at this example...
   
   Question: What is the professor explaining?
   Provide a detailed answer. Reference specific timestamps when relevant.
   ```
3. **Process through VLM**:
   - `processor.apply_chat_template()`: Formats into the model's expected chat format
   - `processor(text=..., images=..., return_tensors="pt")`: Tokenizes text + preprocesses images into tensors
   - `.to(DEVICE)`: Moves tensors to GPU
4. **Generate answer**:
   - `model.generate(max_new_tokens=512, do_sample=False)`: Generates response
   - `do_sample=False`: **Greedy decoding** — always picks the most probable next token (faster, deterministic)
   - `torch.no_grad()`: Disables gradient calculation (saves memory during inference)
5. **Decode**: `processor.batch_decode(skip_special_tokens=True)` converts token IDs back to text
6. **Extract timestamps**: Parses the answer for timestamp references

#### `_extract_timestamps(text, available_timestamps) → list[float]`
Uses regex to find timestamps in the AI's answer:
- `(\d+\.?\d*)\s*s(?:econds?)?` — matches "5.2s", "10 seconds"
- `(\d+):(\d{2})` — matches "01:23" format
- `[Ff]rame\s*(\d+)` — matches "Frame 3" and maps to actual timestamp

---

## `server/models/schemas.py`

**Purpose**: Define the exact shape of all API requests and responses using Pydantic.

### Why Pydantic?
- **Type validation**: If you send `limit: "abc"` instead of a number, it rejects immediately
- **Auto-documentation**: FastAPI generates Swagger docs from these schemas
- **Serialization**: Converts Python objects ↔ JSON automatically

### Key Models

| Model | Used In | Fields |
|-------|---------|--------|
| `SearchRequest` | POST /api/search | query, video_id, search_type, limit |
| `SearchResult` | Response | timestamp, score, source_type, frame_url, transcript_text |
| `ChatRequest` | POST /api/chat | question, video_id, chat_history |
| `ChatResponse` | Response | answer, referenced_timestamps, referenced_frames |
| `VideoInfo` | GET /api/videos | video_id, filename, status, progress, duration, frame_count |
| `ProcessingStatus` | Enum | uploading, extracting, transcribing, embedding, indexing, ready, error |

---

## `server/routes/ingest.py`

**Purpose**: Handle video uploads and orchestrate the full processing pipeline.

### In-Memory Status Tracking
```python
video_status: dict[str, dict] = {}
```
A dictionary that tracks the processing state of every video. Updated by `_update_status()` as the pipeline progresses.

### `run_full_pipeline(video_id, video_path)` — Background Task

This is the **heart of the system**. It runs asynchronously after upload.

```
Step 1: get_video_metadata()        → duration, resolution      [5%]
Step 2: delete_video_data()         → clean old data            [10%]
Step 3: extract_frames()            → JPG images + timestamps   [15-30%]
Step 4: extract_audio()             → WAV file                  [30%]
Step 5: transcribe()                → JSON + SRT transcript     [35-55%]
Step 6: embed_frames()              → CLIP vectors              [55-75%]
Step 7: index_frames()              → Store in Qdrant           [75-85%]
Step 8: index_transcript()          → Store in Qdrant           [85-100%]
```

**Lazy imports**: `from ai_engine.ingestion import ...` is inside the function body, not at the top of the file. This means AI models don't load until the first video is uploaded.

### Endpoints

| Endpoint | Method | What It Does |
|----------|--------|-------------|
| `/api/videos/upload` | POST | Saves file, starts pipeline in background, returns video_id |
| `/api/videos/{id}/status` | GET | Returns current processing status + progress % |
| `/api/videos` | GET | Lists all videos with their statuses |
| `/api/videos/{id}/transcript` | GET | Returns the full JSON transcript |

---

## `server/routes/search.py`

**Purpose**: Single endpoint that routes to visual, text, or hybrid search.

### Endpoint: `POST /api/search`
1. Validates video exists and `status == "ready"`
2. Based on `search_type`:
   - `"visual"` → calls `ai_engine.search.visual_search()`
   - `"text"` → calls `ai_engine.search.text_search()`
   - `"hybrid"` → calls `ai_engine.search.hybrid_search()` (default, recommended)
3. Returns list of `SearchResult` objects

---

## `server/routes/chat.py`

**Purpose**: Full RAG chat endpoint — the most complex API route.

### Endpoint: `POST /api/chat`

**Complete flow**:
1. Validate video is processed
2. Run `hybrid_search()` with the user's question → get top 5 results
3. Collect frame paths and transcript text from search results
4. Call `reason_about_video()` with:
   - Up to 5 frame images
   - Their timestamps
   - Transcript context (concatenated matching transcript lines)
5. Return AI answer + referenced timestamps + referenced frames

---

## `server/main.py`

**Purpose**: FastAPI application entry point — wires everything together.

### Key Components

- **Lifespan**: On startup, calls `ensure_collections()` to create Qdrant collections
- **CORS Middleware**: Allows frontend (port 3000/5173) to call backend (port 8000)
- **Static Files**: Mounts `/data` so frontend can directly load frame images and video files
- **Router Registration**: Includes all 3 route modules

### Why CORS?
Browser security blocks requests from `localhost:3000` to `localhost:8000` (different ports = different origins). CORS headers tell the browser it's allowed.

---

## Frontend

### `vite.config.js`
- **Proxy**: Routes `/api/*` and `/data/*` requests from the frontend dev server to the backend at `:8000`. This avoids CORS issues during development.

### `src/services/api.js`
- Axios client with 120-second timeout (VLM inference can be slow)
- `uploadVideo()`: Multipart form upload with progress callback
- `searchVideo()`: POST to `/api/search`
- `chatWithVideo()`: POST to `/api/chat`

### `src/components/UploadZone.jsx`
- **Drag-and-drop**: Uses `onDrop`, `onDragOver`, `onDragLeave` HTML5 events
- **File validation**: Only accepts `.mp4`, `.mkv`, `.avi`, `.webm`, `.mov`
- **Progress bar**: Axios `onUploadProgress` callback updates progress %

### `src/components/VideoPlayer.jsx`
- Uses **`forwardRef`** + **`useImperativeHandle`** to expose the `jumpTo(seconds)` method
- **Why forwardRef?** Parent components (VideoPage) need to call `playerRef.current.jumpTo(5.0)` when chat timestamps or search results are clicked. This React pattern lets child components expose methods to parents.
- Native HTML5 `<video>` element with controls

### `src/components/ChatWindow.jsx`
- Maintains conversation history in `messages` state
- Sends chat history to backend for context-aware follow-up questions
- **Clickable timestamp badges**: Parses `referenced_timestamps` from API response and renders clickable `<span>` elements that call `onTimestampClick(ts)` → which calls `VideoPlayer.jumpTo(ts)`
- **Typing indicator**: Three animated dots while waiting for VLM response

### `src/components/SearchBar.jsx`
- **Three search modes**: Hybrid (default), Visual-only, Text-only — implemented as tab buttons
- **Result cards**: Show thumbnail (from frame_url), timestamp badge, transcript text, source type badge, and relevance score
- Clicking a result calls `onResultClick(timestamp)` → `VideoPlayer.jumpTo()`

### `src/pages/HomePage.jsx`
- **Status polling**: For processing videos, polls `getVideoStatus()` every 2 seconds
- **Video library grid**: Cards with status badges (Ready ✅ / Processing ⏳ / Error ❌)
- Click on a "Ready" video → navigates to `/video/{video_id}`

### `src/pages/VideoPage.jsx`
- **Split layout**: CSS Grid — left panel (70%) for video + search + transcript, right panel (420px) for chat
- **All timestamp clicks from 3 sources converge**:
  - Search result click → `jumpToTimestamp()`
  - Chat timestamp badge click → `jumpToTimestamp()`
  - Transcript line click → `jumpToTimestamp()`
  - All call → `playerRef.current.jumpTo(seconds)`

### `src/index.css`
- **Design system**: CSS custom properties (variables) for colors, spacing, shadows
- **Dark glassmorphism theme**: `backdrop-filter: blur(20px)` + semi-transparent backgrounds
- **Animations**: `fadeIn` for chat messages, `typing` for dots, `spin` for loading spinners
- **Responsive**: Media query at 900px switches to single-column layout

---

# 6. Data Flow — End to End

### Upload Flow
```
User drops video.mp4
  → UploadZone.jsx → POST /api/videos/upload (multipart)
  → routes/ingest.py saves file to data/uploads/{video_id}/
  → BackgroundTask starts run_full_pipeline()
  → Frontend polls GET /api/videos/{id}/status every 2 seconds
  → When status = "ready", video appears in library
```

### Search Flow
```
User types "person writing" and clicks Search
  → SearchBar.jsx → POST /api/search {query, video_id, search_type: "hybrid"}
  → routes/search.py → ai_engine/search.py hybrid_search()
       → visual_search(): query → CLIP text vector → Qdrant cosine search → ranked frames
       → text_search(): query → Qdrant FastEmbed search → ranked transcript segments
       → RRF fusion → combined ranked results
  → Response: [{timestamp: 5.2, score: 0.032, frame_url: "...", transcript_text: "..."}]
  → SearchBar renders result cards with thumbnails
  → User clicks a result → VideoPlayer.jumpTo(5.2)
```

### Chat Flow
```
User types "What is the professor explaining?"
  → ChatWindow.jsx → POST /api/chat {question, video_id}
  → routes/chat.py:
       1. hybrid_search(question) → top 5 results
       2. Collect frame_paths + transcript_context
       3. reason_about_video(question, frames, timestamps, transcript)
            → Load 5 images
            → Build prompt: [IMG1][IMG2]...[IMG5] + transcript + question
            → SmolVLM generates answer with timestamps
       4. _extract_timestamps(answer) → parse timestamps from answer text
  → Response: {answer: "The professor is explaining...", referenced_timestamps: [5.2, 8.1]}
  → ChatWindow renders answer + clickable [05:02] [08:06] badges
  → User clicks badge → VideoPlayer.jumpTo(5.2)
```

---

# 7. How to Run

### Prerequisites
```
- Python 3.10+
- Node.js 18+
- Docker Desktop
- FFmpeg (choco install ffmpeg)
- NVIDIA GPU + CUDA drivers (recommended)
```

### Step 1: Start Qdrant
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Step 2: Install Python dependencies
```bash
# From project root (Video_based_rag/)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Start FastAPI server
```bash
# From project root
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```
The API docs will be available at `http://localhost:8000/docs`

### Step 4: Start React frontend
```bash
cd frontend
npm install
npm run dev
```

### Step 5: Open browser
```
http://localhost:3000
```

Upload a video, wait for processing, then search and chat! 🧠
