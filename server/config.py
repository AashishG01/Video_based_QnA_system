import os
import torch
from pathlib import Path
from dotenv import load_dotenv

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

load_dotenv()

# =====================================================
# BASE PATHS
# =====================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent     # Video_based_rag/
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FRAMES_DIR = DATA_DIR / "frames"
AUDIO_DIR = DATA_DIR / "audio"

# Create directories on import
for d in [UPLOAD_DIR, FRAMES_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =====================================================
# DEVICE CONFIGURATION (GPU Optimized)
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"[Config] Device: {DEVICE} | Dtype: {TORCH_DTYPE}")

# =====================================================
# QDRANT VECTOR DATABASE
# =====================================================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
FRAMES_COLLECTION = "video_frames"
TRANSCRIPT_COLLECTION = "video_transcript"
CLIP_VECTOR_SIZE = 512
TEXT_VECTOR_SIZE = 384

# =====================================================
# AI MODELS
# =====================================================
# Whisper — GPU accelerated with float16
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = DEVICE
WHISPER_COMPUTE = "float16" if DEVICE == "cuda" else "int8"

# CLIP — Visual + Text embeddings
CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"

# VLM — Vision Language Model for reasoning
VLM_MODEL = os.getenv("VLM_MODEL", "HuggingFaceTB/SmolVLM-500M-Instruct")
VLM_MAX_TOKENS = 2048

# =====================================================
# PROCESSING SETTINGS
# =====================================================
FRAME_RATE = 1
SCENE_THRESHOLD = 0.3
USE_SCENE_DETECTION = True
MAX_SEARCH_RESULTS = 5
EMBEDDING_BATCH_SIZE = 32

# =====================================================
# API SETTINGS
# =====================================================
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
MAX_UPLOAD_SIZE_MB = 500
