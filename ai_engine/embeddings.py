"""
Module B — Embedding Service
GPU-accelerated CLIP (visual) + FastEmbed (text) embeddings.
Batch processing for memory efficiency.
"""

from fastembed import ImageEmbedding, TextEmbedding
from server.config import CLIP_VISION_MODEL, CLIP_TEXT_MODEL, EMBEDDING_BATCH_SIZE


# =====================================================
# LAZY MODEL LOADING
# =====================================================
_image_model = None
_text_model = None


def _get_image_model():
    global _image_model
    if _image_model is None:
        print(f"[Embeddings] Loading CLIP vision model: {CLIP_VISION_MODEL}")
        _image_model = ImageEmbedding(model_name=CLIP_VISION_MODEL)
        print("[Embeddings] CLIP vision loaded ✓")
    return _image_model


def _get_text_model():
    global _text_model
    if _text_model is None:
        print(f"[Embeddings] Loading CLIP text model: {CLIP_TEXT_MODEL}")
        _text_model = TextEmbedding(model_name=CLIP_TEXT_MODEL)
        print("[Embeddings] CLIP text loaded ✓")
    return _text_model


# =====================================================
# FRAME EMBEDDING (CLIP Vision Encoder)
# =====================================================
def embed_frames(frame_paths: list[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """
    Convert frame images to 512-dim CLIP vectors.
    Processes in batches for memory efficiency.
    """
    model = _get_image_model()
    all_embeddings = []

    total = len(frame_paths)
    for i in range(0, total, batch_size):
        batch = frame_paths[i:i + batch_size]
        batch_embeddings = list(model.embed(batch))
        all_embeddings.extend([emb.tolist() for emb in batch_embeddings])
        print(f"[Embeddings] Frames: {min(i + batch_size, total)}/{total}")

    return all_embeddings


# =====================================================
# TEXT EMBEDDING (for search queries → CLIP space)
# =====================================================
def embed_query_visual(query: str) -> list[float]:
    """
    Embed a text query into CLIP's 512-dim visual space.
    Used for cross-modal text → image search.
    """
    model = _get_text_model()
    embedding = list(model.embed([query]))[0]
    return embedding.tolist()
