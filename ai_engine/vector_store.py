"""
Module B — Vector Store Service
All Qdrant database operations: collection management, indexing, deletion.
"""

from qdrant_client import QdrantClient, models
from server.config import (
    QDRANT_HOST, QDRANT_PORT,
    FRAMES_COLLECTION, TRANSCRIPT_COLLECTION,
    CLIP_VECTOR_SIZE, TEXT_VECTOR_SIZE
)


# =====================================================
# CLIENT
# =====================================================
_client = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        print(f"[VectorStore] Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    return _client


# =====================================================
# COLLECTION MANAGEMENT
# =====================================================
def ensure_collections():
    """Create collections if they don't exist."""
    client = _get_client()

    if not client.collection_exists(FRAMES_COLLECTION):
        client.create_collection(
            collection_name=FRAMES_COLLECTION,
            vectors_config=models.VectorParams(
                size=CLIP_VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
        print(f"[VectorStore] Created collection: {FRAMES_COLLECTION}")

    # Note: TRANSCRIPT_COLLECTION is purposefully NOT created manually here.
    # Qdrant's FastEmbed integration (client.add) will automatically create it 
    # with the correct named vector configuration when the first document is added.


def delete_video_data(video_id: str):
    """Delete all vectors for a specific video from both collections."""
    client = _get_client()

    video_filter = models.Filter(
        must=[models.FieldCondition(
            key="video_id",
            match=models.MatchValue(value=video_id)
        )]
    )

    for collection in [FRAMES_COLLECTION, TRANSCRIPT_COLLECTION]:
        if client.collection_exists(collection):
            client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(filter=video_filter)
            )
    print(f"[VectorStore] Deleted all data for video: {video_id}")


# =====================================================
# INDEX FRAMES
# =====================================================
def index_frames(video_id: str, frames: list[dict], embeddings: list[list[float]]):
    """
    Store frame embeddings in Qdrant.
    frames: [{"path": str, "timestamp": float}, ...]
    embeddings: list of 512-dim vectors
    """
    client = _get_client()
    ensure_collections()

    points = []
    for i, (frame, embedding) in enumerate(zip(frames, embeddings)):
        point = models.PointStruct(
            id=abs(hash(f"{video_id}_frame_{i}")) % (2**63),
            vector=embedding,
            payload={
                "video_id": video_id,
                "timestamp": frame["timestamp"],
                "file_path": frame["path"],
                "source": "video_frame",
                "frame_index": i
            }
        )
        points.append(point)

    BATCH = 100
    for i in range(0, len(points), BATCH):
        client.upsert(
            collection_name=FRAMES_COLLECTION,
            points=points[i:i + BATCH]
        )

    print(f"[VectorStore] Indexed {len(points)} frames for video {video_id}")


# =====================================================
# INDEX TRANSCRIPT
# =====================================================
def index_transcript(video_id: str, transcript: list[dict]):
    """
    Store transcript segments in Qdrant using FastEmbed (auto-embedding).
    """
    client = _get_client()
    ensure_collections()

    documents = [seg["text"] for seg in transcript]
    metadata = [
        {
            "video_id": video_id,
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "source": "transcript"
        }
        for seg in transcript
    ]
    ids = [
        abs(hash(f"{video_id}_transcript_{i}")) % (2**63)
        for i in range(len(transcript))
    ]

    client.add(
        collection_name=TRANSCRIPT_COLLECTION,
        documents=documents,
        metadata=metadata,
        ids=ids
    )

    print(f"[VectorStore] Indexed {len(documents)} transcript segments for video {video_id}")


# =====================================================
# SEARCH OPERATIONS
# =====================================================
def search_frames(query_vector: list[float], video_id: str, limit: int = 5):
    """Search visual frames by CLIP vector."""
    client = _get_client()

    results = client.query_points(
        collection_name=FRAMES_COLLECTION,
        query=query_vector,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="video_id",
                match=models.MatchValue(value=video_id)
            )]
        ),
        limit=limit
    )
    return results.points


def search_transcript(query_text: str, video_id: str, limit: int = 5):
    """Search transcript by text (auto-embedded by FastEmbed)."""
    client = _get_client()

    results = client.query(
        collection_name=TRANSCRIPT_COLLECTION,
        query_text=query_text,
        query_filter=models.Filter(
            must=[models.FieldCondition(
                key="video_id",
                match=models.MatchValue(value=video_id)
            )]
        ),
        limit=limit
    )
    return results
