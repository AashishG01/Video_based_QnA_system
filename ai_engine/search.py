"""
Module B — Hybrid Search Service
Combines visual (CLIP) and text (FastEmbed) search using
Reciprocal Rank Fusion (RRF) for best results.
"""

from ai_engine.embeddings import embed_query_visual
from ai_engine.vector_store import search_frames, search_transcript
from server.config import MAX_SEARCH_RESULTS


def visual_search(query: str, video_id: str, limit: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """Search frames using CLIP text → image cross-modal matching."""
    query_vector = embed_query_visual(query)
    results = search_frames(query_vector, video_id, limit)

    return [
        {
            "timestamp": hit.payload["timestamp"],
            "score": round(hit.score, 4),
            "source_type": "visual",
            "frame_path": hit.payload["file_path"],
            "transcript_text": None,
        }
        for hit in results
    ]


def text_search(query: str, video_id: str, limit: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """Search transcript segments using semantic text similarity."""
    results = search_transcript(query, video_id, limit)

    return [
        {
            "timestamp": hit.metadata["start_time"],
            "end_time": hit.metadata["end_time"],
            "score": round(hit.score, 4),
            "source_type": "text",
            "frame_path": None,
            "transcript_text": hit.document,
        }
        for hit in results
    ]


def hybrid_search(query: str, video_id: str, limit: int = MAX_SEARCH_RESULTS) -> list[dict]:
    """
    Fuse visual and text search results using Reciprocal Rank Fusion (RRF).
    RRF formula: score = Σ 1/(k + rank_i) for each ranking list
    """
    fetch_limit = limit * 2

    visual_results = visual_search(query, video_id, fetch_limit)
    text_results = text_search(query, video_id, fetch_limit)

    k = 60
    fused_scores = {}

    for rank, result in enumerate(visual_results):
        ts = result["timestamp"]
        rrf_score = 1.0 / (k + rank + 1)
        if ts not in fused_scores:
            fused_scores[ts] = {
                "timestamp": ts,
                "score": 0.0,
                "source_type": "hybrid",
                "frame_path": result["frame_path"],
                "transcript_text": None,
            }
        fused_scores[ts]["score"] += rrf_score

    for rank, result in enumerate(text_results):
        ts = result["timestamp"]
        rrf_score = 1.0 / (k + rank + 1)
        if ts not in fused_scores:
            fused_scores[ts] = {
                "timestamp": ts,
                "score": 0.0,
                "source_type": "hybrid",
                "frame_path": None,
                "transcript_text": result["transcript_text"],
            }
        fused_scores[ts]["score"] += rrf_score
        if result["transcript_text"]:
            fused_scores[ts]["transcript_text"] = result["transcript_text"]

    _attach_nearest_frames(fused_scores, visual_results)

    sorted_results = sorted(
        fused_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )[:limit]

    for r in sorted_results:
        r["score"] = round(r["score"], 4)

    return sorted_results


def _attach_nearest_frames(fused: dict, visual_results: list[dict]):
    """Attach nearest visual frame to text-only results."""
    if not visual_results:
        return
    for ts, data in fused.items():
        if data["frame_path"] is None:
            nearest = min(visual_results, key=lambda v: abs(v["timestamp"] - ts))
            data["frame_path"] = nearest["frame_path"]
