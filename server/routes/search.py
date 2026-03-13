"""
API Routes — Semantic Search
Visual, text, and hybrid search endpoints.
"""

from fastapi import APIRouter, HTTPException
from server.models.schemas import SearchRequest, SearchResponse, SearchResult
from server.routes.ingest import video_status

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_video(request: SearchRequest):
    """Search a video using visual, text, or hybrid semantic search."""
    if request.video_id not in video_status:
        raise HTTPException(404, f"Video {request.video_id} not found")

    status = video_status[request.video_id]
    if status["status"] != "ready":
        raise HTTPException(400, f"Video is still processing: {status['status']}")

    from ai_engine.search import visual_search, text_search, hybrid_search

    if request.search_type == "visual":
        results = visual_search(request.query, request.video_id, request.limit)
    elif request.search_type == "text":
        results = text_search(request.query, request.video_id, request.limit)
    else:
        results = hybrid_search(request.query, request.video_id, request.limit)

    return SearchResponse(
        query=request.query,
        results=[SearchResult(**r) for r in results],
        total_results=len(results)
    )
