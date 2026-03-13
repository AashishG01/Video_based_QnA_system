"""
API Routes — Chat with Video
VLM-powered conversational reasoning.
"""

from fastapi import APIRouter, HTTPException
from server.models.schemas import ChatRequest, ChatResponse
from server.routes.ingest import video_status

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_with_video(request: ChatRequest):
    """
    Ask a question about a video. The system:
    1. Searches for relevant frames + transcript (RAG retrieval)
    2. Sends them to SmolVLM for reasoning
    3. Returns a natural language answer with timestamp references
    """
    if request.video_id not in video_status:
        raise HTTPException(404, f"Video {request.video_id} not found")

    status = video_status[request.video_id]
    if status["status"] != "ready":
        raise HTTPException(400, f"Video is still processing: {status['status']}")

    from ai_engine.search import hybrid_search
    from ai_engine.reasoning import reason_about_video

    search_results = hybrid_search(request.question, request.video_id, limit=5)

    if not search_results:
        return ChatResponse(
            answer="I couldn't find any relevant content in this video for your question.",
            referenced_timestamps=[],
            referenced_frames=[]
        )

    frame_paths = [r["frame_path"] for r in search_results if r.get("frame_path")]
    frame_timestamps = [r["timestamp"] for r in search_results if r.get("frame_path")]

    transcript_parts = [
        f"[{r['timestamp']}s] {r['transcript_text']}"
        for r in search_results
        if r.get("transcript_text")
    ]
    transcript_context = "\n".join(transcript_parts)

    result = reason_about_video(
        question=request.question,
        frame_paths=frame_paths[:5],
        frame_timestamps=frame_timestamps[:5],
        transcript_context=transcript_context,
    )

    return ChatResponse(
        answer=result["answer"],
        referenced_timestamps=result["referenced_timestamps"],
        referenced_frames=result["referenced_frames"]
    )
