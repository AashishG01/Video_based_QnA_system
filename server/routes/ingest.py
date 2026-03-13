"""
API Routes — Video Ingestion
Upload videos and trigger the processing pipeline.
"""

import uuid
import json
import traceback
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from server.config import UPLOAD_DIR, DATA_DIR
from server.models.schemas import VideoInfo, VideoListResponse, ProcessingStatus

router = APIRouter(prefix="/api", tags=["videos"])

# =====================================================
# IN-MEMORY VIDEO STATUS TRACKER
# =====================================================
video_status: dict[str, dict] = {}


def _update_status(video_id: str, status: str, progress: float, **kwargs):
    """Update video processing status."""
    if video_id in video_status:
        video_status[video_id]["status"] = status
        video_status[video_id]["progress"] = progress
        video_status[video_id].update(kwargs)
    print(f"[Pipeline] {video_id}: {status} ({progress*100:.0f}%)")


# =====================================================
# FULL PIPELINE (runs in background)
# =====================================================
def run_full_pipeline(video_id: str, video_path: str):
    """
    Complete ingestion pipeline:
    1. Extract frames (FFmpeg)
    2. Extract audio (FFmpeg)
    3. Transcribe (Whisper GPU)
    4. Embed frames (CLIP)
    5. Index everything (Qdrant)
    """
    try:
        # Lazy imports — AI models load only when first video is processed
        from ai_engine.ingestion import extract_frames, extract_audio, get_video_metadata
        from ai_engine.transcription import transcribe
        from ai_engine.embeddings import embed_frames
        from ai_engine.vector_store import index_frames, index_transcript, delete_video_data

        _update_status(video_id, "extracting", 0.05)
        metadata = get_video_metadata(video_path)
        _update_status(video_id, "extracting", 0.1, duration=metadata["duration"])

        delete_video_data(video_id)

        _update_status(video_id, "extracting", 0.15)
        frames = extract_frames(video_path, video_id)
        _update_status(video_id, "extracting", 0.30, frame_count=len(frames))

        audio_path = extract_audio(video_path, video_id)
        _update_status(video_id, "transcribing", 0.35)

        transcript = transcribe(audio_path, video_id)
        _update_status(video_id, "embedding", 0.55)

        frame_paths = [f["path"] for f in frames]
        embeddings = embed_frames(frame_paths)
        _update_status(video_id, "indexing", 0.75)

        index_frames(video_id, frames, embeddings)
        _update_status(video_id, "indexing", 0.85)

        index_transcript(video_id, transcript)
        _update_status(video_id, "ready", 1.0)

        print(f"[Pipeline] ✅ Video {video_id} fully processed!")

    except Exception as e:
        print(f"[Pipeline] ❌ Error processing {video_id}: {e}")
        traceback.print_exc()
        _update_status(video_id, "error", 0.0, error_message=str(e))


# =====================================================
# ENDPOINTS
# =====================================================
@router.post("/videos/upload", response_model=VideoInfo)
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload a video file and start processing pipeline."""
    allowed = {".mp4", ".mkv", ".avi", ".webm", ".mov"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported format: {suffix}. Allowed: {allowed}")

    video_id = str(uuid.uuid4())[:8]
    video_dir = UPLOAD_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / file.filename

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    video_status[video_id] = {
        "video_id": video_id,
        "filename": file.filename,
        "status": "uploading",
        "progress": 0.0,
        "duration": None,
        "frame_count": None,
        "error_message": None,
    }

    background_tasks.add_task(run_full_pipeline, video_id, str(video_path))
    return VideoInfo(**video_status[video_id])


@router.get("/videos/{video_id}/status", response_model=VideoInfo)
async def get_video_status(video_id: str):
    """Check processing status of a video."""
    if video_id not in video_status:
        raise HTTPException(404, f"Video {video_id} not found")
    return VideoInfo(**video_status[video_id])


@router.get("/videos", response_model=VideoListResponse)
async def list_videos():
    """List all uploaded/processed videos."""
    videos = [VideoInfo(**v) for v in video_status.values()]
    return VideoListResponse(videos=videos)


@router.get("/videos/{video_id}/transcript")
async def get_transcript(video_id: str):
    """Get the full transcript for a video."""
    transcript_path = DATA_DIR / "audio" / video_id / "transcript.json"
    if not transcript_path.exists():
        raise HTTPException(404, "Transcript not found. Video may still be processing.")

    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    return {
        "video_id": video_id,
        "segments": segments,
        "total_segments": len(segments)
    }
