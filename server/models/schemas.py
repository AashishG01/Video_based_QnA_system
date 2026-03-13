"""
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SearchType(str, Enum):
    VISUAL = "visual"
    TEXT = "text"
    HYBRID = "hybrid"


class ProcessingStatus(str, Enum):
    UPLOADING = "uploading"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    status: ProcessingStatus
    progress: float = Field(ge=0.0, le=1.0)
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    error_message: Optional[str] = None


class VideoListResponse(BaseModel):
    videos: List[VideoInfo]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    video_id: str
    search_type: SearchType = SearchType.HYBRID
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    timestamp: float
    end_time: Optional[float] = None
    score: float
    source_type: str
    frame_url: Optional[str] = None
    transcript_text: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_results: int


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    video_id: str
    chat_history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    referenced_timestamps: List[float]
    referenced_frames: List[str]


class TranscriptSegment(BaseModel):
    start_time: float
    end_time: float
    text: str


class TranscriptResponse(BaseModel):
    video_id: str
    segments: List[TranscriptSegment]
    total_segments: int
