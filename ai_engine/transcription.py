"""
Module A — Transcription Service
GPU-accelerated Whisper ASR with word-level timestamps.
"""

import time
import json
from pathlib import Path
from faster_whisper import WhisperModel
from server.config import WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE, DATA_DIR


# =====================================================
# LAZY MODEL LOADING (loaded once, reused)
# =====================================================
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print(f"[Transcription] Loading Whisper '{WHISPER_MODEL}' on {WHISPER_DEVICE}...")
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE
        )
        print("[Transcription] Whisper loaded ✓")
    return _whisper_model


# =====================================================
# TRANSCRIPTION
# =====================================================
def transcribe(audio_path: str, video_id: str) -> list[dict]:
    """
    Transcribe audio with word-level timestamps.
    Returns: List of segments with {start_time, end_time, text, words[]}.
    Also saves transcript.json and transcript.srt.
    """
    model = _get_whisper()

    print(f"[Transcription] Processing: {audio_path}")
    start = time.time()

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
    )

    print(f"[Transcription] Language: {info.language} (prob={info.language_probability:.2f})")

    transcript = []
    for segment in segments:
        seg_data = {
            "start_time": round(segment.start, 2),
            "end_time": round(segment.end, 2),
            "text": segment.text.strip(),
            "words": []
        }

        if segment.words:
            for word in segment.words:
                seg_data["words"].append({
                    "word": word.word.strip(),
                    "start": round(word.start, 2),
                    "end": round(word.end, 2),
                    "probability": round(word.probability, 3)
                })

        transcript.append(seg_data)
        print(f"  [{seg_data['start_time']}s → {seg_data['end_time']}s] {seg_data['text']}")

    elapsed = time.time() - start
    print(f"[Transcription] Done in {elapsed:.2f}s — {len(transcript)} segments")

    # Save transcript JSON
    save_path = DATA_DIR / "audio" / video_id / "transcript.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    # Also save SRT for external players
    _save_srt(transcript, DATA_DIR / "audio" / video_id / "transcript.srt")

    return transcript


def _save_srt(transcript: list[dict], path: Path):
    """Save transcript in SRT subtitle format."""
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(transcript, 1):
            start_srt = _seconds_to_srt(seg["start_time"])
            end_srt = _seconds_to_srt(seg["end_time"])
            f.write(f"{i}\n{start_srt} --> {end_srt}\n{seg['text']}\n\n")


def _seconds_to_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
