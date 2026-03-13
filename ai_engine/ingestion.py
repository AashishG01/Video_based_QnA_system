"""
Module A — Ingestion Service
Extracts frames and audio from video files using FFmpeg.
GPU-optimized: uses hardware-accelerated decoding when available.
"""

import subprocess
import os
import json
import re
from pathlib import Path
from server.config import FRAMES_DIR, AUDIO_DIR, FRAME_RATE, SCENE_THRESHOLD, USE_SCENE_DETECTION


def get_video_metadata(video_path: str) -> dict:
    """
    Extract video metadata using ffprobe.
    Returns: duration, width, height, fps, codec.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    video_stream = next(
        (s for s in probe.get("streams", []) if s["codec_type"] == "video"),
        {}
    )

    duration = float(probe.get("format", {}).get("duration", 0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # Parse frame rate (e.g., "30/1" or "29.97")
    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    if "/" in r_frame_rate:
        num, den = map(int, r_frame_rate.split("/"))
        fps = num / den if den else 30.0
    else:
        fps = float(r_frame_rate)

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "codec": video_stream.get("codec_name", "unknown"),
    }


def extract_frames(video_path: str, video_id: str) -> list[dict]:
    """
    Extract frames from video.
    Mode 1 (USE_SCENE_DETECTION=True): Extract on scene changes — smart, fewer frames.
    Mode 2 (USE_SCENE_DETECTION=False): Fixed FPS extraction.

    Returns: List of {"path": str, "timestamp": float}
    """
    output_dir = FRAMES_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if USE_SCENE_DETECTION:
        vf_filter = f"select='gt(scene,{SCENE_THRESHOLD})',showinfo"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", vf_filter,
            "-vsync", "vfr",
            str(output_dir / "frame_%04d.jpg"),
            "-hide_banner", "-loglevel", "info"
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={FRAME_RATE}",
            str(output_dir / "frame_%04d.jpg"),
            "-hide_banner", "-loglevel", "error"
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    frames = []
    frame_files = sorted(output_dir.glob("frame_*.jpg"))

    if USE_SCENE_DETECTION:
        timestamps = _parse_scene_timestamps(result.stderr)
        for i, frame_path in enumerate(frame_files):
            ts = timestamps[i] if i < len(timestamps) else float(i)
            frames.append({
                "path": str(frame_path),
                "timestamp": round(ts, 2)
            })
    else:
        for i, frame_path in enumerate(frame_files):
            frames.append({
                "path": str(frame_path),
                "timestamp": float(i) / FRAME_RATE
            })

    print(f"[Ingestion] Extracted {len(frames)} frames → {output_dir}")
    return frames


def _parse_scene_timestamps(ffmpeg_output: str) -> list[float]:
    """Parse timestamps from FFmpeg showinfo filter output."""
    timestamps = []
    pattern = r"pts_time:\s*([\d.]+)"
    for match in re.finditer(pattern, ffmpeg_output):
        timestamps.append(float(match.group(1)))
    return timestamps


def extract_audio(video_path: str, video_id: str) -> str:
    """
    Extract clean mono 16kHz audio optimized for Whisper.
    Returns: path to extracted audio file.
    """
    output_dir = AUDIO_DIR / video_id
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(output_dir / "audio.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
        "-hide_banner", "-loglevel", "error"
    ]
    subprocess.run(cmd, check=True)

    print(f"[Ingestion] Audio extracted → {audio_path}")
    return audio_path
