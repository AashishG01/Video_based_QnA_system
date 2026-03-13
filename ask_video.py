"""
🧠 Video Brain — CLI Script
Ask any question about a video directly from the terminal.

Usage: Just edit VIDEO_PATH and QUESTION below, then run:
    python ask_video.py
"""

import sys
import codecs
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

import os
import uuid
import time
import shutil
from pathlib import Path

# =====================================================
# ✏️ EDIT THESE TWO VALUES
# =====================================================
VIDEO_PATH = r"C:\Users\Hp\Desktop\Video_based_rag\Emotions_-_30_Sec_480P.mp4"   # <-- Your video path here
QUESTION = "what is food item shown in the video"                           # <-- Your question here
# =====================================================

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def format_time(seconds: float) -> str:
    """Format seconds into MM:SS."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def main():
    video_path = VIDEO_PATH
    question = QUESTION
    video_id = str(uuid.uuid4())[:8]

    print("=" * 60)
    print("🧠  VIDEO BRAIN — CLI Mode")
    print("=" * 60)

    if not os.path.exists(video_path):
        print(f"\n❌ Error: File not found: {video_path}")
        print("   Please update VIDEO_PATH at the top of this file.")
        sys.exit(1)

    print(f"\n📹 Video: {video_path}")
    print(f"❓ Question: {question}")
    total_start = time.time()

    # --------------------------------------------------
    # Step 1: Extract metadata
    # --------------------------------------------------
    from ai_engine.ingestion import get_video_metadata, extract_frames, extract_audio

    print(f"\n{'─' * 50}")
    print("📊 Step 1/6 — Extracting metadata...")
    metadata = get_video_metadata(video_path)
    print(f"   Duration: {format_time(metadata['duration'])} | "
          f"Resolution: {metadata['width']}x{metadata['height']} | "
          f"FPS: {metadata['fps']} | Codec: {metadata['codec']}")

    # --------------------------------------------------
    # Step 2: Extract frames
    # --------------------------------------------------
    print(f"\n{'─' * 50}")
    print("🎞️  Step 2/6 — Extracting frames (scene detection)...")
    t = time.time()
    frames = extract_frames(video_path, video_id)
    print(f"   ✅ {len(frames)} frames extracted in {time.time() - t:.1f}s")

    # --------------------------------------------------
    # Step 3: Extract & transcribe audio
    # --------------------------------------------------
    print(f"\n{'─' * 50}")
    print("🎤 Step 3/6 — Extracting audio & transcribing...")
    t = time.time()
    audio_path = extract_audio(video_path, video_id)

    from ai_engine.transcription import transcribe
    transcript = transcribe(audio_path, video_id)
    print(f"   ✅ {len(transcript)} segments transcribed in {time.time() - t:.1f}s")

    if transcript:
        print(f"\n   📝 Full Transcript:")
        print(f"   {'─' * 40}")
        for seg in transcript:
            print(f"   [{format_time(seg['start_time'])}] {seg['text']}")

    # --------------------------------------------------
    # Step 4: Generate embeddings
    # --------------------------------------------------
    print(f"\n{'─' * 50}")
    print("🧮 Step 4/6 — Generating CLIP embeddings...")
    t = time.time()
    from ai_engine.embeddings import embed_frames
    frame_paths = [f["path"] for f in frames]
    embeddings = embed_frames(frame_paths)
    print(f"   ✅ {len(embeddings)} frame embeddings in {time.time() - t:.1f}s")

    # --------------------------------------------------
    # Step 5: Index into Qdrant
    # --------------------------------------------------
    print(f"\n{'─' * 50}")
    print("📦 Step 5/6 — Indexing into Qdrant...")
    t = time.time()
    from ai_engine.vector_store import ensure_collections, index_frames, index_transcript, delete_video_data

    ensure_collections()
    delete_video_data(video_id)
    index_frames(video_id, frames, embeddings)
    index_transcript(video_id, transcript)
    print(f"   ✅ Indexed in {time.time() - t:.1f}s")

    # --------------------------------------------------
    # Step 6: Search & Reason
    # --------------------------------------------------
    print(f"\n{'─' * 50}")
    print("🔍 Step 6/6 — Searching & reasoning...")
    from ai_engine.search import hybrid_search

    search_results = hybrid_search(question, video_id, limit=5)

    print(f"\n   🔎 Top Search Results:")
    print(f"   {'─' * 40}")
    for i, r in enumerate(search_results, 1):
        source = r["source_type"].upper()
        text_preview = (r["transcript_text"] or "")[:80]
        print(f"   {i}. [{format_time(r['timestamp'])}] (score: {r['score']:.4f}) [{source}]")
        if text_preview:
            print(f"      \"{text_preview}\"")

    # VLM Reasoning
    from ai_engine.reasoning import reason_about_video

    frame_paths_for_vlm = [r["frame_path"] for r in search_results if r.get("frame_path")]
    frame_timestamps = [r["timestamp"] for r in search_results if r.get("frame_path")]

    transcript_context = "\n".join([
        f"[{format_time(r['timestamp'])}] {r['transcript_text']}"
        for r in search_results if r.get("transcript_text")
    ])

    result = reason_about_video(
        question=question,
        frame_paths=frame_paths_for_vlm[:5],
        frame_timestamps=frame_timestamps[:5],
        transcript_context=transcript_context,
    )

    elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"🧠 ANSWER:")
    print(f"{'=' * 60}")
    print(f"\n{result['answer']}\n")

    if result["referenced_timestamps"]:
        ts_str = ", ".join([format_time(ts) for ts in result["referenced_timestamps"]])
        print(f"⏱  Referenced timestamps: {ts_str}")

    print(f"\n{'─' * 60}")
    print(f"⏱  Total time: {elapsed:.1f}s")
    print(f"📊 Frames: {len(frames)} | Transcript segments: {len(transcript)} | Search results: {len(search_results)}")
    print(f"{'─' * 60}")

    # Cleanup
    delete_video_data(video_id)
    from server.config import DATA_DIR
    for d in [DATA_DIR / "frames" / video_id, DATA_DIR / "audio" / video_id]:
        if d.exists():
            shutil.rmtree(d)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
