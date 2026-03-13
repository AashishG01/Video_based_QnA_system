"""
Module C — Reasoning Service (Vision Language Model)
Uses moondream2 for high-quality visual question answering.
Analyzes multiple frames individually then synthesizes a combined answer.
"""

import re
import time
import threading
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from server.config import VLM_MAX_TOKENS, DEVICE, TORCH_DTYPE


# =====================================================
# LAZY MODEL LOADING
# =====================================================
_vlm_model = None
_vlm_tokenizer = None

MOONDREAM_MODEL = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2024-08-26"


def _get_vlm():
    global _vlm_model, _vlm_tokenizer
    if _vlm_model is None:
        print(f"[Reasoning] Loading moondream2 on {DEVICE}...")
        _vlm_tokenizer = AutoTokenizer.from_pretrained(
            MOONDREAM_MODEL, revision=MOONDREAM_REVISION
        )
        _vlm_model = AutoModelForCausalLM.from_pretrained(
            MOONDREAM_MODEL,
            revision=MOONDREAM_REVISION,
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
        ).to(DEVICE)
        _vlm_model.eval()
        print("[Reasoning] moondream2 loaded ✓")
    return _vlm_model, _vlm_tokenizer


# =====================================================
# MULTI-FRAME REASONING
# =====================================================
MAX_FRAMES = 5


def reason_about_video(
    question: str,
    frame_paths: list[str],
    frame_timestamps: list[float],
    transcript_context: str = "",
) -> dict:
    """
    Analyze multiple frames + transcript to answer a question.
    Strategy:
      1. Describe each frame individually (moondream2 is single-image)
      2. Ask the question on the most relevant frame
      3. Combine frame descriptions + transcript into a final answer
    """
    model, tokenizer = _get_vlm()

    use_paths = frame_paths[:MAX_FRAMES]
    use_timestamps = frame_timestamps[:MAX_FRAMES]

    images = []
    for path in use_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"[Reasoning] Warning: could not load {path}: {e}")

    if not images:
        return {
            "answer": "I couldn't load any frames to analyze. Please try again.",
            "referenced_timestamps": [],
            "referenced_frames": []
        }

    print(f"[Reasoning] Analyzing {len(images)} frame(s) with moondream2 (this may take a while on CPU)...")

    # Heartbeat timer
    stop_heartbeat = threading.Event()
    def _heartbeat():
        start = time.time()
        while not stop_heartbeat.is_set():
            stop_heartbeat.wait(10)
            if not stop_heartbeat.is_set():
                elapsed = time.time() - start
                m, s = divmod(int(elapsed), 60)
                print(f"   ... still thinking ({m:02d}:{s:02d} elapsed)", flush=True)

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    gen_start = time.time()

    # -----------------------------------------------
    # Step 1: Describe each frame individually
    # -----------------------------------------------
    frame_descriptions = []
    for i, img in enumerate(images):
        ts = use_timestamps[i] if i < len(use_timestamps) else 0
        print(f"   [Frame {i+1}/{len(images)}] Analyzing frame at {ts:.1f}s...")

        enc_image = model.encode_image(img)

        desc = model.answer_question(
            enc_image,
            "Describe everything you see in this image in detail. Include objects, food items, colors, people, actions, and any text visible.",
            tokenizer,
        )
        frame_descriptions.append({"timestamp": ts, "description": desc.strip()})
        print(f"   [Frame {i+1}] -> {desc.strip()[:100]}...")

    # -----------------------------------------------
    # Step 2: Ask the main question on the best frame
    # -----------------------------------------------
    print(f"   Asking main question on best frame...")
    best_image = images[0]
    enc_best = model.encode_image(best_image)

    direct_answer = model.answer_question(
        enc_best,
        question,
        tokenizer,
    )

    # -----------------------------------------------
    # Step 3: Build combined answer
    # -----------------------------------------------
    stop_heartbeat.set()
    heartbeat_thread.join()
    gen_elapsed = time.time() - gen_start
    m, s = divmod(int(gen_elapsed), 60)
    print(f"[Reasoning] VLM inference complete in {m:02d}:{s:02d}")

    # Build a comprehensive answer from all the data
    answer_parts = []

    # Direct answer first
    answer_parts.append(direct_answer.strip())

    # Add frame-by-frame context
    answer_parts.append("\n\nFrame-by-frame analysis:")
    for fd in frame_descriptions:
        ts = fd["timestamp"]
        mm, ss = divmod(int(ts), 60)
        answer_parts.append(f"  [{mm:02d}:{ss:02d}] {fd['description']}")

    # Add transcript context if available
    if transcript_context:
        answer_parts.append(f"\nTranscript context:\n{transcript_context}")

    full_answer = "\n".join(answer_parts)

    referenced_ts = _extract_timestamps(full_answer, use_timestamps)

    return {
        "answer": full_answer,
        "referenced_timestamps": referenced_ts,
        "referenced_frames": use_paths[:len(images)]
    }


def _extract_timestamps(text: str, available_timestamps: list[float]) -> list[float]:
    """Extract timestamp references from the AI's answer."""
    found = set()

    for match in re.finditer(r'(\d+\.?\d*)\s*s(?:econds?)?', text):
        found.add(float(match.group(1)))

    for match in re.finditer(r'(\d+):(\d{2})', text):
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        found.add(float(minutes * 60 + seconds))

    for match in re.finditer(r'[Ff]rame\s*(\d+)', text):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(available_timestamps):
            found.add(available_timestamps[idx])

    return sorted(found)
