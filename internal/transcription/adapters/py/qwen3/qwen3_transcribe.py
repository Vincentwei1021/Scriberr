#!/usr/bin/env python3
"""Qwen3-ASR transcription with FireRedVAD chunking for long-form audio."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid

import numpy as np
import soundfile as sf
import torch
import torchaudio
from qwen_asr import Qwen3ASRModel


TARGET_SAMPLE_RATE = 16000
FRAME_PER_SECOND = 100
MIN_SEGMENT_SEC = 0.3
DEFAULT_MAX_SEGMENT_SEC = 18.0


def ensure_fireredasr2s_importable(source_dir: str | None, vad_model_dir: str | None) -> None:
    candidates: list[str] = []
    if source_dir:
        candidates.append(source_dir)

    env_source = os.environ.get("FIRERED_SOURCE_DIR")
    if env_source:
        candidates.append(env_source)

    if vad_model_dir:
        candidates.append(os.path.dirname(os.path.dirname(os.path.dirname(vad_model_dir))))

    seen: set[str] = set()
    for path in candidates:
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if os.path.isdir(os.path.join(abs_path, "fireredasr2s")) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)

    import fireredasr2s  # noqa: F401


def load_audio(audio_path: str, tmp_dir: str) -> tuple[str, np.ndarray, int]:
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        audio_tensor = resampler(audio_tensor)
        audio_data = audio_tensor.squeeze(0).numpy()
        sample_rate = TARGET_SAMPLE_RATE

    out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(out_path, audio_data, sample_rate)
    return out_path, audio_data, sample_rate


def chunk_windows(duration: float, max_segment_sec: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + max_segment_sec)
        windows.append((start, end))
        start = end
    return windows


def get_speech_segments(wav_path: str, vad_model_dir: str | None, max_segment_sec: float) -> list[tuple[float, float]]:
    audio_data, sample_rate = sf.read(wav_path)
    duration = len(audio_data) / sample_rate if sample_rate > 0 else 0.0
    if duration <= 0:
        return []

    fallback = chunk_windows(duration, max_segment_sec=max_segment_sec)
    if not vad_model_dir or not os.path.exists(vad_model_dir):
        return fallback

    try:
        from fireredasr2s.fireredvad import non_stream_vad

        max_speech_frame = max(int(max_segment_sec * FRAME_PER_SECOND), 50)
        result = non_stream_vad(
            wav_path,
            model_dir=vad_model_dir,
            use_gpu=1,
            speech_threshold=0.4,
            min_speech_frame=20,
            max_speech_frame=max_speech_frame,
            min_silence_frame=20,
            merge_silence_frame=10,
            extend_speech_frame=5,
        )

        timestamps = result.get("timestamps", []) if isinstance(result, dict) else []
        segments: list[tuple[float, float]] = []
        for start, end in timestamps:
            start_f = float(start)
            end_f = float(end)
            if end_f - start_f >= MIN_SEGMENT_SEC:
                segments.append((max(0.0, start_f), min(duration, end_f)))
        return segments or fallback
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"FireRedVAD failed, fallback to fixed windows: {exc}", file=sys.stderr)
        return fallback


def extract_text(result) -> str:
    if not result:
        return ""
    first = result[0]
    if isinstance(first, dict):
        return str(first.get("text", "")).strip()
    return str(getattr(first, "text", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio using Qwen3-ASR with FireRedVAD chunking")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model name or path (default: Qwen/Qwen3-ASR-1.7B)",
    )
    parser.add_argument("--vad-model-dir", help="Path to FireRedVAD (non-stream) model directory")
    parser.add_argument("--source-dir", help="Path to FireRedASR2S repository root")
    parser.add_argument("--max-segment-seconds", type=float, default=DEFAULT_MAX_SEGMENT_SEC)
    args = parser.parse_args()

    tmp_dir = tempfile.mkdtemp(prefix="qwen_fireredvad_")
    try:
        ensure_fireredasr2s_importable(args.source_dir, args.vad_model_dir)

        print(f"Loading model: {args.model}", file=sys.stderr)
        model_kwargs = {"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        model = Qwen3ASRModel.from_pretrained(args.model, **model_kwargs)

        print(f"Loading audio: {args.audio_file}", file=sys.stderr)
        wav_path, audio_data, sample_rate = load_audio(args.audio_file, tmp_dir)
        speech_segments = get_speech_segments(
            wav_path,
            vad_model_dir=args.vad_model_dir,
            max_segment_sec=max(2.0, args.max_segment_seconds),
        )

        text_parts: list[str] = []
        segment_timestamps: list[dict] = []

        for idx, (seg_start, seg_end) in enumerate(speech_segments):
            start_idx = int(max(0.0, seg_start) * sample_rate)
            end_idx = int(max(seg_start, seg_end) * sample_rate)
            if end_idx <= start_idx:
                continue

            chunk = audio_data[start_idx:end_idx]
            if len(chunk) < int(MIN_SEGMENT_SEC * sample_rate):
                continue

            results = model.transcribe(audio=(chunk, sample_rate))
            text = extract_text(results)
            if not text:
                continue

            text_parts.append(text)
            segment_timestamps.append(
                {
                    "segment": text,
                    "start": round(float(start_idx) / float(sample_rate), 3),
                    "end": round(float(end_idx) / float(sample_rate), 3),
                    "chunk": idx,
                }
            )

        transcription = " ".join(part for part in text_parts if part.strip()).strip()
        model_short = args.model.split("/")[-1] if "/" in args.model else args.model
        output = {
            "transcription": transcription,
            "language": "auto",
            "audio_file": args.audio_file,
            "model": model_short,
            "segment_timestamps": segment_timestamps,
            "word_timestamps": [],
            "vad_model_dir": args.vad_model_dir or "",
            "max_segment_seconds": max(2.0, args.max_segment_seconds),
        }

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Result written to: {args.output}", file=sys.stderr)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
