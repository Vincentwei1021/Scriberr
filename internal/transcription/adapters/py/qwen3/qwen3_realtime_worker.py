#!/usr/bin/env python3
"""Realtime Qwen3-ASR worker using FireRed Stream-VAD."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import traceback
from collections import deque

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel


SAMPLE_RATE = 16000
MIN_VAD_SAMPLES = 1600


def emit(event_type: str, **kwargs) -> None:
    print(json.dumps({"type": event_type, **kwargs}, ensure_ascii=False), flush=True)


def ensure_fireredasr2s_importable(vad_model_dir: str, source_dir: str | None = None) -> None:
    candidates: list[str] = []

    env_source = os.environ.get("FIRERED_SOURCE_DIR")
    if env_source:
        candidates.append(env_source)

    if source_dir:
        candidates.append(source_dir)

    if vad_model_dir:
        candidates.append(os.path.dirname(os.path.dirname(os.path.dirname(vad_model_dir))))

    seen: set[str] = set()
    for path in candidates:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if os.path.isdir(os.path.join(abs_path, "fireredasr2s")) and abs_path not in sys.path:
            sys.path.insert(0, abs_path)

    try:
        import fireredasr2s  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import fireredasr2s. Set --source-dir or FIRERED_SOURCE_DIR "
            f"to the FireRedASR2S repository root. Tried: {candidates}"
        ) from exc


def float32_to_int16(audio: np.ndarray) -> np.ndarray:
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)


def read_exact(stream, n: int) -> bytes | None:
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def extract_text(result) -> str:
    if not result:
        return ""
    first = result[0]
    if isinstance(first, dict):
        return str(first.get("text", "")).strip()
    return str(getattr(first, "text", "")).strip()


def load_vad(args):
    ensure_fireredasr2s_importable(args.vad_model_dir, args.source_dir)

    from fireredasr2s.fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

    vad_config = FireRedStreamVadConfig(
        use_gpu=bool(args.use_gpu),
        smooth_window_size=5,
        speech_threshold=0.3,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=20,
    )
    return FireRedStreamVad.from_pretrained(args.vad_model_dir, vad_config)


def load_model(model_name: str) -> Qwen3ASRModel:
    kwargs = {"dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return Qwen3ASRModel.from_pretrained(model_name, **kwargs)


def transcribe_segment(audio_float32: np.ndarray, model: Qwen3ASRModel) -> str:
    if len(audio_float32) < int(SAMPLE_RATE * 0.3):
        return ""

    if np.max(np.abs(audio_float32)) < 0.005:
        return ""

    result = model.transcribe(audio=(audio_float32.astype(np.float32, copy=False), SAMPLE_RATE))
    return extract_text(result)


def run(args) -> None:
    vad = load_vad(args)
    model = load_model(args.model)
    emit("ready", message="Realtime worker ready")

    vad.reset()
    speech_chunks: list[np.ndarray] = []
    in_speech = False
    pre_speech = deque(maxlen=5)
    vad_buf_f32 = np.array([], dtype=np.float32)
    stdin = sys.stdin.buffer

    try:
        while True:
            header = read_exact(stdin, 4)
            if header is None:
                break

            payload_size = struct.unpack("<I", header)[0]
            if payload_size == 0:
                break

            payload = read_exact(stdin, payload_size)
            if payload is None:
                break

            chunk_f32 = np.frombuffer(payload, dtype=np.float32)
            if chunk_f32.size == 0:
                continue

            vad_buf_f32 = np.concatenate([vad_buf_f32, chunk_f32])
            if len(vad_buf_f32) < MIN_VAD_SAMPLES:
                if in_speech:
                    speech_chunks.append(chunk_f32.copy())
                continue

            vad_chunk_f32 = vad_buf_f32
            vad_buf_f32 = np.array([], dtype=np.float32)

            vad_results = vad.detect_chunk(float32_to_int16(vad_chunk_f32))
            ended = any(result.is_speech_end for result in vad_results)
            started = any(result.is_speech_start for result in vad_results)

            if ended and in_speech:
                speech_chunks.append(vad_chunk_f32)
                segment = np.concatenate(speech_chunks)
                in_speech = False
                speech_chunks = []

                text = transcribe_segment(segment, model)
                if text:
                    emit("text", text=text)

            if started and not in_speech:
                in_speech = True
                speech_chunks = [chunk.copy() for chunk in pre_speech]
                speech_chunks.append(vad_chunk_f32)
            elif in_speech and not ended:
                speech_chunks.append(vad_chunk_f32)

            if not in_speech:
                pre_speech.append(vad_chunk_f32)
    finally:
        if in_speech and speech_chunks:
            try:
                text = transcribe_segment(np.concatenate(speech_chunks), model)
                if text:
                    emit("text", text=text)
            except Exception:
                pass
        vad.reset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Qwen3-ASR worker with FireRed Stream-VAD")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--vad-model-dir", required=True, help="Path to FireRedVAD Stream-VAD model directory")
    parser.add_argument("--source-dir", help="Path to FireRedASR2S repository root")
    parser.add_argument("--use-gpu", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.vad_model_dir):
        emit("error", message=f"VAD model dir not found: {args.vad_model_dir}")
        return

    try:
        run(args)
    except Exception as exc:
        emit("error", message=str(exc))
        print(traceback.format_exc(), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
