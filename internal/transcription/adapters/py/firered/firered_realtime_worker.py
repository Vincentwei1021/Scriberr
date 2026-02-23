#!/usr/bin/env python3
"""Realtime FireRed ASR worker using FireRed Stream-VAD.

Protocol (stdin -> stdout):
- Input: length-prefixed binary chunks where payload is float32 PCM (16kHz preferred).
  - 4-byte little-endian unsigned length N
  - N bytes payload
  - N == 0 means end-of-stream
- Output: one JSON object per line
  - {"type": "ready", "message": "..."}
  - {"type": "text", "text": "..."}
  - {"type": "error", "message": "..."}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import traceback
import uuid
from collections import deque

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
MIN_VAD_SAMPLES = 1600  # 100ms @ 16kHz


def emit(event_type: str, **kwargs) -> None:
    payload = {"type": event_type, **kwargs}
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def ensure_fireredasr2s_importable(model_dir: str, source_dir: str | None = None) -> None:
    candidates: list[str] = []

    env_source = os.environ.get("FIRERED_SOURCE_DIR")
    if env_source:
        candidates.append(env_source)

    if source_dir:
        candidates.append(source_dir)

    if model_dir:
        candidates.append(os.path.dirname(os.path.dirname(model_dir)))

    normalized: list[str] = []
    seen: set[str] = set()
    for path in candidates:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        normalized.append(abs_path)

    for repo_root in normalized:
        if os.path.isdir(os.path.join(repo_root, "fireredasr2s")) and repo_root not in sys.path:
            sys.path.insert(0, repo_root)

    try:
        import fireredasr2s  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import fireredasr2s. Set --source-dir or FIRERED_SOURCE_DIR "
            f"to the FireRedASR2S repository root. Tried: {normalized}"
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


def load_models(args):
    ensure_fireredasr2s_importable(args.model_dir, args.source_dir)

    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
    from fireredasr2s.fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

    use_gpu = bool(args.use_gpu)

    vad_config = FireRedStreamVadConfig(
        use_gpu=use_gpu,
        smooth_window_size=5,
        speech_threshold=0.3,
        pad_start_frame=5,
        min_speech_frame=8,
        max_speech_frame=2000,
        min_silence_frame=20,
    )
    vad = FireRedStreamVad.from_pretrained(args.vad_model_dir, vad_config)

    asr_config = FireRedAsr2Config(
        use_gpu=use_gpu,
        use_half=False,
        beam_size=args.beam_size,
        nbest=1,
        decode_max_len=0,
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=False,
    )
    asr_model = FireRedAsr2.from_pretrained("aed", args.model_dir, asr_config)

    punc_model = None
    if args.punc_model_dir and os.path.exists(args.punc_model_dir):
        try:
            from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig

            punc_model = FireRedPunc.from_pretrained(
                args.punc_model_dir,
                FireRedPuncConfig(use_gpu=use_gpu),
            )
        except Exception as exc:  # pragma: no cover - optional model
            emit("log", message=f"Punctuation unavailable: {exc}")

    return vad, asr_model, punc_model


def transcribe_segment(audio_float32: np.ndarray, asr_model, punc_model) -> str:
    if len(audio_float32) < int(SAMPLE_RATE * 0.3):
        return ""

    if np.max(np.abs(audio_float32)) < 0.005:
        return ""

    wav_path = ""
    uttid = str(uuid.uuid4())
    text = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            sf.write(temp_wav.name, audio_float32, SAMPLE_RATE)
            wav_path = temp_wav.name

        results = asr_model.transcribe([uttid], [wav_path])
        text = results[0].get("text", "") if results else ""
        text = re.sub(r"(<blank>)|(<sil>)", "", text).strip()

        if text and punc_model is not None:
            try:
                punc_results = punc_model.process([text], [uttid])
                if punc_results:
                    text = punc_results[0].get("punc_text", text)
            except Exception:
                pass
    finally:
        if wav_path:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    return text.strip()


def run(args) -> None:
    vad, asr_model, punc_model = load_models(args)
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

            vad_chunk_i16 = float32_to_int16(vad_chunk_f32)
            vad_results = vad.detect_chunk(vad_chunk_i16)

            ended = any(r.is_speech_end for r in vad_results)
            started = any(r.is_speech_start for r in vad_results)

            if ended and in_speech:
                speech_chunks.append(vad_chunk_f32)
                segment = np.concatenate(speech_chunks)
                in_speech = False
                speech_chunks = []

                text = transcribe_segment(segment, asr_model, punc_model)
                if text:
                    emit("text", text=text)

            if started and not in_speech:
                in_speech = True
                speech_chunks = [c.copy() for c in pre_speech]
                speech_chunks.append(vad_chunk_f32)
            elif in_speech and not ended:
                speech_chunks.append(vad_chunk_f32)

            if not in_speech:
                pre_speech.append(vad_chunk_f32)

    finally:
        if in_speech and speech_chunks:
            try:
                segment = np.concatenate(speech_chunks)
                text = transcribe_segment(segment, asr_model, punc_model)
                if text:
                    emit("text", text=text)
            except Exception:
                pass
        vad.reset()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FireRed realtime worker")
    parser.add_argument("--model-dir", required=True, help="Path to FireRedASR2-AED model directory")
    parser.add_argument("--vad-model-dir", required=True, help="Path to FireRedVAD Stream-VAD model directory")
    parser.add_argument("--source-dir", help="Path to FireRedASR2S repository root")
    parser.add_argument("--punc-model-dir", help="Path to FireRed punctuation model directory")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam size for decoding")
    parser.add_argument("--use-gpu", type=int, default=1, help="Use GPU if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_dir):
        emit("error", message=f"Model dir not found: {args.model_dir}")
        return

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
