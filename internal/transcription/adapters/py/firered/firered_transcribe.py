#!/usr/bin/env python3
"""FireRedASR2-AED transcription script for Scriberr.

To avoid GPU OOM on long recordings, this script uses FireRedVAD (non-stream)
to segment audio first, then runs ASR chunk by chunk and merges results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import uuid

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000
FRAME_PER_SECOND = 100
MIN_SEGMENT_SEC = 0.3
DEFAULT_MAX_SEGMENT_SEC = 18.0


def ensure_fireredasr2s_importable(model_dir: str, source_dir: str | None = None) -> None:
    """Add FireRedASR2S repository root to sys.path."""
    candidates: list[str] = []

    env_source = os.environ.get("FIRERED_SOURCE_DIR")
    if env_source:
        candidates.append(env_source)

    if source_dir:
        candidates.append(source_dir)

    if model_dir:
        # Typical layout: <repo>/pretrained_models/FireRedASR2-AED
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


def convert_to_16khz_mono(audio_path: str, tmp_dir: str) -> str:
    """Convert audio to 16kHz mono WAV."""
    data, sr = sf.read(audio_path)

    if len(data.shape) > 1:
        data = data.mean(axis=1)

    if sr != TARGET_SAMPLE_RATE:
        import torch
        import torchaudio

        waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
        data = waveform.squeeze(0).numpy()

    out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(out_path, data, TARGET_SAMPLE_RATE)
    return out_path


def resolve_vad_model_dir(model_dir: str, explicit_vad_model_dir: str | None) -> str:
    if explicit_vad_model_dir:
        return explicit_vad_model_dir
    return os.path.join(os.path.dirname(model_dir), "FireRedVAD", "VAD")


def chunk_windows(duration: float, max_segment_sec: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + max_segment_sec)
        windows.append((start, end))
        start = end
    return windows


def get_speech_segments(
    wav_path: str,
    vad_model_dir: str,
    max_segment_sec: float,
) -> list[tuple[float, float]]:
    """Run FireRedVAD and return speech segments.

    Falls back to fixed windows if VAD is unavailable or returns no speech segments.
    """
    data, sr = sf.read(wav_path)
    duration = len(data) / sr if sr > 0 else 0.0

    if duration <= 0:
        return []

    fallback = chunk_windows(duration, max_segment_sec=max_segment_sec)

    if not os.path.exists(vad_model_dir):
        print(f"VAD model dir not found, fallback to fixed windows: {vad_model_dir}", file=sys.stderr)
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

        if segments:
            return segments

        print("VAD returned no speech; fallback to fixed windows", file=sys.stderr)
        return fallback
    except Exception as exc:  # pragma: no cover - best effort fallback
        print(f"VAD failed, fallback to fixed windows: {exc}", file=sys.stderr)
        return fallback


def transcribe_segment(
    model,
    punc_model,
    audio_data: np.ndarray,
    sample_rate: int,
    tmp_dir: str,
    timestamps: bool,
    segment_start_sec: float,
) -> tuple[str, list[dict], list[dict]]:
    """Transcribe one audio segment and return text + adjusted timestamps."""
    segment_file = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(segment_file, audio_data, sample_rate)

    uttid = str(uuid.uuid4())
    results = model.transcribe([uttid], [segment_file])
    if not results:
        return "", [], []

    text = results[0].get("text", "")
    text = re.sub(r"(<blank>)|(<sil>)", "", text).strip()

    if text and punc_model is not None:
        try:
            punc_results = punc_model.process([text], [uttid])
            if punc_results:
                text = punc_results[0].get("punc_text", text)
        except Exception as exc:  # pragma: no cover - optional feature
            print(f"Punctuation failed on one segment: {exc}", file=sys.stderr)

    segment_entries: list[dict] = []
    word_entries: list[dict] = []

    if timestamps and "timestamp" in results[0]:
        ts = results[0]["timestamp"]
        for seg in ts.get("segment", []) or []:
            segment_entries.append(
                {
                    "segment": seg.get("text", ""),
                    "start": float(seg.get("start", 0.0)) + segment_start_sec,
                    "end": float(seg.get("end", 0.0)) + segment_start_sec,
                }
            )

        for word in ts.get("word", []) or []:
            word_entries.append(
                {
                    "word": word.get("text", ""),
                    "start": float(word.get("start", 0.0)) + segment_start_sec,
                    "end": float(word.get("end", 0.0)) + segment_start_sec,
                }
            )

    return text, segment_entries, word_entries


def transcribe_audio(
    audio_path: str,
    model_dir: str,
    output_file: str,
    source_dir: str | None = None,
    beam_size: int = 3,
    timestamps: bool = False,
    use_punc: bool = True,
    punc_model_dir: str | None = None,
    vad_model_dir: str | None = None,
    max_segment_sec: float = DEFAULT_MAX_SEGMENT_SEC,
) -> None:
    """Transcribe audio using FireRedASR2-AED."""
    ensure_fireredasr2s_importable(model_dir, source_dir)

    from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

    print(f"Loading FireRedASR2-AED model from: {model_dir}")
    config = FireRedAsr2Config(
        use_gpu=True,
        use_half=False,
        beam_size=beam_size,
        nbest=1,
        decode_max_len=0,
        softmax_smoothing=1.25,
        aed_length_penalty=0.6,
        eos_penalty=1.0,
        return_timestamp=timestamps,
    )
    model = FireRedAsr2.from_pretrained("aed", model_dir, config)

    punc_model = None
    if use_punc and punc_model_dir and os.path.exists(punc_model_dir):
        try:
            from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig

            punc_model = FireRedPunc.from_pretrained(punc_model_dir, FireRedPuncConfig(use_gpu=True))
        except Exception as exc:  # pragma: no cover - best effort optional feature
            print(f"Punctuation model unavailable: {exc}", file=sys.stderr)

    tmp_dir = tempfile.mkdtemp()
    try:
        wav_path = convert_to_16khz_mono(audio_path, tmp_dir)

        vad_path = resolve_vad_model_dir(model_dir, vad_model_dir)
        speech_segments = get_speech_segments(
            wav_path,
            vad_model_dir=vad_path,
            max_segment_sec=max_segment_sec,
        )

        wav_data, wav_sr = sf.read(wav_path)
        if wav_sr != TARGET_SAMPLE_RATE:
            raise RuntimeError(f"Unexpected sample rate after conversion: {wav_sr}")

        text_parts: list[str] = []
        segment_timestamps: list[dict] = []
        word_timestamps: list[dict] = []

        for idx, (seg_start, seg_end) in enumerate(speech_segments):
            start_idx = int(max(0.0, seg_start) * wav_sr)
            end_idx = int(max(seg_start, seg_end) * wav_sr)
            if end_idx <= start_idx:
                continue

            chunk = wav_data[start_idx:end_idx]
            if len(chunk) < int(MIN_SEGMENT_SEC * wav_sr):
                continue

            chunk_text, chunk_segments, chunk_words = transcribe_segment(
                model=model,
                punc_model=punc_model,
                audio_data=chunk,
                sample_rate=wav_sr,
                tmp_dir=tmp_dir,
                timestamps=timestamps,
                segment_start_sec=float(start_idx) / float(wav_sr),
            )

            if not chunk_text:
                continue

            text_parts.append(chunk_text)

            if timestamps and chunk_segments:
                segment_timestamps.extend(chunk_segments)
                word_timestamps.extend(chunk_words)
            else:
                segment_timestamps.append(
                    {
                        "segment": chunk_text,
                        "start": float(start_idx) / float(wav_sr),
                        "end": float(end_idx) / float(wav_sr),
                    }
                )

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(speech_segments)} segments", file=sys.stderr)

        text = "\n".join(part for part in text_parts if part.strip()).strip()

        output_data = {
            "transcription": text,
            "language": "zh",
            "audio_file": audio_path,
            "model": "FireRedASR2-AED",
            "segment_timestamps": segment_timestamps,
            "word_timestamps": word_timestamps,
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as file_obj:
            json.dump(output_data, file_obj, indent=2, ensure_ascii=False)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe audio using FireRedASR2-AED")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--model-dir", required=True, help="Path to FireRedASR2-AED model directory")
    parser.add_argument("--source-dir", help="Path to FireRedASR2S repository root")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam size for decoding")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps")
    parser.add_argument("--no-punc", dest="use_punc", action="store_false", help="Disable punctuation")
    parser.add_argument("--punc-model-dir", help="Path to punctuation model directory")
    parser.add_argument("--vad-model-dir", help="Path to FireRedVAD (non-stream) model directory")
    parser.add_argument("--max-segment-seconds", type=float, default=DEFAULT_MAX_SEGMENT_SEC, help="Max seconds per VAD segment")

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")

    transcribe_audio(
        audio_path=args.audio_file,
        model_dir=args.model_dir,
        output_file=args.output,
        source_dir=args.source_dir,
        beam_size=args.beam_size,
        timestamps=args.timestamps,
        use_punc=args.use_punc,
        punc_model_dir=args.punc_model_dir,
        vad_model_dir=args.vad_model_dir,
        max_segment_sec=max(2.0, args.max_segment_seconds),
    )


if __name__ == "__main__":
    main()
