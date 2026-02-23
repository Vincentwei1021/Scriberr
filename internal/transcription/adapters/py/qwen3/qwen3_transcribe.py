#!/usr/bin/env python3
"""Qwen3-ASR transcription script for Scriberr."""

import argparse
import json
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from qwen_asr import Qwen3ASRModel


TARGET_SAMPLE_RATE = 16000


def load_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio file and convert to 16kHz mono float32 numpy array."""
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")

    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16kHz if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        audio_tensor = torch.from_numpy(audio_data).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE
        )
        audio_tensor = resampler(audio_tensor)
        audio_data = audio_tensor.squeeze(0).numpy()
        sample_rate = TARGET_SAMPLE_RATE

    return audio_data, sample_rate


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Qwen3-ASR")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model name or path (default: Qwen/Qwen3-ASR-1.7B)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}", file=sys.stderr)
    model = Qwen3ASRModel.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )

    print(f"Loading audio: {args.audio_file}", file=sys.stderr)
    audio_data, sample_rate = load_audio(args.audio_file)
    duration = len(audio_data) / sample_rate

    print("Transcribing...", file=sys.stderr)
    results = model.transcribe(audio=(audio_data, sample_rate))

    text = results[0].text if results else ""

    # Extract model short name for output
    model_short = args.model.split("/")[-1] if "/" in args.model else args.model

    # Build segment timestamps - single segment covering full audio if no fine-grained timestamps
    segment_timestamps = [{"segment": text, "start": 0.0, "end": round(duration, 3)}]

    output = {
        "transcription": text,
        "language": "auto",
        "audio_file": args.audio_file,
        "model": model_short,
        "segment_timestamps": segment_timestamps,
        "word_timestamps": [],
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Result written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
