#\!/usr/bin/env python3
"""
FireRedASR2-AED transcription script.
Processes audio files using FireRedASR2-AED model for Chinese/English ASR.
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf


def ensure_fireredasr2s_importable(model_dir: str):
    """Add FireRedASR2S source to Python path."""
    repo_root = os.path.dirname(os.path.dirname(model_dir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def convert_to_16khz_mono(audio_path: str, tmp_dir: str) -> str:
    """Convert audio to 16kHz mono WAV if needed."""
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr \!= 16000:
        import torch
        import torchaudio
        waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)
        data = waveform.squeeze(0).numpy()
    out_path = os.path.join(tmp_dir, f"{uuid.uuid4().hex}.wav")
    sf.write(out_path, data, 16000)
    return out_path


def transcribe_audio(
    audio_path: str,
    model_dir: str,
    output_file: str,
    beam_size: int = 3,
    timestamps: bool = False,
    use_punc: bool = True,
    punc_model_dir: str = None,
):
    """Transcribe audio using FireRedASR2-AED."""
    ensure_fireredasr2s_importable(model_dir)

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
    print("Model loaded successfully")

    # Load punctuation model if available
    punc_model = None
    if use_punc and punc_model_dir and os.path.exists(punc_model_dir):
        try:
            from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig
            punc_config = FireRedPuncConfig(use_gpu=True)
            punc_model = FireRedPunc.from_pretrained(punc_model_dir, punc_config)
            print("Punctuation model loaded")
        except Exception as e:
            print(f"Punctuation model not available: {e}")

    # Convert audio format if needed
    tmp_dir = tempfile.mkdtemp()
    try:
        wav_path = convert_to_16khz_mono(audio_path, tmp_dir)
        print(f"Transcribing: {audio_path}")

        uttid = str(uuid.uuid4())
        results = model.transcribe([uttid], [wav_path])

        if not results:
            print("Error: No transcription results")
            sys.exit(1)

        text = results[0].get("text", "")
        text = re.sub(r"(<blank>)|(<sil>)", "", text).strip()

        # Add punctuation
        if text and punc_model is not None:
            try:
                punc_results = punc_model.process([text], [uttid])
                if punc_results:
                    text = punc_results[0].get("punc_text", text)
            except Exception as e:
                print(f"Punctuation failed, using raw text: {e}")

        # Build output
        output_data = {
            "transcription": text,
            "language": "zh",
            "audio_file": audio_path,
            "model": "FireRedASR2-AED",
            "segment_timestamps": [],
            "word_timestamps": [],
        }

        # If no segment timestamps, create single segment for full text
        if not output_data["segment_timestamps"] and text:
            data, sr = sf.read(wav_path)
            duration = len(data) / sr
            output_data["segment_timestamps"] = [{
                "segment": text,
                "start": 0.0,
                "end": duration,
            }]

        # Write output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Transcription: {text[:200]}")
        print(f"Results saved to: {output_file}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using FireRedASR2-AED"
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--model-dir", required=True, help="Path to FireRedASR2-AED model directory")
    parser.add_argument("--beam-size", type=int, default=3, help="Beam size for decoding")
    parser.add_argument("--timestamps", action="store_true", help="Include timestamps")
    parser.add_argument("--no-punc", dest="use_punc", action="store_false", help="Disable punctuation")
    parser.add_argument("--punc-model-dir", help="Path to punctuation model directory")

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    transcribe_audio(
        audio_path=args.audio_file,
        model_dir=args.model_dir,
        output_file=args.output,
        beam_size=args.beam_size,
        timestamps=args.timestamps,
        use_punc=args.use_punc,
        punc_model_dir=args.punc_model_dir,
    )


if __name__ == "__main__":
    main()
