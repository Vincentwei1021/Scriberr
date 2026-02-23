#!/usr/bin/env python3
"""
FunASR CAM++ speaker diarization script.
Uses VAD + speaker embeddings + clustering for speaker diarization.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file and convert to 16kHz mono."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def extract_vad_segments(vad_model, audio_path: str):
    """Run VAD to detect speech segments."""
    res = vad_model.generate(input=audio_path)

    segments = []
    if res and len(res) > 0:
        # FunASR VAD returns list of [start_ms, end_ms] pairs
        vad_segments = res[0].get("value", [])
        if not vad_segments and isinstance(res[0], list):
            vad_segments = res[0]
        for seg in vad_segments:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                start_ms, end_ms = seg
                segments.append((start_ms / 1000.0, end_ms / 1000.0))

    return segments


def extract_embeddings(sv_model, audio: np.ndarray, segments: list, sr: int = 16000):
    """Extract speaker embeddings for each VAD segment."""
    embeddings = []

    for start, end in segments:
        start_sample = int(start * sr)
        end_sample = int(end * sr)

        # Ensure bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        if end_sample <= start_sample:
            continue

        segment_audio = audio[start_sample:end_sample]

        # Skip very short segments (< 200ms)
        if len(segment_audio) < sr * 0.2:
            embeddings.append(None)
            continue

        # Extract embedding
        res = sv_model.generate(input=segment_audio)
        if res and len(res) > 0:
            emb = res[0].get("spk_embedding", None)
            if emb is None:
                # Try alternative key names
                for key in ["embedding", "emb", "vector"]:
                    emb = res[0].get(key, None)
                    if emb is not None:
                        break
            if emb is not None:
                if isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy()
                elif isinstance(emb, list):
                    emb = np.array(emb)
                embeddings.append(emb.flatten())
            else:
                embeddings.append(None)
        else:
            embeddings.append(None)

    return embeddings


def cluster_speakers(
    embeddings: list,
    min_speakers: int = None,
    max_speakers: int = None,
):
    """Cluster speaker embeddings to assign speaker labels."""
    # Filter out None embeddings and track valid indices
    valid_indices = []
    valid_embeddings = []
    for i, emb in enumerate(embeddings):
        if emb is not None:
            valid_indices.append(i)
            valid_embeddings.append(emb)

    if len(valid_embeddings) == 0:
        return [0] * len(embeddings)

    if len(valid_embeddings) == 1:
        labels = [0] * len(embeddings)
        labels[valid_indices[0]] = 0
        return labels

    emb_matrix = np.stack(valid_embeddings)

    # Normalize embeddings
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_matrix = emb_matrix / norms

    # Compute pairwise cosine distances
    distances = pdist(emb_matrix, metric="cosine")

    # Agglomerative clustering
    Z = linkage(distances, method="ward")

    # Determine number of clusters
    if min_speakers is not None and min_speakers == max_speakers:
        n_clusters = min_speakers
    elif max_speakers is not None:
        # Use distance threshold to find optimal clusters, bounded by constraints
        # Try different thresholds to find one within the speaker range
        best_n = None
        for threshold in np.arange(0.3, 3.0, 0.1):
            cluster_labels = fcluster(Z, t=threshold, criterion="distance")
            n = len(set(cluster_labels))
            if min_speakers is not None and n < min_speakers:
                continue
            if n <= max_speakers:
                best_n = n
                break
        if best_n is None:
            best_n = max_speakers if max_speakers else 2
        n_clusters = best_n
    elif min_speakers is not None:
        # At least min_speakers
        best_n = None
        for threshold in np.arange(0.3, 3.0, 0.1):
            cluster_labels = fcluster(Z, t=threshold, criterion="distance")
            n = len(set(cluster_labels))
            if n >= min_speakers:
                best_n = n
                break
        if best_n is None:
            best_n = min_speakers
        n_clusters = best_n
    else:
        # Auto-detect: use a reasonable default threshold
        cluster_labels = fcluster(Z, t=1.0, criterion="distance")
        n_clusters = len(set(cluster_labels))

    # Clamp n_clusters to valid range
    n_clusters = max(1, min(n_clusters, len(valid_embeddings)))

    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Map back to all segments (including those with None embeddings)
    labels = [0] * len(embeddings)
    for i, valid_idx in enumerate(valid_indices):
        labels[valid_idx] = int(cluster_labels[i]) - 1  # fcluster is 1-indexed

    return labels


def diarize_audio(
    audio_path: str,
    output_file: str,
    min_speakers: int = None,
    max_speakers: int = None,
    output_format: str = "json",
):
    """Perform speaker diarization using FunASR CAM++."""
    print(f"Loading FunASR VAD model...")
    vad_model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")

    print(f"Loading FunASR CAM++ speaker embedding model...")
    sv_model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")

    print(f"Processing audio file: {audio_path}")

    # Load and preprocess audio
    audio = load_audio(audio_path, target_sr=16000)
    print(f"Audio loaded: {len(audio) / 16000:.2f} seconds")

    # Step 1: VAD
    print("Running VAD...")
    vad_segments = extract_vad_segments(vad_model, audio_path)
    print(f"Found {len(vad_segments)} speech segments")

    if len(vad_segments) == 0:
        print("Warning: No speech segments detected")
        result = {
            "audio_file": audio_path,
            "model": "FunASR-CAM++",
            "segments": [],
            "speakers": [],
            "speaker_count": 0,
            "total_duration": len(audio) / 16000,
            "processing_info": {
                "total_segments": 0,
                "total_speech_time": 0.0,
            },
        }
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        return

    # Step 2: Extract speaker embeddings
    print("Extracting speaker embeddings...")
    embeddings = extract_embeddings(sv_model, audio, vad_segments, sr=16000)

    # Step 3: Cluster embeddings
    print("Clustering speakers...")
    speaker_labels = cluster_speakers(
        embeddings,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # Build segments with speaker labels
    segments = []
    speakers_set = set()
    total_speech_time = 0.0

    for i, (start, end) in enumerate(vad_segments):
        speaker_id = f"SPEAKER_{speaker_labels[i]:02d}"
        speakers_set.add(speaker_id)
        duration = end - start
        total_speech_time += duration
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker_id,
                "confidence": 1.0,
                "duration": round(duration, 3),
            }
        )

    speakers = sorted(speakers_set)
    total_duration = max(seg["end"] for seg in segments) if segments else 0

    print(f"Diarization completed: {len(speakers)} speakers, {len(segments)} segments")

    if output_format == "rttm":
        with open(output_file, "w") as f:
            audio_name = Path(audio_path).stem
            for seg in segments:
                f.write(
                    f"SPEAKER {audio_name} 1 {seg['start']:.3f} {seg['duration']:.3f} "
                    f"<NA> <NA> {seg['speaker']} <NA> <NA>\n"
                )
    else:
        result = {
            "audio_file": audio_path,
            "model": "FunASR-CAM++",
            "segments": segments,
            "speakers": speakers,
            "speaker_count": len(speakers),
            "total_duration": round(total_duration, 3),
            "processing_info": {
                "total_segments": len(segments),
                "total_speech_time": round(total_speech_time, 3),
            },
        }
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization using FunASR CAM++"
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--output", "-o", required=True, help="Output file path"
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None, help="Minimum number of speakers"
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None, help="Maximum number of speakers"
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "rttm"],
        default="json",
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Validate speaker constraints
    if args.min_speakers is not None and args.min_speakers < 1:
        print("Error: min_speakers must be at least 1")
        sys.exit(1)

    if args.max_speakers is not None and args.max_speakers < 1:
        print("Error: max_speakers must be at least 1")
        sys.exit(1)

    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        print("Error: min_speakers cannot be greater than max_speakers")
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        diarize_audio(
            audio_path=args.audio_file,
            output_file=args.output,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            output_format=args.output_format,
        )
    except Exception as e:
        print(f"Error during diarization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
