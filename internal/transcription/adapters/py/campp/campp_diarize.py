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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

# SpeechBrain expects torchaudio backend APIs that were removed in newer torchaudio.
# DiariZen pulls SpeechBrain transitively via pyannote-audio.
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["ffmpeg"]

    torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]

if not hasattr(torchaudio, "AudioMetaData"):
    class _AudioMetaData:
        def __init__(
            self,
            sample_rate: int = 0,
            num_frames: int = 0,
            num_channels: int = 0,
            bits_per_sample: int = 0,
            encoding: str = "",
        ):
            self.sample_rate = sample_rate
            self.num_frames = num_frames
            self.num_channels = num_channels
            self.bits_per_sample = bits_per_sample
            self.encoding = encoding

    torchaudio.AudioMetaData = _AudioMetaData  # type: ignore[attr-defined]

if not hasattr(torchaudio, "set_audio_backend"):
    def _set_audio_backend(_backend):
        return None

    torchaudio.set_audio_backend = _set_audio_backend  # type: ignore[attr-defined]

DEFAULT_VAD_MODEL = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
DEFAULT_SPEAKER_MODEL = "iic/speech_campplus_sv_zh-cn_16k-common"
DEFAULT_ERES2_MODEL = os.environ.get(
    "FUNASR_ERES2_MODEL_ID",
    "iic/speech_eres2netv2_sv_zh-cn_16k-common",
)
DIARIZE_MODEL_CAMPP = "funasr_campp"
DIARIZE_MODEL_DIARIZEN_LARGE = "funasr_diarizen_large"
DEFAULT_DIARIZEN_REPO_ID = os.environ.get(
    "DIARIZEN_REPO_ID",
    "BUT-FIT/diarizen-wavlm-large-s80-md",
)


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


def normalize_diarize_model(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in {"", DIARIZE_MODEL_CAMPP, "campp", "funasr-campp"}:
        return DIARIZE_MODEL_CAMPP
    if value in {
        DIARIZE_MODEL_DIARIZEN_LARGE,
        "diarizen_large",
        "diarizen-large",
        "diarizen",
    }:
        return DIARIZE_MODEL_DIARIZEN_LARGE
    return DIARIZE_MODEL_CAMPP


def model_id_candidates(model_id: str) -> list[str]:
    base = (model_id or "").strip()
    if not base:
        return []

    candidates = [base]
    if base.startswith("iic/"):
        candidates.append("damo/" + base.split("/", 1)[1])
    elif base.startswith("damo/"):
        candidates.append("iic/" + base.split("/", 1)[1])
    return candidates


def load_automodel_best_effort(model_id: str, purpose: str):
    from funasr import AutoModel

    last_error = None
    for candidate in model_id_candidates(model_id):
        try:
            print(f"Loading {purpose} model: {candidate}")
            return AutoModel(model=candidate), candidate
        except Exception as exc:  # pragma: no cover - env/model availability varies
            last_error = exc
            print(f"Failed to load {purpose} model {candidate}: {exc}", file=sys.stderr)
    return None, (str(last_error) if last_error else "unknown error")


def normalize_speaker_model(raw: str | None) -> str:
    speaker_model = (raw or "").strip()
    if not speaker_model:
        return DEFAULT_SPEAKER_MODEL
    if speaker_model == "auto_eres2netv2":
        return DEFAULT_ERES2_MODEL
    return speaker_model


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


def build_segmented_result(audio_path: str, resolved_model: str, resolved_speaker_model: str, segments: list[dict]):
    speakers = sorted({seg["speaker"] for seg in segments})
    total_duration = max((seg["end"] for seg in segments), default=0.0)
    total_speech_time = sum(seg["duration"] for seg in segments)
    return {
        "audio_file": audio_path,
        "model": resolved_model,
        "speaker_model": resolved_speaker_model,
        "segments": segments,
        "speakers": speakers,
        "speaker_count": len(speakers),
        "total_duration": round(total_duration, 3),
        "processing_info": {
            "total_segments": len(segments),
            "total_speech_time": round(total_speech_time, 3),
        },
    }


def write_result(audio_path: str, output_file: str, result: dict, output_format: str):
    if output_format == "rttm":
        audio_name = Path(audio_path).stem
        with open(output_file, "w", encoding="utf-8") as f:
            for seg in result.get("segments", []):
                f.write(
                    f"SPEAKER {audio_name} 1 {seg['start']:.3f} {seg['duration']:.3f} "
                    f"<NA> <NA> {seg['speaker']} <NA> <NA>\n"
                )
        return

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def write_single_speaker_fallback(
    audio_path: str,
    output_file: str,
    output_format: str,
    reason: str,
):
    """Write a robust single-speaker fallback diarization result."""
    duration = 0.0
    try:
        audio, sr = sf.read(audio_path)
        if sr and sr > 0:
            duration = float(len(audio)) / float(sr)
    except Exception as exc:  # pragma: no cover - best effort fallback
        print(f"Failed to read audio duration for fallback: {exc}", file=sys.stderr)

    segments = []
    if duration > 0:
        segments.append(
            {
                "start": 0.0,
                "end": round(duration, 3),
                "speaker": "SPEAKER_00",
                "confidence": 0.0,
                "duration": round(duration, 3),
            }
        )

    result = build_segmented_result(
        audio_path=audio_path,
        resolved_model="fallback-single-speaker",
        resolved_speaker_model="none",
        segments=segments,
    )
    result["processing_info"]["fallback_reason"] = reason
    write_result(audio_path, output_file, result, output_format)


def try_diarizen_large(
    audio_path: str,
    output_file: str,
    output_format: str,
    diarize_model: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
):
    if diarize_model != DIARIZE_MODEL_DIARIZEN_LARGE:
        return False

    try:
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        from diarizen.pipelines.inference import DiariZenPipeline

        # PyTorch 2.6+ defaults torch.load(weights_only=True), while DiariZen
        # and its pyannote dependencies still require loading full objects.
        torch_load = torch.load
        serialization_load = getattr(torch.serialization, "load", None)
        if hasattr(torch.serialization, "add_safe_globals"):
            try:
                from torch.torch_version import TorchVersion

                torch.serialization.add_safe_globals([TorchVersion])
            except Exception:
                pass

        def _compat_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return torch_load(*args, **kwargs)

        torch.load = _compat_torch_load
        if serialization_load is not None:
            torch.serialization.load = _compat_torch_load

        print(f"Loading DiariZen pipeline: {DEFAULT_DIARIZEN_REPO_ID}")
        diar_pipeline = DiariZenPipeline.from_pretrained(DEFAULT_DIARIZEN_REPO_ID)
        torch.load = torch_load
        if serialization_load is not None:
            torch.serialization.load = serialization_load
        if min_speakers is not None:
            diar_pipeline.min_speakers = max(1, min_speakers)
        if max_speakers is not None:
            diar_pipeline.max_speakers = max(1, max_speakers)

        print("Running DiariZen-large diarization...")
        annotation = diar_pipeline(audio_path, sess_name=Path(audio_path).stem)
        segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            spk = str(speaker).strip()
            if not spk:
                spk = "SPEAKER_00"
            if not spk.startswith("SPEAKER_"):
                spk = f"SPEAKER_{spk}"
            segments.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "speaker": spk,
                    "confidence": 1.0,
                    "duration": round(max(0.0, end - start), 3),
                }
            )

        result = build_segmented_result(
            audio_path=audio_path,
            resolved_model=f"DiariZenPipeline({DEFAULT_DIARIZEN_REPO_ID})",
            resolved_speaker_model="pyannote/wespeaker-voxceleb-resnet34-LM",
            segments=segments,
        )
        write_result(audio_path, output_file, result, output_format)
        print("DiariZen-large diarization completed.")
        return True
    except Exception as exc:  # pragma: no cover - depends on runtime model availability
        try:
            torch.load = torch_load
        except Exception:
            pass
        try:
            if serialization_load is not None:
                torch.serialization.load = serialization_load
        except Exception:
            pass
        print(f"DiariZen-large execution failed: {exc}", file=sys.stderr)
        return False


def diarize_audio(
    audio_path: str,
    output_file: str,
    min_speakers: int = None,
    max_speakers: int = None,
    speaker_model: str = DEFAULT_SPEAKER_MODEL,
    diarize_model: str = DIARIZE_MODEL_CAMPP,
    output_format: str = "json",
):
    """Perform speaker diarization with DiariZen preference and CAM++ fallback."""
    normalized_diarize_model = normalize_diarize_model(diarize_model)

    if try_diarizen_large(
        audio_path,
        output_file,
        output_format,
        normalized_diarize_model,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    ):
        return
    if normalized_diarize_model == DIARIZE_MODEL_DIARIZEN_LARGE:
        print(
            "DiariZen-large diarization failed; falling back to CAM++ path for stability.",
            file=sys.stderr,
        )
        try:
            import funasr  # noqa: F401
        except Exception:
            print(
                "FunASR runtime is unavailable in current environment; "
                "using single-speaker fallback diarization output.",
                file=sys.stderr,
            )
            write_single_speaker_fallback(
                audio_path=audio_path,
                output_file=output_file,
                output_format=output_format,
                reason="diarizen_failed_and_funasr_unavailable",
            )
            return

    print(f"Loading FunASR VAD model...")
    vad_model, detail = load_automodel_best_effort(DEFAULT_VAD_MODEL, "VAD")
    if vad_model is None:
        raise RuntimeError(f"Failed to load VAD model: {detail}")

    resolved_speaker_model = normalize_speaker_model(speaker_model)
    if (
        normalized_diarize_model == DIARIZE_MODEL_DIARIZEN_LARGE
        and resolved_speaker_model == DEFAULT_SPEAKER_MODEL
    ):
        # For the DiariZen meeting profile, prefer ERes2NetV2 embeddings on the fallback path.
        resolved_speaker_model = DEFAULT_ERES2_MODEL

    sv_model, detail = load_automodel_best_effort(resolved_speaker_model, "speaker embedding")
    if sv_model is None:
        if resolved_speaker_model != DEFAULT_SPEAKER_MODEL:
            print(
                f"Speaker model {resolved_speaker_model} unavailable ({detail}); "
                f"falling back to {DEFAULT_SPEAKER_MODEL}.",
                file=sys.stderr,
            )
            resolved_speaker_model = DEFAULT_SPEAKER_MODEL
            sv_model, detail = load_automodel_best_effort(resolved_speaker_model, "speaker embedding")
    if sv_model is None:
        raise RuntimeError(f"Failed to load speaker embedding model: {detail}")

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
        result = build_segmented_result(
            audio_path=audio_path,
            resolved_model="FunASR-CAM++",
            resolved_speaker_model=resolved_speaker_model,
            segments=[],
        )
        result["total_duration"] = round(len(audio) / 16000, 3)
        write_result(audio_path, output_file, result, output_format)
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

    for i, (start, end) in enumerate(vad_segments):
        speaker_id = f"SPEAKER_{speaker_labels[i]:02d}"
        duration = end - start
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker_id,
                "confidence": 1.0,
                "duration": round(duration, 3),
            }
        )

    result = build_segmented_result(
        audio_path=audio_path,
        resolved_model="FunASR-CAM++",
        resolved_speaker_model=resolved_speaker_model,
        segments=segments,
    )
    print(f"Diarization completed: {result['speaker_count']} speakers, {len(segments)} segments")
    write_result(audio_path, output_file, result, output_format)

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
    parser.add_argument(
        "--speaker-model",
        default=DEFAULT_SPEAKER_MODEL,
        help="Speaker embedding model ID",
    )
    parser.add_argument(
        "--diarize-model",
        default=DIARIZE_MODEL_CAMPP,
        help="Diarization mode: funasr_campp or funasr_diarizen_large",
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
            speaker_model=args.speaker_model,
            diarize_model=args.diarize_model,
            output_format=args.output_format,
        )
    except Exception as e:
        print(f"Error during diarization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
