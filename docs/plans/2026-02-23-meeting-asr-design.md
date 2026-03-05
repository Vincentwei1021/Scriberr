# Meeting Transcription System - Design Document

## Goal

Build a Chinese meeting transcription + summarization system on top of Scriberr (Go + SvelteKit), deployed on EC2 g6e (NVIDIA L40S 48GB).

## Architecture

Scriberr provides the full application framework: Web UI, REST API, job queue, auth, file management. We extend it with custom adapters for Chinese-optimized ASR and diarization.

```
Scriberr (Go backend + SvelteKit frontend)
  |
  +-- TranscriptionAdapter: firered_asr (FireRedASR2-AED 1.1B)
  |     +-- firered_transcribe.py (uv run)
  |
  +-- TranscriptionAdapter: qwen3_asr (Qwen3-ASR-1.7B)
  |     +-- qwen3_transcribe.py (uv run)
  |
  +-- DiarizationAdapter: campp (FunASR CAM++)
  |     +-- campp_diarize.py (uv run)
  |
  +-- LLM Service: OpenAI-compatible -> Bedrock Claude Sonnet
        +-- Chinese meeting summary prompt
```

## Component Details

### 1. FireRedASR2-AED Adapter

- **Go**: `internal/transcription/adapters/firered_adapter.go`
- **Python**: `internal/transcription/adapters/py/firered/firered_transcribe.py`
- **Model**: FireRedTeam/FireRedASR2-AED (1.1B params, CER 2.89% on Chinese)
- **Dependencies**: fireredasr2s, torch, soundfile
- **Capabilities**: zh/en, timestamps, GPU required, 4GB+ VRAM
- **Pattern**: Follows parakeet_adapter.go exactly

### 2. Qwen3-ASR Adapter

- **Go**: `internal/transcription/adapters/qwen3_adapter.go`
- **Python**: `internal/transcription/adapters/py/qwen3/qwen3_transcribe.py`
- **Model**: Qwen/Qwen3-ASR-1.7B
- **Dependencies**: qwen-asr, torch, transformers
- **Capabilities**: 52 languages, timestamps, GPU required, 6GB+ VRAM

### 3. CAM++ Diarization Adapter

- **Go**: `internal/transcription/adapters/campp_adapter.go`
- **Python**: `internal/transcription/adapters/py/campp/campp_diarize.py`
- **Model**: FunASR CAM++ (via funasr package)
- **Dependencies**: funasr, torch, modelscope
- **Capabilities**: Speaker diarization optimized for Chinese speech, 2-20 speakers

### 4. LLM Summary

- Scriberr already has `internal/llm/openai.go` supporting OpenAI-compatible APIs
- Configure Bedrock Claude via OpenAI-compatible endpoint or LiteLLM proxy
- Custom Chinese summary prompt template

### 5. Deployment

- Build Scriberr from source with custom adapters
- Docker image extending cuda base
- Volume mounts for pretrained models (reuse existing on g6e)
- Port 8080 for Scriberr web UI

## Key Decisions

- **FunASR over 3D-Speaker**: FunASR provides end-to-end diarization pipeline (embedding + clustering), 3D-Speaker is a training toolkit
- **Build from source**: Needed to add custom adapters to Go binary
- **uv for Python**: Scriberr's standard pattern, each adapter gets its own Python env
- **Model reuse**: Mount existing pretrained models from g6e filesystem rather than re-downloading
