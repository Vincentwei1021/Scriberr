# Meeting ASR System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend Scriberr with FireRedASR2-AED, Qwen3-ASR, and FunASR CAM++ adapters for Chinese meeting transcription + diarization + LLM summarization on EC2 g6e.

**Architecture:** Scriberr (Go backend) calls Python scripts via `uv run` for ML inference. Each adapter = Go file (implements interface) + Python script (does inference) + pyproject.toml (deps). We add 3 new adapters (2 transcription, 1 diarization), register them in main.go, build a custom Docker image, and configure Bedrock Claude for summaries.

**Tech Stack:** Go 1.24, Python 3.12, FireRedASR2S, Qwen3-ASR, FunASR, PyTorch CUDA, Docker, AWS Bedrock

**Remote Server:** `ssh g6e-routine` (EC2 g6e, NVIDIA L40S 48GB, Ubuntu)

---

## Pre-requisites

Scriberr source is already cloned at `~/scriberr` on g6e. All work happens on g6e via SSH.

Key reference files:
- Adapter interface: `~/scriberr/internal/transcription/interfaces/interfaces.go`
- Base adapter: `~/scriberr/internal/transcription/adapters/base_adapter.go`
- Reference transcription adapter: `~/scriberr/internal/transcription/adapters/parakeet_adapter.go`
- Reference diarization adapter: `~/scriberr/internal/transcription/adapters/pyannote_adapter.go`
- Reference Python script: `~/scriberr/internal/transcription/adapters/py/nvidia/parakeet_transcribe.py`
- Reference Python diarize: `~/scriberr/internal/transcription/adapters/py/pyannote/pyannote_diarize.py`
- Adapter registration: `~/scriberr/cmd/server/main.go` (function `registerAdapters`)
- LLM service: `~/scriberr/internal/llm/openai.go`
- Existing FireRedASR2S code: `~/server/FireRedASR2S/` (has pretrained models)
- Existing server with working FireRedASR2 inference: `~/server/main.py`

---

### Task 1: Kill existing GPU processes and prepare workspace

**Files:**
- None (server operations only)

**Step 1: Kill existing GPU processes on g6e**

```bash
ssh g6e-routine "kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null; sleep 2; nvidia-smi"
```

Expected: All GPU processes terminated, 0 MiB GPU memory used.

**Step 2: Install Go and uv on g6e**

```bash
ssh g6e-routine "
# Install Go 1.24
curl -LO https://go.dev/dl/go1.24.4.linux-amd64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.24.4.linux-amd64.tar.gz
echo 'export PATH=\$PATH:/usr/local/go/bin' >> ~/.bashrc
export PATH=\$PATH:/usr/local/go/bin
go version

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH=\$HOME/.local/bin:\$PATH' >> ~/.bashrc
export PATH=\$HOME/.local/bin:\$PATH
uv --version
"
```

Expected: `go version go1.24.4 linux/amd64` and `uv X.Y.Z`

**Step 3: Install Node.js for frontend build**

```bash
ssh g6e-routine "
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version && npm --version
"
```

Expected: Node 20.x and npm available.

**Step 4: Symlink FireRedASR2S pretrained models into scriberr workspace**

```bash
ssh g6e-routine "
ln -sf ~/server/FireRedASR2S/pretrained_models ~/scriberr/pretrained_models_firered
ls -la ~/scriberr/pretrained_models_firered/
"
```

Expected: Symlink pointing to FireRedASR2-AED, FireRedPunc, FireRedVAD models.

**Step 5: Commit**

```bash
ssh g6e-routine "cd ~/scriberr && git checkout -b feature/chinese-asr"
```

---

### Task 2: Create FireRedASR2-AED Python transcription script

**Files:**
- Create: `internal/transcription/adapters/py/firered/firered_transcribe.py`
- Create: `internal/transcription/adapters/py/firered/pyproject.toml`

**Step 1: Create the pyproject.toml for FireRedASR2 dependencies**

Create `~/scriberr/internal/transcription/adapters/py/firered/pyproject.toml`:

```toml
[project]
name = "firered-transcription"
version = "0.1.0"
description = "Audio transcription using FireRedASR2-AED"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.5.0",
    "torchaudio>=2.5.0",
    "soundfile",
    "numpy",
    "kaldiio",
    "sentencepiece",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cpu", marker = "platform_machine != 'x86_64' and sys_platform != 'darwin'" },
    { index = "pytorch", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
]
torchaudio = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cpu", marker = "platform_machine != 'x86_64' and sys_platform != 'darwin'" },
    { index = "pytorch", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

**Step 2: Create the FireRedASR2 transcription Python script**

Create `~/scriberr/internal/transcription/adapters/py/firered/firered_transcribe.py`:

```python
#!/usr/bin/env python3
"""
FireRedASR2-AED transcription script.
Processes audio files using FireRedASR2-AED model for Chinese/English ASR.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import uuid
from pathlib import Path

import soundfile as sf
import numpy as np


def ensure_fireredasr2s_importable(model_dir: str):
    """Add FireRedASR2S source to Python path so we can import it."""
    # model_dir is like .../FireRedASR2S/pretrained_models/FireRedASR2-AED
    # We need the parent of pretrained_models (the repo root) on sys.path
    repo_root = os.path.dirname(os.path.dirname(model_dir))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def convert_to_16khz_mono(audio_path: str, tmp_dir: str) -> str:
    """Convert audio to 16kHz mono WAV if needed."""
    data, sr = sf.read(audio_path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        import torchaudio
        import torch
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
        # Clean up special tokens
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

        # Handle timestamps if available
        if timestamps and "timestamp" in results[0]:
            ts = results[0]["timestamp"]
            if "segment" in ts:
                for seg in ts["segment"]:
                    output_data["segment_timestamps"].append({
                        "segment": seg.get("text", ""),
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                    })
            if "word" in ts:
                for word in ts["word"]:
                    output_data["word_timestamps"].append({
                        "word": word.get("text", ""),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0),
                    })

        # If no segment timestamps, create a single segment for full text
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

        print(f"Transcription: {text[:200]}...")
        print(f"Results saved to: {output_file}")

    finally:
        import shutil
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
```

**Step 3: Commit**

```bash
cd ~/scriberr
git add internal/transcription/adapters/py/firered/
git commit -m "feat: add FireRedASR2-AED Python transcription script"
```

---

### Task 3: Create FireRedASR2-AED Go adapter

**Files:**
- Create: `internal/transcription/adapters/firered_adapter.go`

**Step 1: Create the Go adapter**

Create `~/scriberr/internal/transcription/adapters/firered_adapter.go` following the exact pattern of `parakeet_adapter.go`:

- Embed `py/firered/*` scripts
- Struct `FireRedAdapter` embedding `*BaseAdapter`
- `NewFireRedAdapter(envPath string, modelDir string)` constructor with capabilities: zh/en, GPU required, 4GB VRAM
- `GetSupportedModels()` returning `["firered-asr2-aed"]`
- `PrepareEnvironment()`: copy scripts, run `uv sync`, verify FireRedASR2S importable
- `Transcribe()`: validate input, create temp dir, convert audio, exec `uv run ... firered_transcribe.py`, parse JSON result
- Key differences from parakeet:
  - Pass `--model-dir` pointing to pretrained model directory
  - Pass `--punc-model-dir` for punctuation
  - Parse output format matching `firered_transcribe.py` output

Important: The `//go:embed py/firered/*` directive needs to be in this file. The FireRedASR2S source code needs to be available - symlinked into the env or referenced via `--model-dir` flag.

**Step 2: Commit**

```bash
cd ~/scriberr
git add internal/transcription/adapters/firered_adapter.go
git commit -m "feat: add FireRedASR2-AED Go adapter"
```

---

### Task 4: Create Qwen3-ASR Python transcription script

**Files:**
- Create: `internal/transcription/adapters/py/qwen3/qwen3_transcribe.py`
- Create: `internal/transcription/adapters/py/qwen3/pyproject.toml`

**Step 1: Create pyproject.toml for Qwen3-ASR**

```toml
[project]
name = "qwen3-transcription"
version = "0.1.0"
description = "Audio transcription using Qwen3-ASR"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.5.0",
    "torchaudio>=2.5.0",
    "qwen-asr",
    "transformers>=4.45.0",
    "accelerate",
    "soundfile",
    "numpy",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cpu", marker = "platform_machine != 'x86_64' and sys_platform != 'darwin'" },
    { index = "pytorch", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
]
torchaudio = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cpu", marker = "platform_machine != 'x86_64' and sys_platform != 'darwin'" },
    { index = "pytorch", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

**Step 2: Create the Qwen3-ASR Python script**

Create `qwen3_transcribe.py` that:
- Loads `Qwen/Qwen3-ASR-1.7B` via `qwen_asr.Qwen3ASRModel.from_pretrained()`
- Accepts `audio_file`, `--output`, `--model` args
- Transcribes using `model.transcribe(audio=(audio_data, sample_rate))`
- Outputs JSON with same format as firered: `{transcription, language, segment_timestamps, word_timestamps}`
- Reference: `~/server/main.py` shows working Qwen3-ASR usage

**Step 3: Commit**

```bash
cd ~/scriberr
git add internal/transcription/adapters/py/qwen3/
git commit -m "feat: add Qwen3-ASR Python transcription script"
```

---

### Task 5: Create Qwen3-ASR Go adapter

**Files:**
- Create: `internal/transcription/adapters/qwen3_adapter.go`

**Step 1: Create the Go adapter**

Same pattern as firered_adapter.go but:
- `//go:embed py/qwen3/*`
- Capabilities: 52 languages (zh, en, ja, ko, ...), GPU required, 6GB VRAM
- `PrepareEnvironment()`: uv sync with qwen-asr deps, download model from HuggingFace
- `Transcribe()`: exec `uv run ... qwen3_transcribe.py`, parse JSON

**Step 2: Commit**

```bash
cd ~/scriberr
git add internal/transcription/adapters/qwen3_adapter.go
git commit -m "feat: add Qwen3-ASR Go adapter"
```

---

### Task 6: Create FunASR CAM++ Python diarization script

**Files:**
- Create: `internal/transcription/adapters/py/campp/campp_diarize.py`
- Create: `internal/transcription/adapters/py/campp/pyproject.toml`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "campp-diarization"
version = "0.1.0"
description = "Speaker diarization using FunASR CAM++"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.5.0",
    "torchaudio>=2.5.0",
    "funasr",
    "modelscope",
    "soundfile",
    "numpy",
    "scipy",
]

[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cpu", marker = "platform_machine != 'x86_64' and sys_platform != 'darwin'" },
    { index = "pytorch", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
]
torchaudio = [
    { index = "pytorch-cpu", marker = "sys_platform == 'darwin'" },
    { index = "pytorch-cpu", marker = "platform_machine != 'x86_64' and sys_platform != 'darwin'" },
    { index = "pytorch", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

**Step 2: Create the CAM++ diarization script**

Create `campp_diarize.py` that:
- Uses FunASR's `AutoModel` to load CAM++ speaker verification model
- Uses FunASR's VAD model for speech segmentation
- Performs speaker embedding extraction + clustering (spectral or agglomerative)
- Accepts `audio_file`, `--output`, `--min-speakers`, `--max-speakers`, `--output-format` (json/rttm)
- Outputs JSON matching Scriberr's `DiarizationResult` format: `{segments: [{start, end, speaker, confidence}], speakers, speaker_count}`
- FunASR models to use:
  - VAD: `iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`
  - Speaker embedding: `iic/speech_campplus_sv_zh-cn_16k-common`

**Step 3: Commit**

```bash
cd ~/scriberr
git add internal/transcription/adapters/py/campp/
git commit -m "feat: add FunASR CAM++ diarization Python script"
```

---

### Task 7: Create CAM++ Go diarization adapter

**Files:**
- Create: `internal/transcription/adapters/campp_adapter.go`

**Step 1: Create the Go adapter**

Follow `pyannote_adapter.go` pattern:
- `//go:embed py/campp/*`
- Implements `DiarizationAdapter` interface
- Capabilities: language-agnostic, GPU optional, 2GB VRAM, optimized for Chinese
- `GetMaxSpeakers() = 20`, `GetMinSpeakers() = 1`
- `PrepareEnvironment()`: copy scripts, uv sync
- `Diarize()`: exec `uv run ... campp_diarize.py`, parse JSON result into `DiarizationResult`
- No HF token required (unlike PyAnnote)

**Step 2: Commit**

```bash
cd ~/scriberr
git add internal/transcription/adapters/campp_adapter.go
git commit -m "feat: add FunASR CAM++ Go diarization adapter"
```

---

### Task 8: Register new adapters in main.go

**Files:**
- Modify: `cmd/server/main.go` (function `registerAdapters`, around line 140)

**Step 1: Add adapter registrations**

In `registerAdapters()`, add after existing registrations:

```go
// FireRedASR2-AED environment
fireredEnvPath := filepath.Join(cfg.WhisperXEnv, "firered")
fireredModelDir := os.Getenv("FIRERED_MODEL_DIR")
if fireredModelDir == "" {
    fireredModelDir = "/app/models/FireRedASR2-AED"
}
registry.RegisterTranscriptionAdapter("firered_asr",
    adapters.NewFireRedAdapter(fireredEnvPath, fireredModelDir))

// Qwen3-ASR environment
qwen3EnvPath := filepath.Join(cfg.WhisperXEnv, "qwen3")
registry.RegisterTranscriptionAdapter("qwen3_asr",
    adapters.NewQwen3ASRAdapter(qwen3EnvPath))

// FunASR CAM++ diarization environment
camppEnvPath := filepath.Join(cfg.WhisperXEnv, "campp")
registry.RegisterDiarizationAdapter("campp",
    adapters.NewCAMPPAdapter(camppEnvPath))
```

**Step 2: Add `os` import if not present**

**Step 3: Verify build compiles**

```bash
cd ~/scriberr && go build ./cmd/server/
```

Expected: Clean build, no errors.

**Step 4: Commit**

```bash
cd ~/scriberr
git add cmd/server/main.go
git commit -m "feat: register FireRedASR2, Qwen3-ASR, and CAM++ adapters"
```

---

### Task 9: Build and test Scriberr locally on g6e

**Files:**
- None (build and test)

**Step 1: Build the frontend**

```bash
cd ~/scriberr/web/frontend && npm ci && npm run build
```

**Step 2: Copy frontend build into Go embed path**

```bash
rm -rf ~/scriberr/internal/web/dist
mkdir -p ~/scriberr/internal/web
cp -r ~/scriberr/web/frontend/dist ~/scriberr/internal/web/dist
```

**Step 3: Build the Go binary**

```bash
cd ~/scriberr && CGO_ENABLED=0 go build -o scriberr-bin cmd/server/main.go
```

**Step 4: Run Scriberr with environment variables**

```bash
export DATABASE_PATH=/tmp/scriberr-test.db
export UPLOAD_DIR=/tmp/scriberr-uploads
export WHISPERX_ENV=/tmp/scriberr-env
export FIRERED_MODEL_DIR=$HOME/server/FireRedASR2S/pretrained_models/FireRedASR2-AED
export HOST=0.0.0.0
export PORT=8080
mkdir -p $UPLOAD_DIR $WHISPERX_ENV
cd ~/scriberr && ./scriberr-bin
```

Expected: Server starts on port 8080, adapter registration logs show firered_asr, qwen3_asr, campp.

**Step 5: Test web UI access**

Open `http://<g6e-public-ip>:8080` in browser. Verify UI loads.

**Step 6: Test transcription with a Chinese audio file**

Upload a Chinese audio file via the UI. Select FireRedASR2-AED model. Verify transcription completes.

---

### Task 10: Configure LLM summarization for Bedrock Claude

**Files:**
- None (configuration via Scriberr UI/API)

**Step 1: Set up a LiteLLM proxy for Bedrock**

On g6e, install litellm and configure it as a proxy:

```bash
pip install litellm[proxy]
```

Create `~/litellm-config.yaml`:
```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-4-20250514
      aws_region_name: us-east-1
```

Start the proxy:
```bash
litellm --config ~/litellm-config.yaml --port 4000 &
```

**Step 2: Configure Scriberr LLM settings**

In Scriberr UI Settings > LLM Configuration:
- Provider: OpenAI Compatible
- API Key: (any placeholder, litellm doesn't need one for Bedrock)
- Base URL: `http://localhost:4000/v1`
- Model: `claude-sonnet`

**Step 3: Create Chinese meeting summary template**

Via Scriberr UI, create a Summary Template with the following prompt:

```
你是一个专业的会议纪要助手。请根据以下会议转录文本，生成结构化的中文会议纪要。

## 输出格式

### 会议主题
[一句话概括会议核心议题]

### 参会人员
[列出转录中识别到的说话人]

### 核心讨论要点
- [要点1]
- [要点2]
- [要点3]

### 决策事项
- [决策1]
- [决策2]

### 待办事项 (Action Items)
| 事项 | 负责人 | 截止日期 |
|------|--------|----------|
| [事项描述] | [负责人] | [日期/待定] |

### 关键时间节点
- [时间节点1]
- [时间节点2]

---

以下是会议转录内容：

{transcript}
```

**Step 4: Test end-to-end**

Upload audio -> Transcribe with FireRedASR2 -> Diarize with CAM++ -> Summarize with Bedrock Claude.

---

### Task 11: Docker deployment

**Files:**
- Create: `~/scriberr/Dockerfile.custom`
- Create: `~/scriberr/docker-compose.custom.yml`

**Step 1: Create custom Dockerfile extending the CUDA build**

Based on `Dockerfile.cuda`, add:
- Copy FireRedASR2S source code into container
- Set `FIRERED_MODEL_DIR` environment variable
- Volume mount for pretrained models

**Step 2: Create docker-compose.custom.yml**

```yaml
version: "3.9"
services:
  scriberr:
    build:
      context: .
      dockerfile: Dockerfile.custom
    ports:
      - "8080:8080"
    volumes:
      - scriberr_data:/app/data
      - env_data:/app/whisperx-env
      - ./pretrained_models_firered:/app/models
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - FIRERED_MODEL_DIR=/app/models/FireRedASR2-AED
      - APP_ENV=production

  litellm:
    image: ghcr.io/berriai/litellm:main-latest
    ports:
      - "4000:4000"
    volumes:
      - ./litellm-config.yaml:/app/config.yaml
    command: ["--config", "/app/config.yaml", "--port", "4000"]
    environment:
      - AWS_DEFAULT_REGION=us-east-1

volumes:
  scriberr_data: {}
  env_data: {}
```

**Step 3: Build and deploy**

```bash
cd ~/scriberr && docker compose -f docker-compose.custom.yml up --build -d
```

**Step 4: Verify deployment**

```bash
curl http://localhost:8080/api/v1/status
```

**Step 5: Commit**

```bash
cd ~/scriberr
git add Dockerfile.custom docker-compose.custom.yml
git commit -m "feat: add custom Docker deployment with Chinese ASR models"
```

---

## Task Dependencies

```
Task 1 (setup) -> Task 2, Task 4, Task 6 (can run in parallel)
Task 2 -> Task 3
Task 4 -> Task 5
Task 6 -> Task 7
Task 3, 5, 7 -> Task 8 (register adapters)
Task 8 -> Task 9 (build & test)
Task 9 -> Task 10 (LLM config)
Task 9 -> Task 11 (Docker deployment)
```

Tasks 2+3, 4+5, 6+7 can be developed in parallel by different agents.
