# Meeting Transcription System - Implementation Plan

## 🎯 目标

基于 [Scriberr](https://github.com/rishikanthc/Scriberr) 搭建一套**中文会议转写+总结系统**，满足以下需求：

1. **高精度中文 ASR**：接入 SOTA 中文模型（FireRedASR2（https://github.com/FireRedTeam/FireRedASR2S?tab=readme-ov-file） / Qwen3-ASR（https://github.com/QwenLM/Qwen3-ASR?tab=readme-ov-file）），替换默认的 WhisperX
2. **说话人分离**：支持多人会议场景，自动标注"谁在说话"，FunASR 的 CAM++ 说话人分离模块（https://github.com/FunAudioLLM/Fun-ASR?tab=readme-ov-file）
3. **会议总结**：LLM 自动生成结构化会议纪要
4. **分离部署**：GPU 推理后端部署在 Linux 服务器（现有 g6e），前端通过浏览器访问
5. **录音文件后处理**：上传音频文件 → 自动转写+分离+总结，非实时

---

## 📐 架构设计

```
┌─────────────────────────────────────────────────────┐
│              EC2 g6e (Linux GPU Server)               │
│                                                       │
│  ┌─────────────────────────────────────────────────┐ │
│  │           Scriberr (Go, port 8080)              │ │
│  │                                                   │ │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────┐ │ │
│  │  │ Web UI   │  │ REST API │  │ Job Queue     │ │ │
│  │  │(SvelteKit│  │ /api/v1  │  │ (Processing)  │ │ │
│  │  └──────────┘  └──────────┘  └───────┬───────┘ │ │
│  │                                       │         │ │
│  │  ┌────────────────────────────────────▼───────┐ │ │
│  │  │        Transcription Pipeline              │ │ │
│  │  │  ┌─────────────┐  ┌──────────────────┐    │ │ │
│  │  │  │ ASR Adapter  │  │ Diarize Adapter  │    │ │ │
│  │  │  │ (Pluggable)  │  │ (Pluggable)      │    │ │ │
│  │  │  └──────┬──────┘  └────────┬─────────┘    │ │ │
│  │  └─────────┼──────────────────┼───────────────┘ │ │
│  │            │                  │                   │ │
│  │  ┌─────────▼──────────────────▼───────────────┐ │ │
│  │  │        Python Subprocess (uv run)          │ │ │
│  │  │  ┌──────────────┐  ┌─────────────────┐    │ │ │
│  │  │  │ FireRedASR /  │  │ PyAnnote /      │    │ │ │
│  │  │  │ Qwen3-ASR    │  │ FunASR CAM++    │    │ │ │
│  │  │  │ (GPU)        │  │ (GPU)           │    │ │ │
│  │  │  └──────────────┘  └─────────────────┘    │ │ │
│  │  └────────────────────────────────────────────┘ │ │
│  │                                                   │ │
│  │  ┌────────────────────────────────────────────┐ │ │
│  │  │ LLM Summary (Bedrock Claude)      │ │ │
│  │  └────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘

┌──────────────────────┐
│  Any Device (Browser) │  ──── HTTP ────▶  g6e:8080
│  macOS / Windows / iPad│
└──────────────────────┘
```

---

## 🔧 技术选型

| 组件 | 选型 | 理由 |
|---|---|---|
| **基础框架** | Scriberr (Go + SvelteKit) | Adapter 架构、完整 API、Docker 部署 |
| **ASR 模型（首选）** | FireRedASR2-AED (1.1B) | 中文 SOTA（CER 2.89%），含 VAD+标点 |
| **ASR 模型（备选）** | Qwen3-ASR-1.7B | 多语言强（52语言），中文 CER 3.76% |
| **说话人分离** | FunASR CAM++ | 效果更好 |
| **LLM 总结** | Bedrock Claude Sonnet4.6 | 模型强
| **Python 环境** | uv（Scriberr 默认） | 快速依赖管理，已集成 |
| **部署** | Docker (CUDA) on g6e | GPU 加速推理 |

---

## 📋 实施步骤

### Step 1: 环境搭建 + Scriberr 部署

**目标**：在 g6e 上跑通 Scriberr 默认配置

- [ ] g6e 上确认 CUDA + Docker + NVIDIA Container Toolkit 就绪
- [ ] Docker 部署 Scriberr（`docker-compose.cuda.yml`）
- [ ] 验证默认 WhisperX 转写流程：上传一段中文音频 → 转写 → 总结
- [ ] 确认 Web UI 可从外部浏览器访问（配置 Security Group）

**产出**：可访问的 Scriberr 实例，默认 WhisperX 可用

---

### Step 2: 开发 FireRedASR Adapter

**目标**：实现 FireRedASR2-AED 的 Scriberr Adapter

#### 2a. Python 推理脚本

创建 `internal/transcription/adapters/py/firered/firered_transcribe.py`：


#### 2b. Go Adapter

创建 `internal/transcription/adapters/firered_adapter.go`：

- 实现 `TranscriptionAdapter` 接口
- `GetCapabilities()`: 声明支持 zh/en，需要 GPU，4GB+ 内存
- `Initialize()`: 用 `uv` 安装 Python 依赖（fireredasr + torch）
- `Transcribe()`: 调用 Python 脚本，解析 JSON 输出
- `GetParameterSchema()`: beam_size、语言等参数

参考现有的 `parakeet_adapter.go`，结构完全一致，核心改动：
1. Python 脚本路径和参数
2. 模型下载逻辑（从 HuggingFace 下载 FireRedTeam/FireRedASR2-AED）
3. 输出格式映射

#### 2c. 注册 Adapter

在 `cmd/server/main.go` 的模型注册部分添加：
```go
registry.RegisterTranscriptionAdapter("firered_asr", 
    adapters.NewFireRedASRAdapter(cfg.WhisperXEnv))
```

#### 2d. 可选：Qwen3-ASR Adapter

同样的方式，创建 `qwen3_asr_adapter.go` + `qwen3_asr_transcribe.py`。
Qwen3-ASR 的推理代码更简单（基于 vLLM 或 transformers）。
可以作为第二选择，用于多语言场景。

**产出**：FireRedASR Adapter 可用，中文转写精度大幅提升

---

### Step 3: 优化说话人分离

**目标**：确保 FunASR CAM++ 在中文场景下工作良好

考虑替换为 FunASR 的 CAM++ 说话人分离模型
  - 创建 `campp_adapter.go` + `campp_diarize.py`
  - FunASR CAM++ 对中文语音做过专门优化

**产出**：说话人分离在中文会议场景下可靠工作

---

### Step 4: LLM 会议总结优化

**目标**：生成高质量中文会议纪要

- [ ] 直接对接 Bedrock Claude Sonnet4.6（通过 OpenAI 兼容 API）
- [ ] 定制 Summary Prompt，输出格式：
  - 📋 会议主题
  - 👥 参会人员
  - 📝 核心讨论要点
  - ✅ 决策事项
  - 🎯 待办事项（Action Items）+ 负责人
  - 📅 关键时间节点
- [ ] 在 Scriberr 的 LLM 配置中设置中文 prompt 模板

**产出**：上传音频 → 自动输出结构化中文会议纪要

---

### Step 5: 部署优化 + 生产化

**目标**：稳定、安全、可日常使用

- [ ] Docker Compose 配置：
  - Scriberr 主服务
  - NVIDIA Container Runtime
- [ ] 安全加固：
  - 配置 HTTPS（Let's Encrypt 或 Nginx 反代）
  - 启用 Scriberr 内置认证（JWT）
- [ ] 存储：
  - EBS 卷挂载，持久化音频文件和数据库
- [ ] 性能优化：
  - 模型预加载（避免每次推理都加载模型）
  - 并发控制（GPU 内存管理）

**产出**：生产就绪的会议转写系统

---

## 🚀 后续扩展（可选）

1. **实时转写**：Scriberr 支持流式，可后续接入实时会议场景
2. **腾讯会议/飞书集成**：自动获取会议录音并处理
3. **Webhook 通知**：转写完成后自动推送到飞书/Slack
4. **自定义词汇表**：针对公司内部术语做 hot-word 优化

---


