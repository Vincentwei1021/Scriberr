package adapters

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"scriberr/internal/transcription/interfaces"
	"scriberr/pkg/logger"
)

//go:embed py/qwen3/*
var qwen3Scripts embed.FS

// Qwen3ASRAdapter implements the TranscriptionAdapter interface for Qwen3-ASR
type Qwen3ASRAdapter struct {
	*BaseAdapter
	envPath          string
	fireRedModelDir  string
	fireRedSourceDir string
}

// NewQwen3ASRAdapter creates a new Qwen3-ASR adapter
func NewQwen3ASRAdapter(envPath string) *Qwen3ASRAdapter {
	fireRedModelDir := strings.TrimSpace(os.Getenv("FIRERED_MODEL_DIR"))
	if fireRedModelDir == "" {
		fireRedModelDir = "/app/models/FireRedASR2-AED"
	}

	capabilities := interfaces.ModelCapabilities{
		ModelID:     "qwen3_asr",
		ModelFamily: "qwen",
		DisplayName: "Qwen3-ASR 1.7B",
		Description: "Qwen3-ASR for multilingual transcription (52 languages)",
		Version:     "1.0.0",
		SupportedLanguages: []string{
			"zh", "en", "ja", "ko", "fr", "de", "es", "ru", "ar", "auto",
		},
		SupportedFormats:  []string{"wav", "mp3", "flac", "m4a"},
		RequiresGPU:       true,
		MemoryRequirement: 6144, // 6GB recommended
		Features: map[string]bool{
			"timestamps":        false,
			"multilingual":      true,
			"chinese_optimized": true,
		},
		Metadata: map[string]string{
			"engine":    "qwen_asr",
			"framework": "transformers",
			"license":   "Apache-2.0",
			"model_id":  "Qwen/Qwen3-ASR-1.7B",
		},
	}

	schema := []interfaces.ParameterSchema{
		{
			Name:        "model",
			Type:        "string",
			Required:    false,
			Default:     "Qwen/Qwen3-ASR-1.7B",
			Description: "Model name or path",
			Group:       "basic",
		},
		{
			Name:        "auto_convert_audio",
			Type:        "bool",
			Required:    false,
			Default:     true,
			Description: "Automatically convert audio to 16kHz mono",
			Group:       "advanced",
		},
	}

	baseAdapter := NewBaseAdapter("qwen3_asr", envPath, capabilities, schema)

	adapter := &Qwen3ASRAdapter{
		BaseAdapter:      baseAdapter,
		envPath:          envPath,
		fireRedModelDir:  fireRedModelDir,
		fireRedSourceDir: resolveFireRedSourceDir(fireRedModelDir),
	}

	return adapter
}

// GetSupportedModels returns the available Qwen3-ASR models
func (q *Qwen3ASRAdapter) GetSupportedModels() []string {
	return []string{"qwen3-asr-1.7b"}
}

// PrepareEnvironment sets up the Qwen3-ASR environment
func (q *Qwen3ASRAdapter) PrepareEnvironment(ctx context.Context) error {
	logger.Info("Preparing Qwen3-ASR environment", "env_path", q.envPath)

	// Copy Python entrypoints used by batch and realtime transcription.
	if err := q.copyPythonScripts(); err != nil {
		return fmt.Errorf("failed to copy transcription scripts: %w", err)
	}

	// Check if environment is already ready
	if CheckEnvironmentReady(q.envPath, "from qwen_asr import Qwen3ASRModel; import kaldiio, peft, transformers; print('ok')") {
		logger.Info("Qwen3-ASR environment already ready")
		q.initialized = true
		return nil
	}

	// Setup environment
	if err := q.setupQwen3Environment(); err != nil {
		return fmt.Errorf("failed to setup Qwen3-ASR environment: %w", err)
	}

	q.initialized = true
	logger.Info("Qwen3-ASR environment prepared successfully")
	return nil
}

// setupQwen3Environment creates the Python environment for Qwen3-ASR
func (q *Qwen3ASRAdapter) setupQwen3Environment() error {
	if err := os.MkdirAll(q.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create qwen3 directory: %w", err)
	}

	// Read pyproject.toml from embedded FS
	pyprojectContent, err := qwen3Scripts.ReadFile("py/qwen3/pyproject.toml")
	if err != nil {
		return fmt.Errorf("failed to read embedded pyproject.toml: %w", err)
	}

	// Replace the hardcoded PyTorch URL with the dynamic one based on environment
	contentStr := strings.Replace(
		string(pyprojectContent),
		"https://download.pytorch.org/whl/cu126",
		GetPyTorchWheelURL(),
		1,
	)

	pyprojectPath := filepath.Join(q.envPath, "pyproject.toml")
	if err := os.WriteFile(pyprojectPath, []byte(contentStr), 0644); err != nil {
		return fmt.Errorf("failed to write pyproject.toml: %w", err)
	}

	// Run uv sync
	logger.Info("Installing Qwen3-ASR dependencies")
	cmd := exec.Command("uv", "sync", "--native-tls")
	cmd.Dir = q.envPath
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("uv sync failed: %w: %s", err, strings.TrimSpace(string(out)))
	}

	return nil
}

// copyPythonScripts creates the Python scripts for Qwen3-ASR transcription workflows.
func (q *Qwen3ASRAdapter) copyPythonScripts() error {
	if err := os.MkdirAll(q.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	for _, scriptName := range []string{"qwen3_transcribe.py", "qwen3_realtime_worker.py"} {
		scriptContent, err := qwen3Scripts.ReadFile(filepath.Join("py/qwen3", scriptName))
		if err != nil {
			return fmt.Errorf("failed to read embedded %s: %w", scriptName, err)
		}

		scriptPath := filepath.Join(q.envPath, scriptName)
		if err := os.WriteFile(scriptPath, scriptContent, 0755); err != nil {
			return fmt.Errorf("failed to write %s: %w", scriptName, err)
		}
	}

	return nil
}

// Transcribe processes audio using Qwen3-ASR
func (q *Qwen3ASRAdapter) Transcribe(ctx context.Context, input interfaces.AudioInput, params map[string]interface{}, procCtx interfaces.ProcessingContext) (*interfaces.TranscriptResult, error) {
	startTime := time.Now()
	q.LogProcessingStart(input, procCtx)
	defer func() {
		q.LogProcessingEnd(procCtx, time.Since(startTime), nil)
	}()

	// Validate input
	if err := q.ValidateAudioInput(input); err != nil {
		return nil, fmt.Errorf("invalid audio input: %w", err)
	}

	// Validate parameters
	if err := q.ValidateParameters(params); err != nil {
		return nil, fmt.Errorf("invalid parameters: %w", err)
	}

	// Create temporary directory
	tempDir, err := q.CreateTempDirectory(procCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer q.CleanupTempDirectory(tempDir)

	// Convert audio if needed
	audioInput := input
	if q.GetBoolParameter(params, "auto_convert_audio") {
		convertedInput, err := q.ConvertAudioFormat(ctx, input, "wav", 16000)
		if err != nil {
			logger.Warn("Audio conversion failed, using original", "error", err)
		} else {
			audioInput = convertedInput
		}
	}

	// Build command arguments
	args, err := q.buildQwen3Args(audioInput, params, tempDir)
	if err != nil {
		return nil, fmt.Errorf("failed to build command: %w", err)
	}

	// Execute Qwen3-ASR
	cmd := exec.CommandContext(ctx, "uv", args...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

	// Setup log file
	logFile, err := os.OpenFile(filepath.Join(procCtx.OutputDirectory, "transcription.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		logger.Warn("Failed to create log file", "error", err)
	} else {
		defer logFile.Close()
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}

	logger.Info("Executing Qwen3-ASR command", "args", strings.Join(args, " "))

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.Canceled {
			return nil, fmt.Errorf("transcription was cancelled")
		}

		// Read tail of log file for context
		logPath := filepath.Join(procCtx.OutputDirectory, "transcription.log")
		logTail, readErr := q.ReadLogTail(logPath, 2048)
		if readErr != nil {
			logger.Warn("Failed to read log tail", "error", readErr)
		}

		logger.Error("Qwen3-ASR execution failed", "error", err)
		return nil, fmt.Errorf("Qwen3-ASR execution failed: %w\nLogs:\n%s", err, logTail)
	}

	// Parse result
	result, err := q.parseResult(tempDir)
	if err != nil {
		return nil, fmt.Errorf("failed to parse result: %w", err)
	}

	result.ProcessingTime = time.Since(startTime)
	result.ModelUsed = "Qwen3-ASR-1.7B"

	logger.Info("Qwen3-ASR transcription completed",
		"segments", len(result.Segments),
		"processing_time", result.ProcessingTime)

	return result, nil
}

// buildQwen3Args builds the command arguments for Qwen3-ASR
func (q *Qwen3ASRAdapter) buildQwen3Args(input interfaces.AudioInput, params map[string]interface{}, tempDir string) ([]string, error) {
	outputFile := filepath.Join(tempDir, "result.json")

	scriptPath := filepath.Join(q.envPath, "qwen3_transcribe.py")
	args := []string{
		"run", "--native-tls", "--project", q.envPath, "python", scriptPath,
		input.FilePath,
		"--output", outputFile,
	}

	// Add model if not default
	if model := q.GetStringParameter(params, "model"); model != "" && model != "Qwen/Qwen3-ASR-1.7B" {
		args = append(args, "--model", model)
	}

	if q.fireRedSourceDir != "" {
		args = append(args, "--source-dir", q.fireRedSourceDir)
	}

	vadDir := filepath.Join(filepath.Dir(q.fireRedModelDir), "FireRedVAD", "VAD")
	if _, err := os.Stat(vadDir); err == nil {
		args = append(args, "--vad-model-dir", vadDir)
	}
	args = append(args, "--max-segment-seconds", "18")

	return args, nil
}

// parseResult parses the Qwen3-ASR output
func (q *Qwen3ASRAdapter) parseResult(tempDir string) (*interfaces.TranscriptResult, error) {
	resultFile := filepath.Join(tempDir, "result.json")

	data, err := os.ReadFile(resultFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read result file: %w", err)
	}

	var qwen3Result struct {
		Transcription     string `json:"transcription"`
		Language          string `json:"language"`
		AudioFile         string `json:"audio_file"`
		Model             string `json:"model"`
		SegmentTimestamps []struct {
			Segment string  `json:"segment"`
			Start   float64 `json:"start"`
			End     float64 `json:"end"`
		} `json:"segment_timestamps"`
		WordTimestamps []struct {
			Word  string  `json:"word"`
			Start float64 `json:"start"`
			End   float64 `json:"end"`
		} `json:"word_timestamps"`
	}

	if err := json.Unmarshal(data, &qwen3Result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON result: %w", err)
	}

	// Convert to standard format
	result := &interfaces.TranscriptResult{
		Text:     qwen3Result.Transcription,
		Language: qwen3Result.Language,
		Segments: make([]interfaces.TranscriptSegment, len(qwen3Result.SegmentTimestamps)),
	}

	for i, seg := range qwen3Result.SegmentTimestamps {
		result.Segments[i] = interfaces.TranscriptSegment{
			Start: seg.Start,
			End:   seg.End,
			Text:  seg.Segment,
		}
	}

	// Convert word timestamps if present
	if len(qwen3Result.WordTimestamps) > 0 {
		result.WordSegments = make([]interfaces.TranscriptWord, len(qwen3Result.WordTimestamps))
		for i, word := range qwen3Result.WordTimestamps {
			result.WordSegments[i] = interfaces.TranscriptWord{
				Start: word.Start,
				End:   word.End,
				Word:  word.Word,
				Score: 1.0,
			}
		}
	}

	return result, nil
}

// GetEstimatedProcessingTime provides Qwen3-ASR-specific time estimation
func (q *Qwen3ASRAdapter) GetEstimatedProcessingTime(input interfaces.AudioInput) time.Duration {
	baseTime := q.BaseAdapter.GetEstimatedProcessingTime(input)
	// Qwen3-ASR is relatively fast on GPU
	return time.Duration(float64(baseTime) * 0.2)
}
