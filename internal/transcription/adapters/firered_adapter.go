package adapters

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"scriberr/internal/transcription/interfaces"
	"scriberr/pkg/logger"
)

//go:embed py/firered/*
var fireredScripts embed.FS

// FireRedAdapter implements the TranscriptionAdapter interface for FireRedASR2-AED
type FireRedAdapter struct {
	*BaseAdapter
	envPath   string
	modelDir  string
	sourceDir string
}

// NewFireRedAdapter creates a new FireRedASR2-AED adapter
func NewFireRedAdapter(envPath, modelDir string) *FireRedAdapter {
	capabilities := interfaces.ModelCapabilities{
		ModelID:            "firered_asr",
		ModelFamily:        "firered",
		DisplayName:        "FireRedASR2-AED 1.1B",
		Description:        "FireRedASR2-AED for Chinese/English transcription with punctuation",
		Version:            "2.0.0",
		SupportedLanguages: []string{"zh", "en"},
		SupportedFormats:   []string{"wav", "mp3", "flac", "m4a"},
		RequiresGPU:        true,
		MemoryRequirement:  4096,
		Features: map[string]bool{
			"timestamps":        true,
			"word_level":        false,
			"chinese_optimized": true,
			"punctuation":       true,
		},
		Metadata: map[string]string{
			"engine":      "fireredasr2s",
			"framework":   "pytorch",
			"license":     "Apache-2.0",
			"sample_rate": "16000",
		},
	}

	schema := []interfaces.ParameterSchema{
		{
			Name:        "beam_size",
			Type:        "int",
			Required:    false,
			Default:     3,
			Min:         &[]float64{1}[0],
			Max:         &[]float64{10}[0],
			Description: "Beam size for decoding",
			Group:       "basic",
		},
		{
			Name:        "timestamps",
			Type:        "bool",
			Required:    false,
			Default:     false,
			Description: "Include timestamps in output",
			Group:       "basic",
		},
		{
			Name:        "use_punc",
			Type:        "bool",
			Required:    false,
			Default:     true,
			Description: "Enable punctuation restoration",
			Group:       "basic",
		},
		{
			Name:        "auto_convert_audio",
			Type:        "bool",
			Required:    false,
			Default:     true,
			Description: "Automatically convert audio to 16kHz mono WAV",
			Group:       "advanced",
		},
	}

	baseAdapter := NewBaseAdapter("firered_asr", envPath, capabilities, schema)

	return &FireRedAdapter{
		BaseAdapter: baseAdapter,
		envPath:     envPath,
		modelDir:    modelDir,
		sourceDir:   resolveFireRedSourceDir(modelDir),
	}
}

func resolveFireRedSourceDir(modelDir string) string {
	if sourceDir := strings.TrimSpace(os.Getenv("FIRERED_SOURCE_DIR")); sourceDir != "" {
		return sourceDir
	}

	if modelDir == "" {
		return ""
	}

	// Typical layout: <repo>/pretrained_models/FireRedASR2-AED
	return filepath.Dir(filepath.Dir(modelDir))
}

// GetSupportedModels returns the available FireRedASR2 models
func (f *FireRedAdapter) GetSupportedModels() []string {
	return []string{"firered-asr2-aed"}
}

// PrepareEnvironment sets up the FireRedASR2 environment
func (f *FireRedAdapter) PrepareEnvironment(ctx context.Context) error {
	logger.Info("Preparing FireRedASR2 environment", "env_path", f.envPath)

	if err := f.copyEmbeddedScripts(); err != nil {
		return fmt.Errorf("failed to copy embedded scripts: %w", err)
	}

	if CheckEnvironmentReady(f.envPath, "from fireredasr2s.fireredasr2 import FireRedAsr2") {
		logger.Info("FireRedASR2 environment already ready")
		f.initialized = true
		return nil
	}

	if err := f.setupFireRedEnvironment(); err != nil {
		return fmt.Errorf("failed to setup FireRedASR2 environment: %w", err)
	}

	f.initialized = true
	logger.Info("FireRedASR2 environment prepared successfully")
	return nil
}

func (f *FireRedAdapter) setupFireRedEnvironment() error {
	if err := os.MkdirAll(f.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create firered directory: %w", err)
	}

	pyprojectContent, err := fireredScripts.ReadFile("py/firered/pyproject.toml")
	if err != nil {
		return fmt.Errorf("failed to read embedded pyproject.toml: %w", err)
	}

	contentStr := strings.Replace(
		string(pyprojectContent),
		"https://download.pytorch.org/whl/cu126",
		GetPyTorchWheelURL(),
		1,
	)

	pyprojectPath := filepath.Join(f.envPath, "pyproject.toml")
	if err := os.WriteFile(pyprojectPath, []byte(contentStr), 0644); err != nil {
		return fmt.Errorf("failed to write pyproject.toml: %w", err)
	}

	logger.Info("Installing FireRedASR2 dependencies")
	cmd := exec.Command("uv", "sync", "--native-tls")
	cmd.Dir = f.envPath
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("uv sync failed: %w: %s", err, strings.TrimSpace(string(out)))
	}

	return nil
}

func (f *FireRedAdapter) copyEmbeddedScripts() error {
	if err := os.MkdirAll(f.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	scripts := []string{
		"firered_transcribe.py",
		"firered_realtime_worker.py",
	}

	for _, scriptName := range scripts {
		content, err := fireredScripts.ReadFile(filepath.Join("py/firered", scriptName))
		if err != nil {
			return fmt.Errorf("failed to read embedded %s: %w", scriptName, err)
		}

		scriptPath := filepath.Join(f.envPath, scriptName)
		if err := os.WriteFile(scriptPath, content, 0755); err != nil {
			return fmt.Errorf("failed to write %s: %w", scriptName, err)
		}
	}

	return nil
}

// Transcribe processes audio using FireRedASR2-AED
func (f *FireRedAdapter) Transcribe(ctx context.Context, input interfaces.AudioInput, params map[string]interface{}, procCtx interfaces.ProcessingContext) (*interfaces.TranscriptResult, error) {
	startTime := time.Now()
	f.LogProcessingStart(input, procCtx)
	defer func() {
		f.LogProcessingEnd(procCtx, time.Since(startTime), nil)
	}()

	if err := f.ValidateAudioInput(input); err != nil {
		return nil, fmt.Errorf("invalid audio input: %w", err)
	}

	if err := f.ValidateParameters(params); err != nil {
		return nil, fmt.Errorf("invalid parameters: %w", err)
	}

	tempDir, err := f.CreateTempDirectory(procCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer f.CleanupTempDirectory(tempDir)

	audioInput := input
	if f.GetBoolParameter(params, "auto_convert_audio") {
		convertedInput, err := f.ConvertAudioFormat(ctx, input, "wav", 16000)
		if err != nil {
			logger.Warn("Audio conversion failed, using original", "error", err)
		} else {
			audioInput = convertedInput
		}
	}

	args, err := f.buildFireRedArgs(audioInput, params, tempDir)
	if err != nil {
		return nil, fmt.Errorf("failed to build command: %w", err)
	}

	cmd := exec.CommandContext(ctx, "uv", args...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

	logFile, err := os.OpenFile(filepath.Join(procCtx.OutputDirectory, "transcription.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		logger.Warn("Failed to create log file", "error", err)
	} else {
		defer logFile.Close()
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}

	logger.Info("Executing FireRedASR2 command", "args", strings.Join(args, " "))

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.Canceled {
			return nil, fmt.Errorf("transcription was cancelled")
		}

		logPath := filepath.Join(procCtx.OutputDirectory, "transcription.log")
		logTail, readErr := f.ReadLogTail(logPath, 2048)
		if readErr != nil {
			logger.Warn("Failed to read log tail", "error", readErr)
		}

		logger.Error("FireRedASR2 execution failed", "error", err)
		return nil, fmt.Errorf("FireRedASR2 execution failed: %w\nLogs:\n%s", err, logTail)
	}

	result, err := f.parseResult(tempDir)
	if err != nil {
		return nil, fmt.Errorf("failed to parse result: %w", err)
	}

	result.ProcessingTime = time.Since(startTime)
	result.ModelUsed = "FireRedASR2-AED"
	result.Metadata = f.CreateDefaultMetadata(params)

	logger.Info("FireRedASR2 transcription completed",
		"segments", len(result.Segments),
		"processing_time", result.ProcessingTime)

	return result, nil
}

func (f *FireRedAdapter) buildFireRedArgs(input interfaces.AudioInput, params map[string]interface{}, tempDir string) ([]string, error) {
	outputFile := filepath.Join(tempDir, "result.json")

	scriptPath := filepath.Join(f.envPath, "firered_transcribe.py")
	args := []string{
		"run", "--native-tls", "--project", f.envPath, "python", scriptPath,
		input.FilePath,
		"--output", outputFile,
		"--model-dir", f.modelDir,
	}

	if f.sourceDir != "" {
		args = append(args, "--source-dir", f.sourceDir)
	}

	beamSize := f.GetIntParameter(params, "beam_size")
	if beamSize > 0 {
		args = append(args, "--beam-size", strconv.Itoa(beamSize))
	}

	if f.GetBoolParameter(params, "timestamps") {
		args = append(args, "--timestamps")
	}

	if f.GetBoolParameter(params, "use_punc") {
		puncDir := filepath.Join(filepath.Dir(f.modelDir), "FireRedPunc")
		if _, err := os.Stat(puncDir); err == nil {
			args = append(args, "--punc-model-dir", puncDir)
		}
	} else {
		args = append(args, "--no-punc")
	}

	// Use FireRed's non-stream VAD model for long-audio segmentation to avoid OOM.
	vadDir := filepath.Join(filepath.Dir(f.modelDir), "FireRedVAD", "VAD")
	if _, err := os.Stat(vadDir); err == nil {
		args = append(args, "--vad-model-dir", vadDir)
	}
	args = append(args, "--max-segment-seconds", "18")

	return args, nil
}

func (f *FireRedAdapter) parseResult(tempDir string) (*interfaces.TranscriptResult, error) {
	resultFile := filepath.Join(tempDir, "result.json")

	data, err := os.ReadFile(resultFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read result file: %w", err)
	}

	var fireRedResult struct {
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

	if err := json.Unmarshal(data, &fireRedResult); err != nil {
		return nil, fmt.Errorf("failed to parse JSON result: %w", err)
	}

	result := &interfaces.TranscriptResult{
		Text:     fireRedResult.Transcription,
		Language: fireRedResult.Language,
		Segments: make([]interfaces.TranscriptSegment, len(fireRedResult.SegmentTimestamps)),
	}

	for i, seg := range fireRedResult.SegmentTimestamps {
		result.Segments[i] = interfaces.TranscriptSegment{
			Start: seg.Start,
			End:   seg.End,
			Text:  seg.Segment,
		}
	}

	if len(fireRedResult.WordTimestamps) > 0 {
		result.WordSegments = make([]interfaces.TranscriptWord, len(fireRedResult.WordTimestamps))
		for i, word := range fireRedResult.WordTimestamps {
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

// GetEstimatedProcessingTime provides FireRedASR2-specific time estimation
func (f *FireRedAdapter) GetEstimatedProcessingTime(input interfaces.AudioInput) time.Duration {
	baseTime := f.BaseAdapter.GetEstimatedProcessingTime(input)
	return time.Duration(float64(baseTime) * 0.2)
}
