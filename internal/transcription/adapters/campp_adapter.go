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

//go:embed py/campp/*
var camppScripts embed.FS

// CAMPPAdapter implements the DiarizationAdapter interface for FunASR CAM++
type CAMPPAdapter struct {
	*BaseAdapter
	envPath string
}

// NewCAMPPAdapter creates a new CAM++ diarization adapter
func NewCAMPPAdapter(envPath string) *CAMPPAdapter {
	capabilities := interfaces.ModelCapabilities{
		ModelID:            "campp",
		ModelFamily:        "funasr_campp",
		DisplayName:        "FunASR CAM++ Speaker Diarization",
		Description:        "CAM++ speaker diarization optimized for Chinese speech",
		Version:            "1.0.0",
		SupportedLanguages: []string{"*"}, // Language-agnostic
		SupportedFormats:   []string{"wav", "mp3", "flac", "m4a"},
		RequiresGPU:        false, // Optional GPU support
		MemoryRequirement:  2048,  // 2GB recommended
		Features: map[string]bool{
			"speaker_detection":  true,
			"confidence_scores":  true,
			"flexible_speakers":  true,
		},
		Metadata: map[string]string{
			"engine":    "funasr",
			"framework": "pytorch",
			"license":   "Apache-2.0",
			"model_hub": "modelscope",
		},
	}

	schema := []interfaces.ParameterSchema{
		{
			Name:        "min_speakers",
			Type:        "int",
			Required:    false,
			Default:     nil,
			Min:         &[]float64{1}[0],
			Max:         &[]float64{20}[0],
			Description: "Minimum number of speakers",
			Group:       "basic",
		},
		{
			Name:        "max_speakers",
			Type:        "int",
			Required:    false,
			Default:     nil,
			Min:         &[]float64{1}[0],
			Max:         &[]float64{20}[0],
			Description: "Maximum number of speakers",
			Group:       "basic",
		},
		{
			Name:        "output_format",
			Type:        "string",
			Required:    false,
			Default:     "json",
			Options:     []string{"json", "rttm"},
			Description: "Output format for diarization results",
			Group:       "advanced",
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

	baseAdapter := NewBaseAdapter("campp", envPath, capabilities, schema)

	return &CAMPPAdapter{
		BaseAdapter: baseAdapter,
		envPath:     envPath,
	}
}

// GetMaxSpeakers returns the maximum number of speakers CAM++ can handle
func (c *CAMPPAdapter) GetMaxSpeakers() int {
	return 20
}

// GetMinSpeakers returns the minimum number of speakers CAM++ requires
func (c *CAMPPAdapter) GetMinSpeakers() int {
	return 1
}

// PrepareEnvironment sets up the dedicated CAM++ environment
func (c *CAMPPAdapter) PrepareEnvironment(ctx context.Context) error {
	logger.Info("Preparing CAM++ environment", "env_path", c.envPath)

	// Always ensure diarization script exists
	if err := c.copyDiarizationScript(); err != nil {
		return fmt.Errorf("failed to create diarization script: %w", err)
	}

	// Check if FunASR is already available
	if CheckEnvironmentReady(c.envPath, "from funasr import AutoModel") {
		logger.Info("FunASR already available in environment")
		c.initialized = true
		return nil
	}

	// Create environment if it doesn't exist or is incomplete
	if err := c.setupCAMPPEnvironment(); err != nil {
		return fmt.Errorf("failed to setup CAM++ environment: %w", err)
	}

	// Verify FunASR is now available
	testCmd := exec.Command("uv", "run", "--native-tls", "--project", c.envPath, "python", "-c", "from funasr import AutoModel")
	if testCmd.Run() != nil {
		logger.Warn("CAM++ environment test still failed after setup")
	}

	c.initialized = true
	logger.Info("CAM++ environment prepared successfully")
	return nil
}

// setupCAMPPEnvironment creates the Python environment
func (c *CAMPPAdapter) setupCAMPPEnvironment() error {
	if err := os.MkdirAll(c.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create campp directory: %w", err)
	}

	// Read pyproject.toml for CAM++
	pyprojectContent, err := camppScripts.ReadFile("py/campp/pyproject.toml")
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

	pyprojectPath := filepath.Join(c.envPath, "pyproject.toml")
	if err := os.WriteFile(pyprojectPath, []byte(contentStr), 0644); err != nil {
		return fmt.Errorf("failed to write pyproject.toml: %w", err)
	}

	// Run uv sync
	logger.Info("Installing CAM++ dependencies")
	cmd := exec.Command("uv", "sync", "--native-tls")
	cmd.Dir = c.envPath
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("uv sync failed: %w: %s", err, strings.TrimSpace(string(out)))
	}

	return nil
}

// copyDiarizationScript creates the Python script for CAM++ diarization
func (c *CAMPPAdapter) copyDiarizationScript() error {
	if err := os.MkdirAll(c.envPath, 0755); err != nil {
		return fmt.Errorf("failed to create campp directory: %w", err)
	}

	scriptContent, err := camppScripts.ReadFile("py/campp/campp_diarize.py")
	if err != nil {
		return fmt.Errorf("failed to read embedded campp_diarize.py: %w", err)
	}

	scriptPath := filepath.Join(c.envPath, "campp_diarize.py")
	if err := os.WriteFile(scriptPath, scriptContent, 0755); err != nil {
		return fmt.Errorf("failed to write diarization script: %w", err)
	}

	return nil
}

// Diarize processes audio using FunASR CAM++
func (c *CAMPPAdapter) Diarize(ctx context.Context, input interfaces.AudioInput, params map[string]interface{}, procCtx interfaces.ProcessingContext) (*interfaces.DiarizationResult, error) {
	startTime := time.Now()
	c.LogProcessingStart(input, procCtx)
	defer func() {
		c.LogProcessingEnd(procCtx, time.Since(startTime), nil)
	}()

	// Validate input
	if err := c.ValidateAudioInput(input); err != nil {
		return nil, fmt.Errorf("invalid audio input: %w", err)
	}

	// Validate parameters
	if err := c.ValidateParameters(params); err != nil {
		return nil, fmt.Errorf("invalid parameters: %w", err)
	}

	// Create temporary directory
	tempDir, err := c.CreateTempDirectory(procCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %w", err)
	}
	defer c.CleanupTempDirectory(tempDir)

	// Build command arguments
	args, err := c.buildCAMPPArgs(input, params, tempDir)
	if err != nil {
		return nil, fmt.Errorf("failed to build command: %w", err)
	}

	// Execute CAM++
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

	logger.Info("Executing CAM++ command", "args", strings.Join(args, " "))

	if err := cmd.Run(); err != nil {
		if ctx.Err() == context.Canceled {
			return nil, fmt.Errorf("diarization was cancelled")
		}

		logPath := filepath.Join(procCtx.OutputDirectory, "transcription.log")
		logTail, readErr := c.ReadLogTail(logPath, 2048)
		if readErr != nil {
			logger.Warn("Failed to read log tail", "error", readErr)
		}

		logger.Error("CAM++ execution failed", "error", err)
		return nil, fmt.Errorf("CAM++ execution failed: %w\nLogs:\n%s", err, logTail)
	}

	// Parse result
	result, err := c.parseResult(tempDir, input, params)
	if err != nil {
		return nil, fmt.Errorf("failed to parse result: %w", err)
	}

	result.ProcessingTime = time.Since(startTime)
	result.ModelUsed = "FunASR-CAM++"
	result.Metadata = c.CreateDefaultMetadata(params)

	logger.Info("CAM++ diarization completed",
		"segments", len(result.Segments),
		"speakers", result.SpeakerCount,
		"processing_time", result.ProcessingTime)

	return result, nil
}

// buildCAMPPArgs builds the command arguments for CAM++
func (c *CAMPPAdapter) buildCAMPPArgs(input interfaces.AudioInput, params map[string]interface{}, tempDir string) ([]string, error) {
	outputFormat := c.GetStringParameter(params, "output_format")
	var outputFile string
	if outputFormat == OutputFormatJSON {
		outputFile = filepath.Join(tempDir, "result.json")
	} else {
		outputFile = filepath.Join(tempDir, "result.rttm")
	}

	scriptPath := filepath.Join(c.envPath, "campp_diarize.py")
	args := []string{
		"run", "--native-tls", "--project", c.envPath, "python", scriptPath,
		input.FilePath,
		"--output", outputFile,
	}

	// Add speaker constraints
	if minSpeakers := c.GetIntParameter(params, "min_speakers"); minSpeakers > 0 {
		args = append(args, "--min-speakers", strconv.Itoa(minSpeakers))
	}
	if maxSpeakers := c.GetIntParameter(params, "max_speakers"); maxSpeakers > 0 {
		args = append(args, "--max-speakers", strconv.Itoa(maxSpeakers))
	}

	// Add output format
	args = append(args, "--output-format", outputFormat)

	return args, nil
}

// parseResult parses the CAM++ output
func (c *CAMPPAdapter) parseResult(tempDir string, input interfaces.AudioInput, params map[string]interface{}) (*interfaces.DiarizationResult, error) {
	outputFormat := c.GetStringParameter(params, "output_format")

	if outputFormat == OutputFormatJSON {
		return c.parseJSONResult(tempDir)
	}
	return c.parseRTTMResult(tempDir, input)
}

// parseJSONResult parses JSON format output
func (c *CAMPPAdapter) parseJSONResult(tempDir string) (*interfaces.DiarizationResult, error) {
	resultFile := filepath.Join(tempDir, "result.json")

	data, err := os.ReadFile(resultFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read result file: %w", err)
	}

	var camppResult struct {
		AudioFile string `json:"audio_file"`
		Model     string `json:"model"`
		Segments  []struct {
			Start      float64 `json:"start"`
			End        float64 `json:"end"`
			Speaker    string  `json:"speaker"`
			Confidence float64 `json:"confidence"`
			Duration   float64 `json:"duration"`
		} `json:"segments"`
		Speakers      []string `json:"speakers"`
		SpeakerCount  int      `json:"speaker_count"`
		TotalDuration float64  `json:"total_duration"`
	}

	if err := json.Unmarshal(data, &camppResult); err != nil {
		return nil, fmt.Errorf("failed to parse JSON result: %w", err)
	}

	result := &interfaces.DiarizationResult{
		Segments:     make([]interfaces.DiarizationSegment, len(camppResult.Segments)),
		SpeakerCount: camppResult.SpeakerCount,
		Speakers:     camppResult.Speakers,
	}

	for i, seg := range camppResult.Segments {
		result.Segments[i] = interfaces.DiarizationSegment{
			Start:      seg.Start,
			End:        seg.End,
			Speaker:    seg.Speaker,
			Confidence: seg.Confidence,
		}
	}

	return result, nil
}

// parseRTTMResult parses RTTM format output
func (c *CAMPPAdapter) parseRTTMResult(tempDir string, input interfaces.AudioInput) (*interfaces.DiarizationResult, error) {
	resultFile := filepath.Join(tempDir, "result.rttm")

	data, err := os.ReadFile(resultFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read result file: %w", err)
	}

	var segments []interfaces.DiarizationSegment
	speakers := make(map[string]bool)

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "SPEAKER") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) < 8 {
			continue
		}

		start, err := strconv.ParseFloat(parts[3], 64)
		if err != nil {
			continue
		}

		duration, err := strconv.ParseFloat(parts[4], 64)
		if err != nil {
			continue
		}

		end := start + duration
		speaker := parts[7]
		speakers[speaker] = true

		segments = append(segments, interfaces.DiarizationSegment{
			Start:      start,
			End:        end,
			Speaker:    speaker,
			Confidence: 1.0,
		})
	}

	speakerList := make([]string, 0, len(speakers))
	for speaker := range speakers {
		speakerList = append(speakerList, speaker)
	}

	return &interfaces.DiarizationResult{
		Segments:     segments,
		SpeakerCount: len(speakers),
		Speakers:     speakerList,
	}, nil
}

// GetEstimatedProcessingTime provides CAM++-specific time estimation
func (c *CAMPPAdapter) GetEstimatedProcessingTime(input interfaces.AudioInput) time.Duration {
	baseTime := c.BaseAdapter.GetEstimatedProcessingTime(input)
	// CAM++ is typically fast for diarization
	return time.Duration(float64(baseTime) * 0.5)
}
