package adapters

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"scriberr/internal/transcription/interfaces"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveFireRedSourceDirPrefersEnv(t *testing.T) {
	t.Setenv("FIRERED_SOURCE_DIR", "/tmp/firered-source")
	assert.Equal(t, "/tmp/firered-source", resolveFireRedSourceDir("/models/FireRedASR2-AED"))
}

func TestResolveFireRedSourceDirFromModelLayout(t *testing.T) {
	t.Setenv("FIRERED_SOURCE_DIR", "")
	assert.Equal(t, "/opt/FireRedASR2S", resolveFireRedSourceDir("/opt/FireRedASR2S/pretrained_models/FireRedASR2-AED"))
}

func TestBuildFireRedArgsIncludesSourceDirNoPuncAndVAD(t *testing.T) {
	t.Setenv("FIRERED_SOURCE_DIR", "/opt/FireRedASR2S")

	modelsRoot := t.TempDir()
	modelDir := filepath.Join(modelsRoot, "FireRedASR2-AED")
	vadDir := filepath.Join(modelsRoot, "FireRedVAD", "VAD")
	require.NoError(t, os.MkdirAll(modelDir, 0o755))
	require.NoError(t, os.MkdirAll(vadDir, 0o755))

	adapter := NewFireRedAdapter(t.TempDir(), modelDir)
	args, err := adapter.buildFireRedArgs(interfaces.AudioInput{FilePath: "/tmp/input.wav"}, map[string]interface{}{
		"beam_size": 4,
		"use_punc":  false,
	}, t.TempDir())

	require.NoError(t, err)
	assert.Contains(t, args, "--source-dir")
	assert.Contains(t, args, "/opt/FireRedASR2S")
	assert.Contains(t, args, "--beam-size")
	assert.Contains(t, args, "4")
	assert.Contains(t, args, "--model-type")
	assert.Contains(t, args, "aed")
	assert.Contains(t, args, "--no-punc")
	assert.Contains(t, args, "--vad-model-dir")
	assert.Contains(t, args, vadDir)
	assert.Contains(t, args, "--max-segment-seconds")
	assert.Contains(t, args, "18")
}

func TestBuildFireRedArgsUsesLLMVariantWhenModelDirExists(t *testing.T) {
	t.Setenv("FIRERED_SOURCE_DIR", "/opt/FireRedASR2S")

	modelsRoot := t.TempDir()
	aedDir := filepath.Join(modelsRoot, "FireRedASR2-AED")
	llmDir := filepath.Join(modelsRoot, "FireRedASR2-LLM")
	require.NoError(t, os.MkdirAll(aedDir, 0o755))
	require.NoError(t, os.MkdirAll(llmDir, 0o755))
	t.Setenv("FIRERED_MODEL_DIR_LLM", llmDir)

	adapter := NewFireRedAdapter(t.TempDir(), aedDir)
	args, err := adapter.buildFireRedArgs(interfaces.AudioInput{FilePath: "/tmp/input.wav"}, map[string]interface{}{
		"model_variant": "llm8b",
	}, t.TempDir())

	require.NoError(t, err)
	assert.Contains(t, args, "--model-dir")
	assert.Contains(t, args, llmDir)
	assert.Contains(t, args, "--model-type")
	assert.Contains(t, args, "llm")
}

func TestParseFireRedResult(t *testing.T) {
	adapter := NewFireRedAdapter(t.TempDir(), "/models/FireRedASR2-AED")
	tempDir := t.TempDir()

	output := map[string]interface{}{
		"transcription": "hello world",
		"language":      "zh",
		"segment_timestamps": []map[string]interface{}{
			{"segment": "hello world", "start": 0.0, "end": 1.2},
		},
	}
	data, err := json.Marshal(output)
	require.NoError(t, err)

	require.NoError(t, os.WriteFile(filepath.Join(tempDir, "result.json"), data, 0o644))

	result, err := adapter.parseResult(tempDir)
	require.NoError(t, err)
	assert.Equal(t, "hello world", result.Text)
	assert.Len(t, result.Segments, 1)
	assert.Equal(t, 1.2, result.Segments[0].End)
}

func TestEmbeddedFireRedPyprojectIncludesRuntimeDependencies(t *testing.T) {
	pyproject, err := fireredScripts.ReadFile("py/firered/pyproject.toml")
	require.NoError(t, err)

	content := string(pyproject)
	for _, dep := range []string{
		"\"kaldi-native-fbank\"",
		"\"transformers==4.51.3\"",
		"\"accelerate\"",
		"\"peft\"",
		"\"cn2an\"",
		"\"textgrid\"",
	} {
		assert.Contains(t, content, dep)
	}
}

func TestCopyEmbeddedScriptsIncludesRealtimeWorker(t *testing.T) {
	adapter := NewFireRedAdapter(t.TempDir(), "/models/FireRedASR2-AED")
	require.NoError(t, adapter.copyEmbeddedScripts())

	_, err := os.Stat(filepath.Join(adapter.envPath, "firered_transcribe.py"))
	require.NoError(t, err)

	_, err = os.Stat(filepath.Join(adapter.envPath, "firered_realtime_worker.py"))
	require.NoError(t, err)
}

func TestEmbeddedFireRedTranscribeUsesVADSegmentation(t *testing.T) {
	script, err := fireredScripts.ReadFile("py/firered/firered_transcribe.py")
	require.NoError(t, err)

	content := string(script)
	assert.Contains(t, content, "non_stream_vad")
	assert.Contains(t, content, "--vad-model-dir")
	assert.Contains(t, content, "max_segment_seconds")
}
