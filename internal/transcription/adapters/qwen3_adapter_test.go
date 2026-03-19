package adapters

import (
	"os"
	"path/filepath"
	"testing"

	"scriberr/internal/transcription/interfaces"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildQwen3ArgsIncludesFireRedVADAndSourceDir(t *testing.T) {
	t.Setenv("FIRERED_SOURCE_DIR", "/opt/FireRedASR2S")

	modelsRoot := t.TempDir()
	fireRedModelDir := filepath.Join(modelsRoot, "FireRedASR2-AED")
	vadDir := filepath.Join(modelsRoot, "FireRedVAD", "VAD")
	require.NoError(t, os.MkdirAll(fireRedModelDir, 0o755))
	require.NoError(t, os.MkdirAll(vadDir, 0o755))
	t.Setenv("FIRERED_MODEL_DIR", fireRedModelDir)

	adapter := NewQwen3ASRAdapter(t.TempDir())
	args, err := adapter.buildQwen3Args(interfaces.AudioInput{FilePath: "/tmp/input.wav"}, map[string]interface{}{
		"model": "Qwen/Qwen3-ASR-1.7B",
	}, t.TempDir())

	require.NoError(t, err)
	assert.Contains(t, args, "--source-dir")
	assert.Contains(t, args, "/opt/FireRedASR2S")
	assert.Contains(t, args, "--vad-model-dir")
	assert.Contains(t, args, vadDir)
	assert.Contains(t, args, "--max-segment-seconds")
	assert.Contains(t, args, "18")
}

func TestQwenEmbeddedPyprojectIncludesFireRedVADDependencies(t *testing.T) {
	pyproject, err := qwen3Scripts.ReadFile("py/qwen3/pyproject.toml")
	require.NoError(t, err)

	content := string(pyproject)
	for _, dep := range []string{
		"\"kaldi-native-fbank\"",
		"\"peft\"",
		"\"cn2an\"",
		"\"textgrid\"",
		"\"sentencepiece\"",
	} {
		assert.Contains(t, content, dep)
	}
}
