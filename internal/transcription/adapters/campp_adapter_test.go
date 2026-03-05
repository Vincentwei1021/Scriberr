package adapters

import (
	"path/filepath"
	"testing"

	"scriberr/internal/transcription/interfaces"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildCAMPPArgsIncludesSpeakerBoundsAndFormat(t *testing.T) {
	adapter := NewCAMPPAdapter(t.TempDir())

	args, err := adapter.buildCAMPPArgs(interfaces.AudioInput{FilePath: "/tmp/input.wav"}, map[string]interface{}{
		"min_speakers":  2,
		"max_speakers":  5,
		"output_format": "json",
		"diarize_model": "funasr_diarizen_large",
	}, t.TempDir())
	require.NoError(t, err)

	assert.Contains(t, args, "--min-speakers")
	assert.Contains(t, args, "2")
	assert.Contains(t, args, "--max-speakers")
	assert.Contains(t, args, "5")
	assert.Contains(t, args, "--output-format")
	assert.Contains(t, args, "json")
	assert.Contains(t, args, "--speaker-model")
	assert.Contains(t, args, "iic/speech_campplus_sv_zh-cn_16k-common")
	assert.Contains(t, args, "--diarize-model")
	assert.Contains(t, args, "funasr_diarizen_large")
}

func TestEmbeddedCAMPPPyprojectIncludesTorchcodec(t *testing.T) {
	pyproject, err := camppScripts.ReadFile("py/campp/pyproject.toml")
	require.NoError(t, err)

	assert.Contains(t, string(pyproject), "\"torchcodec\"")
}

func TestBuildCAMPPArgsUsesDiariZenProjectPath(t *testing.T) {
	envRoot := t.TempDir()
	adapter := NewCAMPPAdapter(filepath.Join(envRoot, "campp"))

	args, err := adapter.buildCAMPPArgs(interfaces.AudioInput{FilePath: "/tmp/input.wav"}, map[string]interface{}{
		"output_format": "json",
		"diarize_model": "funasr_diarizen_large",
	}, t.TempDir())
	require.NoError(t, err)

	assert.Contains(t, args, "--project")
	assert.Contains(t, args, filepath.Join(envRoot, "diarizen"))
}

func TestEmbeddedDiariZenPyprojectIncludesBUTDependencies(t *testing.T) {
	pyproject, err := camppScripts.ReadFile("py/diarizen/pyproject.toml")
	require.NoError(t, err)

	content := string(pyproject)
	assert.Contains(t, content, "diarizen @ git+https://github.com/BUTSpeechFIT/DiariZen.git")
	assert.Contains(t, content, "pyannote-audio @ git+https://github.com/BUTSpeechFIT/DiariZen.git#subdirectory=pyannote-audio")
}
