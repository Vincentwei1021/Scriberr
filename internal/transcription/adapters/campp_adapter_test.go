package adapters

import (
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
	}, t.TempDir())
	require.NoError(t, err)

	assert.Contains(t, args, "--min-speakers")
	assert.Contains(t, args, "2")
	assert.Contains(t, args, "--max-speakers")
	assert.Contains(t, args, "5")
	assert.Contains(t, args, "--output-format")
	assert.Contains(t, args, "json")
}

func TestEmbeddedCAMPPPyprojectIncludesTorchcodec(t *testing.T) {
	pyproject, err := camppScripts.ReadFile("py/campp/pyproject.toml")
	require.NoError(t, err)

	assert.Contains(t, string(pyproject), "\"torchcodec\"")
}
