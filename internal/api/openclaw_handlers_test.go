package api

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildSRTFromRawTranscriptSegments(t *testing.T) {
	raw := `{"segments":[{"start":0.1,"end":2.5,"text":"Hello","speaker":"speaker_00"},{"start":2.5,"end":4.0,"text":"Hi there"}]}`
	speakerMap := map[string]string{"speaker_00": "Alice"}

	srt, err := buildSRTFromRawTranscript(raw, speakerMap)
	require.NoError(t, err)
	assert.Contains(t, srt, "Alice: Hello")
	assert.Contains(t, srt, "Hi there")
	assert.Contains(t, srt, "00:00:00,100 --> 00:00:02,500")
}

func TestBuildSRTFromRawTranscriptTextFallback(t *testing.T) {
	raw := `{"text":"Full transcript text"}`
	srt, err := buildSRTFromRawTranscript(raw, nil)
	require.NoError(t, err)
	assert.Contains(t, srt, "Full transcript text")
	assert.Contains(t, srt, "00:00:00,000 --> 00:00:05,000")
}

func TestBuildSRTFromRawTranscriptInvalid(t *testing.T) {
	_, err := buildSRTFromRawTranscript(`{"segments":[{"start":0}]}`, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no usable transcript segments")
}
