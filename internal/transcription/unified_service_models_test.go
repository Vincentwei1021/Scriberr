package transcription

import (
	"testing"

	"scriberr/internal/models"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSelectModelsWithChineseAdapters(t *testing.T) {
	service := NewUnifiedTranscriptionService(nil, t.TempDir(), t.TempDir())

	transcriptionModelID, diarizationModelID, err := service.selectModels(models.WhisperXParams{
		ModelFamily:  FamilyFireRed,
		Diarize:      true,
		DiarizeModel: DiarizeCAMPP,
	})
	require.NoError(t, err)
	assert.Equal(t, ModelFireRed, transcriptionModelID)
	assert.Equal(t, ModelCAMPP, diarizationModelID)

	transcriptionModelID, diarizationModelID, err = service.selectModels(models.WhisperXParams{
		ModelFamily: FamilyQwen,
		Diarize:     false,
	})
	require.NoError(t, err)
	assert.Equal(t, ModelQwen3, transcriptionModelID)
	assert.Empty(t, diarizationModelID)
}

func TestConvertParametersForQwenUsesSafeDefaultModel(t *testing.T) {
	service := NewUnifiedTranscriptionService(nil, t.TempDir(), t.TempDir())

	params := models.WhisperXParams{
		Model: "small",
	}

	converted := service.convertParametersForModel(params, ModelQwen3)
	assert.Equal(t, "Qwen/Qwen3-ASR-1.7B", converted["model"])
	assert.Equal(t, true, converted["auto_convert_audio"])

	params.Model = "Qwen/Qwen3-ASR-1.7B"
	converted = service.convertParametersForModel(params, ModelQwen3)
	assert.Equal(t, "Qwen/Qwen3-ASR-1.7B", converted["model"])
}

func TestConvertParametersForCAMPP(t *testing.T) {
	service := NewUnifiedTranscriptionService(nil, t.TempDir(), t.TempDir())

	minSpeakers := 2
	maxSpeakers := 6
	params := models.WhisperXParams{
		MinSpeakers: &minSpeakers,
		MaxSpeakers: &maxSpeakers,
	}

	converted := service.convertParametersForModel(params, ModelCAMPP)
	assert.Equal(t, OutputFormatJSON, converted["output_format"])
	assert.Equal(t, true, converted["auto_convert_audio"])
	assert.Equal(t, minSpeakers, converted["min_speakers"])
	assert.Equal(t, maxSpeakers, converted["max_speakers"])
}

func TestConvertParametersForFireRedUsesBeamSizeFallback(t *testing.T) {
	service := NewUnifiedTranscriptionService(nil, t.TempDir(), t.TempDir())

	converted := service.convertParametersForModel(models.WhisperXParams{
		BeamSize: 0,
	}, ModelFireRed)

	assert.Equal(t, 3, converted["beam_size"])
	assert.Equal(t, true, converted["use_punc"])
	assert.Equal(t, true, converted["auto_convert_audio"])
}
