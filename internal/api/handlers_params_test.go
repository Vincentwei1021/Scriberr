package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"scriberr/internal/models"
	"scriberr/internal/transcription"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newJSONContext(body string) *gin.Context {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/transcription/job/start", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	c.Request = req
	return c
}

func TestGetValidatedTranscriptionParams_ForceCudaForFireRedAndCAMPP(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := &Handler{}

	ctx := newJSONContext(`{"model_family":"firered","model":"firered-asr2-aed","device":"cpu","diarize":true,"diarize_model":"pyannote"}`)
	params, err := h.getValidatedTranscriptionParams(ctx, &models.TranscriptionJob{}, "job-firered")

	require.NoError(t, err)
	require.NotNil(t, params)
	assert.Equal(t, "cuda", params.Device)
	assert.Equal(t, transcription.DiarizeCAMPP, params.DiarizeModel)
}

func TestGetValidatedTranscriptionParams_ForceCudaForQwen(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := &Handler{}

	ctx := newJSONContext(`{"model_family":"qwen","model":"Qwen/Qwen3-ASR-1.7B","device":"cpu","diarize":false}`)
	params, err := h.getValidatedTranscriptionParams(ctx, &models.TranscriptionJob{}, "job-qwen")

	require.NoError(t, err)
	require.NotNil(t, params)
	assert.Equal(t, "cuda", params.Device)
}

func TestGetValidatedTranscriptionParams_KeepWhisperDevice(t *testing.T) {
	gin.SetMode(gin.TestMode)
	h := &Handler{}

	ctx := newJSONContext(`{"model_family":"whisper","model":"small","device":"cpu","diarize":false}`)
	params, err := h.getValidatedTranscriptionParams(ctx, &models.TranscriptionJob{}, "job-whisper")

	require.NoError(t, err)
	require.NotNil(t, params)
	assert.Equal(t, "cpu", params.Device)
}
