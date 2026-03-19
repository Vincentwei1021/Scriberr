package api

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestBuildRealtimeWorkerArgs(t *testing.T) {
	envPath := t.TempDir()
	scriptPath := filepath.Join(envPath, "qwen3_realtime_worker.py")
	vadDir := filepath.Join(envPath, "FireRedVAD", "Stream-VAD")
	require.NoError(t, os.MkdirAll(vadDir, 0o755))

	args := buildRealtimeWorkerArgs(envPath, scriptPath, realtimeQwenModel, "/opt/FireRedASR2S", vadDir)

	assert.Contains(t, args, "--project")
	assert.Contains(t, args, envPath)
	assert.Contains(t, args, "--model")
	assert.Contains(t, args, realtimeQwenModel)
	assert.Contains(t, args, "--source-dir")
	assert.Contains(t, args, "/opt/FireRedASR2S")
	assert.Contains(t, args, "--vad-model-dir")
	assert.Contains(t, args, vadDir)
}

func TestResolveRealtimeModelDirs(t *testing.T) {
	vadDir := resolveRealtimeModelDirs("/app/models/FireRedASR2-AED")
	assert.Equal(t, "/app/models/FireRedVAD/Stream-VAD", vadDir)
}

func TestWriteLengthPrefixedChunk(t *testing.T) {
	buf := &bytes.Buffer{}
	payload := []byte{1, 2, 3, 4}

	require.NoError(t, writeLengthPrefixedChunk(buf, payload))
	out := buf.Bytes()
	require.Len(t, out, 8)

	length := binary.LittleEndian.Uint32(out[:4])
	assert.Equal(t, uint32(4), length)
	assert.Equal(t, payload, out[4:])

	buf.Reset()
	require.NoError(t, writeLengthPrefixedChunk(buf, nil))
	out = buf.Bytes()
	require.Len(t, out, 4)
	assert.Equal(t, uint32(0), binary.LittleEndian.Uint32(out[:4]))
}
