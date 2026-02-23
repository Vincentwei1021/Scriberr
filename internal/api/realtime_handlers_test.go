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

func TestResolveRealtimeModelDirs(t *testing.T) {
	vad, punc := resolveRealtimeModelDirs("/app/models/FireRedASR2-AED")
	assert.Equal(t, "/app/models/FireRedVAD/Stream-VAD", vad)
	assert.Equal(t, "/app/models/FireRedPunc", punc)
}

func TestBuildRealtimeWorkerArgs(t *testing.T) {
	envPath := t.TempDir()
	scriptPath := filepath.Join(envPath, "firered_realtime_worker.py")
	puncDir := filepath.Join(envPath, "FireRedPunc")
	require.NoError(t, os.MkdirAll(puncDir, 0o755))

	args := buildRealtimeWorkerArgs(
		envPath,
		scriptPath,
		"/app/models/FireRedASR2-AED",
		"/app/FireRedASR2S",
		"/app/models/FireRedVAD/Stream-VAD",
		puncDir,
	)

	assert.Contains(t, args, "--project")
	assert.Contains(t, args, envPath)
	assert.Contains(t, args, "--model-dir")
	assert.Contains(t, args, "/app/models/FireRedASR2-AED")
	assert.Contains(t, args, "--source-dir")
	assert.Contains(t, args, "/app/FireRedASR2S")
	assert.Contains(t, args, "--vad-model-dir")
	assert.Contains(t, args, "/app/models/FireRedVAD/Stream-VAD")
	assert.Contains(t, args, "--punc-model-dir")
	assert.Contains(t, args, puncDir)
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
