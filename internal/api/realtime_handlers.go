package api

import (
	"bufio"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"scriberr/internal/transcription/adapters"
	"scriberr/pkg/logger"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"
)

var realtimeWSUpgrader = websocket.Upgrader{
	ReadBufferSize:  32 * 1024,
	WriteBufferSize: 32 * 1024,
	CheckOrigin: func(_ *http.Request) bool {
		return true
	},
}

type realtimeWorkerEvent struct {
	Type    string `json:"type"`
	Text    string `json:"text,omitempty"`
	Message string `json:"message,omitempty"`
}

func resolveRealtimeSourceDir(modelDir string) string {
	if sourceDir := strings.TrimSpace(os.Getenv("FIRERED_SOURCE_DIR")); sourceDir != "" {
		return sourceDir
	}
	if modelDir == "" {
		return ""
	}
	return filepath.Dir(filepath.Dir(modelDir))
}

func resolveRealtimeModelDirs(modelDir string) (vadModelDir, puncModelDir string) {
	baseDir := filepath.Dir(modelDir)
	return filepath.Join(baseDir, "FireRedVAD", "Stream-VAD"), filepath.Join(baseDir, "FireRedPunc")
}

func buildRealtimeWorkerArgs(envPath, scriptPath, modelDir, sourceDir, vadModelDir, puncModelDir string) []string {
	args := []string{
		"run", "--native-tls", "--project", envPath, "python", scriptPath,
		"--model-dir", modelDir,
		"--vad-model-dir", vadModelDir,
	}

	if sourceDir != "" {
		args = append(args, "--source-dir", sourceDir)
	}

	if puncModelDir != "" {
		if _, err := os.Stat(puncModelDir); err == nil {
			args = append(args, "--punc-model-dir", puncModelDir)
		}
	}

	return args
}

func writeLengthPrefixedChunk(w io.Writer, payload []byte) error {
	if len(payload) > int(^uint32(0)) {
		return fmt.Errorf("payload too large: %d", len(payload))
	}

	var header [4]byte
	binary.LittleEndian.PutUint32(header[:], uint32(len(payload)))
	if _, err := w.Write(header[:]); err != nil {
		return err
	}
	if len(payload) == 0 {
		return nil
	}
	_, err := w.Write(payload)
	return err
}

func (h *Handler) RealtimeTranscriptionWS(c *gin.Context) {
	conn, err := realtimeWSUpgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		logger.Warn("Failed to upgrade realtime websocket", "error", err)
		return
	}
	defer conn.Close()

	sendEvent := func(ev realtimeWorkerEvent) error {
		payload, marshalErr := json.Marshal(ev)
		if marshalErr != nil {
			return marshalErr
		}
		return conn.WriteMessage(websocket.TextMessage, payload)
	}

	ctx, cancel := context.WithCancel(c.Request.Context())
	defer cancel()

	envPath := filepath.Join(h.config.WhisperXEnv, "firered")
	modelDir := strings.TrimSpace(os.Getenv("FIRERED_MODEL_DIR"))
	if modelDir == "" {
		modelDir = "/app/models/FireRedASR2-AED"
	}
	sourceDir := resolveRealtimeSourceDir(modelDir)
	vadModelDir, puncModelDir := resolveRealtimeModelDirs(modelDir)

	adapter := adapters.NewFireRedAdapter(envPath, modelDir)
	if err := adapter.PrepareEnvironment(ctx); err != nil {
		_ = sendEvent(realtimeWorkerEvent{Type: "error", Message: fmt.Sprintf("Failed to prepare FireRed runtime: %v", err)})
		return
	}

	scriptPath := filepath.Join(envPath, "firered_realtime_worker.py")
	if _, err := os.Stat(scriptPath); err != nil {
		_ = sendEvent(realtimeWorkerEvent{Type: "error", Message: "Realtime worker script is missing"})
		return
	}

	args := buildRealtimeWorkerArgs(envPath, scriptPath, modelDir, sourceDir, vadModelDir, puncModelDir)
	cmd := exec.CommandContext(ctx, "uv", args...)
	cmd.Env = append(os.Environ(), "PYTHONUNBUFFERED=1")

	stdin, err := cmd.StdinPipe()
	if err != nil {
		_ = sendEvent(realtimeWorkerEvent{Type: "error", Message: fmt.Sprintf("Failed to start worker stdin: %v", err)})
		return
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = sendEvent(realtimeWorkerEvent{Type: "error", Message: fmt.Sprintf("Failed to start worker stdout: %v", err)})
		_ = stdin.Close()
		return
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		_ = sendEvent(realtimeWorkerEvent{Type: "error", Message: fmt.Sprintf("Failed to start worker stderr: %v", err)})
		_ = stdin.Close()
		return
	}

	if err := cmd.Start(); err != nil {
		_ = sendEvent(realtimeWorkerEvent{Type: "error", Message: fmt.Sprintf("Failed to launch worker: %v", err)})
		_ = stdin.Close()
		return
	}
	logger.Info("Realtime transcription worker started", "pid", cmd.Process.Pid)

	var writeMu sync.Mutex
	workerDone := make(chan error, 1)
	cmdDone := make(chan error, 1)
	wsDone := make(chan error, 1)

	go func() {
		scanner := bufio.NewScanner(stderr)
		scanner.Buffer(make([]byte, 0, 4096), 1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line != "" {
				logger.Debug("Realtime worker stderr", "line", line)
			}
		}
	}()

	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 0, 4096), 1024*1024)
		for scanner.Scan() {
			line := scanner.Bytes()
			var event realtimeWorkerEvent
			if err := json.Unmarshal(line, &event); err != nil {
				logger.Warn("Invalid realtime worker output", "line", string(line), "error", err)
				continue
			}
			if event.Type == "" {
				continue
			}

			writeMu.Lock()
			err := conn.WriteMessage(websocket.TextMessage, line)
			writeMu.Unlock()
			if err != nil {
				workerDone <- err
				return
			}
		}

		if err := scanner.Err(); err != nil {
			workerDone <- err
			return
		}
		workerDone <- io.EOF
	}()

	go func() {
		cmdDone <- cmd.Wait()
	}()

	go func() {
		for {
			messageType, payload, err := conn.ReadMessage()
			if err != nil {
				wsDone <- err
				return
			}

			if messageType != websocket.BinaryMessage || len(payload) == 0 {
				continue
			}

			if err := writeLengthPrefixedChunk(stdin, payload); err != nil {
				wsDone <- err
				return
			}
		}
	}()

	_ = sendEvent(realtimeWorkerEvent{Type: "system", Message: "Connecting realtime worker..."})

	for {
		select {
		case err := <-workerDone:
			if err != nil && !errors.Is(err, io.EOF) {
				logger.Warn("Realtime worker stream ended with error", "error", err)
			}
			goto shutdown
		case err := <-cmdDone:
			if err != nil {
				_ = sendEvent(realtimeWorkerEvent{Type: "error", Message: fmt.Sprintf("Realtime worker exited: %v", err)})
				logger.Warn("Realtime worker exited", "error", err)
			}
			goto shutdown
		case err := <-wsDone:
			if err != nil && !websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway, websocket.CloseNoStatusReceived) && !errors.Is(err, net.ErrClosed) {
				logger.Debug("Realtime websocket closed", "error", err)
			}
			goto shutdown
		}
	}

shutdown:
	_ = writeLengthPrefixedChunk(stdin, nil)
	_ = stdin.Close()
	cancel()

	select {
	case <-cmdDone:
	case <-time.After(3 * time.Second):
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}
}
