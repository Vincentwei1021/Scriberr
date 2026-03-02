// --- DOM refs ---
const recordingDot = document.getElementById('recordingDot');
const timerEl = document.getElementById('timer');
const statusText = document.getElementById('statusText');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const closeBtn = document.getElementById('closeBtn');
const transcriptEl = document.getElementById('transcript');
const asrStatusEl = document.getElementById('asrStatus');

// --- State ---
let isRecording = false;
let recordingStartTime = 0;
let timerInterval = null;
let mediaRecorder = null;
let recordingChunks = [];
let recordedBlob = null;
let lastRecordingDurationMs = 0;

// Audio capture
let displayStream = null;
let systemStream = null;
let micStream = null;
let mixAudioContext = null;
let mixSystemSource = null;
let mixMicSource = null;
let mixSystemGain = null;
let mixMicGain = null;
let mixDestination = null;
let mixCompressor = null;
let asrAudioContext = null;
let scriptProcessor = null;
let sourceNode = null;
let sinkNode = null;

const SYSTEM_MIX_GAIN = 0.65;
const MIC_MIX_GAIN = 1.45;
const MAX_GAIN = 2.0;

// WebSocket
let wsSocket = null;
let wsReady = false;

// --- Helpers ---
function formatTime(ms) {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60).toString().padStart(2, '0');
  const sec = (totalSec % 60).toString().padStart(2, '0');
  return `${min}:${sec}`;
}

function downsampleTo16k(input, sourceSampleRate) {
  if (sourceSampleRate === 16000) return input;
  const ratio = sourceSampleRate / 16000;
  const outputLength = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(outputLength);
  let offset = 0;
  for (let i = 0; i < outputLength; i++) {
    const nextOffset = Math.min(input.length, Math.round((i + 1) * ratio));
    let total = 0, count = 0;
    for (let j = offset; j < nextOffset; j++) { total += input[j]; count++; }
    output[i] = count > 0 ? total / count : 0;
    offset = nextOffset;
  }
  return output;
}

function clampGain(value) {
  return Math.max(0, Math.min(MAX_GAIN, value));
}

function setASRStatus(text, cls) {
  asrStatusEl.textContent = text;
  asrStatusEl.className = 'asr-status' + (cls ? ' ' + cls : '');
}

function appendTranscript(text) {
  if (transcriptEl.querySelector('.transcript-placeholder')) {
    transcriptEl.innerHTML = '';
  }
  const line = document.createElement('div');
  line.textContent = text;
  transcriptEl.appendChild(line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function formatRealtimeTimestamp() {
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: 'Asia/Shanghai',
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  }).format(new Date());
}

// --- Cleanup ---
function cleanupAudio() {
  if (scriptProcessor) { scriptProcessor.onaudioprocess = null; scriptProcessor.disconnect(); scriptProcessor = null; }
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (sinkNode) { sinkNode.disconnect(); sinkNode = null; }
  if (asrAudioContext) { asrAudioContext.close(); asrAudioContext = null; }
  if (mixSystemSource) { mixSystemSource.disconnect(); mixSystemSource = null; }
  if (mixMicSource) { mixMicSource.disconnect(); mixMicSource = null; }
  if (mixSystemGain) { mixSystemGain.disconnect(); mixSystemGain = null; }
  if (mixMicGain) { mixMicGain.disconnect(); mixMicGain = null; }
  if (mixCompressor) { mixCompressor.disconnect(); mixCompressor = null; }
  if (mixAudioContext) { mixAudioContext.close(); mixAudioContext = null; }
  mixDestination = null;
  if (systemStream) { systemStream.getTracks().forEach(t => t.stop()); systemStream = null; }
  if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (displayStream) { displayStream.getTracks().forEach(t => t.stop()); displayStream = null; }
}

function cleanupWebSocket() {
  wsReady = false;
  if (wsSocket && wsSocket.readyState === WebSocket.OPEN) {
    wsSocket.close(1000, 'recording_stopped');
  }
  wsSocket = null;
}

// --- Recording ---
async function startRecording() {
  try {
    // 1. Get system audio via getDisplayMedia (loopback handled by main process)
    displayStream = await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    });

    // Stop video track immediately
    const videoTrack = displayStream.getVideoTracks()[0];
    if (videoTrack) { videoTrack.stop(); displayStream.removeTrack(videoTrack); }

    const audioTracks = displayStream.getAudioTracks();
    if (audioTracks.length === 0) {
      setASRStatus('No audio track captured', 'error');
      cleanupAudio();
      return;
    }

    systemStream = new MediaStream(audioTracks);
    let streamForRecording = systemStream;

    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });

      mixAudioContext = new AudioContext();
      mixSystemSource = mixAudioContext.createMediaStreamSource(systemStream);
      mixMicSource = mixAudioContext.createMediaStreamSource(micStream);
      mixSystemGain = mixAudioContext.createGain();
      mixMicGain = mixAudioContext.createGain();
      mixSystemGain.gain.value = clampGain(SYSTEM_MIX_GAIN);
      mixMicGain.gain.value = clampGain(MIC_MIX_GAIN);
      mixDestination = mixAudioContext.createMediaStreamDestination();
      mixCompressor = mixAudioContext.createDynamicsCompressor();
      mixCompressor.threshold.value = -18;
      mixCompressor.knee.value = 20;
      mixCompressor.ratio.value = 3.5;
      mixCompressor.attack.value = 0.003;
      mixCompressor.release.value = 0.25;

      mixSystemSource.connect(mixSystemGain);
      mixMicSource.connect(mixMicGain);
      mixSystemGain.connect(mixCompressor);
      mixMicGain.connect(mixCompressor);
      mixCompressor.connect(mixDestination);

      streamForRecording = mixDestination.stream;
      setASRStatus('Recording system + microphone', 'ready');
    } catch (micError) {
      console.warn('Microphone unavailable in floating recorder:', micError);
      setASRStatus('Microphone unavailable, recording system audio only', 'error');
    }

    // 2. Start MediaRecorder for full recording
    mediaRecorder = new MediaRecorder(streamForRecording);
    recordingChunks = [];
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordingChunks.push(e.data); };
    mediaRecorder.onstop = async () => {
      recordedBlob = new Blob(recordingChunks, { type: recordingChunks[0]?.type || 'audio/webm' });
      const elapsed = recordingStartTime > 0 ? (Date.now() - recordingStartTime) : 0;
      lastRecordingDurationMs = Math.max(lastRecordingDurationMs, elapsed);
      // Auto-send to main window for SystemAudioRecorder upload flow
      await sendBlobToMainWindow();
    };
    mediaRecorder.start(1000);

    // 3. Start realtime ASR WebSocket
    const serverUrl = await window.floatingBridge.getServerUrl();
    if (serverUrl) {
      startRealtimeASR(streamForRecording, serverUrl);
    } else {
      setASRStatus('Server not connected - recording without ASR', 'error');
    }

    // 4. Update UI state
    isRecording = true;
    recordedBlob = null;
    recordingStartTime = Date.now();
    lastRecordingDurationMs = 0;
    recordingDot.classList.remove('idle');
    statusText.textContent = 'Recording System Audio';
    startBtn.style.display = 'none';
    stopBtn.style.display = '';

    timerInterval = setInterval(() => {
      timerEl.textContent = formatTime(Date.now() - recordingStartTime);
    }, 500);

    // Handle stream end (user stops sharing)
    audioTracks[0].addEventListener('ended', () => { if (isRecording) stopRecording(); });

  } catch (error) {
    console.error('Failed to start recording:', error);
    setASRStatus('Failed to start: ' + error.message, 'error');
    cleanupAudio();
  }
}

function stopRecording() {
  const elapsed = recordingStartTime > 0 ? (Date.now() - recordingStartTime) : 0;
  lastRecordingDurationMs = Math.max(lastRecordingDurationMs, elapsed);
  recordingStartTime = 0;
  isRecording = false;
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') { mediaRecorder.stop(); }
  // MediaRecorder.onstop will call sendBlobToMainWindow() automatically

  cleanupWebSocket();
  cleanupAudio();

  recordingDot.classList.add('idle');
  statusText.textContent = 'Sending to main window...';
  stopBtn.style.display = 'none';
  setASRStatus('ASR stopped', '');
}

async function sendBlobToMainWindow() {
  if (!recordedBlob) return;

  statusText.textContent = 'Transferring...';
  try {
    const arrayBuffer = await recordedBlob.arrayBuffer();
    const title = `System Recording ${new Date().toISOString().slice(0, 19).replace('T', ' ')}`;
    await window.floatingBridge.recordingComplete(arrayBuffer, title, lastRecordingDurationMs);
    // Window will be closed by main process after data is forwarded
  } catch (error) {
    console.error('Transfer failed:', error);
    setASRStatus('Transfer failed: ' + error.message, 'error');
    statusText.textContent = 'Transfer failed';
  }
}

// --- Realtime ASR ---
function startRealtimeASR(audioStream, serverUrl) {
  setASRStatus('Connecting to ASR...', '');

  const wsUrl = serverUrl.replace(/^http/, 'ws') + '/api/v1/transcription/realtime/ws';
  wsSocket = new WebSocket(wsUrl);

  wsSocket.onopen = () => {
    setASRStatus('Waiting for ASR worker...', '');
  };

  wsSocket.onmessage = (event) => {
    if (typeof event.data !== 'string') return;
    let payload;
    try { payload = JSON.parse(event.data); } catch { return; }

    if (payload.type === 'ready') {
      wsReady = true;
      setASRStatus('Realtime ASR active (FireRedVAD)', 'ready');
      startAudioCapture(audioStream);
    } else if (payload.type === 'text' && payload.text) {
      appendTranscript(`[${formatRealtimeTimestamp()}] ${payload.text}`);
    } else if (payload.type === 'error') {
      setASRStatus('ASR error: ' + (payload.message || 'unknown'), 'error');
    }
  };

  wsSocket.onerror = () => {
    setASRStatus('WebSocket connection failed', 'error');
  };

  wsSocket.onclose = () => {
    wsReady = false;
    if (isRecording) {
      setASRStatus('ASR connection lost - still recording audio', 'error');
    }
  };
}

function startAudioCapture(audioStream) {
  asrAudioContext = new AudioContext();
  sourceNode = asrAudioContext.createMediaStreamSource(audioStream);
  scriptProcessor = asrAudioContext.createScriptProcessor(4096, 1, 1);
  sinkNode = asrAudioContext.createGain();
  sinkNode.gain.value = 0;

  scriptProcessor.onaudioprocess = (event) => {
    if (!wsReady || !wsSocket || wsSocket.readyState !== WebSocket.OPEN) return;
    if (!asrAudioContext) return;
    const channelData = event.inputBuffer.getChannelData(0);
    if (!channelData || channelData.length === 0) return;
    const copy = new Float32Array(channelData);
    const pcm16k = downsampleTo16k(copy, asrAudioContext.sampleRate);
    if (pcm16k.length > 0) {
      wsSocket.send(pcm16k.buffer);
    }
  };

  sourceNode.connect(scriptProcessor);
  scriptProcessor.connect(sinkNode);
  sinkNode.connect(asrAudioContext.destination);
}

// --- Event listeners ---
startBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
closeBtn.addEventListener('click', () => {
  if (isRecording) stopRecording();
  cleanupWebSocket();
  cleanupAudio();
  window.floatingBridge.close();
});

// Auto-start recording when window opens
startRecording();
