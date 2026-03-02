import { useState, useEffect, useRef } from "react";
import {
	MonitorSpeaker,
	Mic,
	Square,
	Upload,
	Loader2,
	ChevronDown,
	Settings,
	XCircle,
	AlertCircle,
	CheckCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { useToast } from "@/components/ui/toast";
import { useGlobalUpload } from "@/contexts/GlobalUploadContext";

const SYSTEM_MIX_MULTIPLIER = 0.65;
const MIC_MIX_MULTIPLIER = 1.45;

const clampGain = (value: number) => Math.max(0, Math.min(2, value));

interface SystemAudioRecorderProps {
	isOpen: boolean;
	onClose: () => void;
	onRecordingComplete: (blob: Blob, title: string, source?: string) => void;
	initialBlob?: Blob | null;
	initialDurationMs?: number | null;
}

export function SystemAudioRecorder({
	isOpen,
	onClose,
	onRecordingComplete,
	initialBlob,
	initialDurationMs,
}: SystemAudioRecorderProps) {
	const {
		sendToOpenClawAfterTranscription,
		setSendToOpenClawAfterTranscription,
		selectedOpenClawProfileId,
		setSelectedOpenClawProfileId,
		openClawProfiles,
		openClawProfilesLoading,
		openClawProfilesError,
		refreshOpenClawProfiles,
	} = useGlobalUpload();

	// Recording state
	const [isRecording, setIsRecording] = useState(false);
	const [recordingTime, setRecordingTime] = useState(0);
	const [title, setTitle] = useState("");
	const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
	const [isUploading, setIsUploading] = useState(false);
	const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
	const recordingChunksRef = useRef<Blob[]>([]);
	const timerIntervalRef = useRef<number | null>(null);

	// Audio Streams & Web Audio API
	const [systemStream, setSystemStream] = useState<MediaStream | null>(null);
	const [micStream, setMicStream] = useState<MediaStream | null>(null);
	const [audioContext, setAudioContext] = useState<AudioContext | null>(null);
	const [systemGainNode, setSystemGainNode] = useState<GainNode | null>(null);
	const [micGainNode, setMicGainNode] = useState<GainNode | null>(null);

	// Volume Controls
	const [systemVolume, setSystemVolume] = useState(100);
	const [micVolume, setMicVolume] = useState(100);

	// Device Selection
	const [availableDevices, setAvailableDevices] = useState<MediaDeviceInfo[]>([]);
	const [selectedDevice, setSelectedDevice] = useState("");

	// Audio Settings
	const [autoGainControl, setAutoGainControl] = useState(true);

	// Error & Compatibility
	const [compatibilityError, setCompatibilityError] = useState<string | null>(null);
	const [permissionDenied, setPermissionDenied] = useState(false);
	const [micAvailable, setMicAvailable] = useState(true);

	// Realtime ASR state
	const [realtimeStatus, setRealtimeStatus] = useState<"idle" | "connecting" | "ready" | "error">("idle");
	const [realtimeError, setRealtimeError] = useState("");
	const [liveTranscript, setLiveTranscript] = useState("");
	const realtimeSocketRef = useRef<WebSocket | null>(null);
	const realtimeAudioContextRef = useRef<AudioContext | null>(null);
	const realtimeProcessorRef = useRef<ScriptProcessorNode | null>(null);
	const realtimeSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
	const realtimeSinkRef = useRef<GainNode | null>(null);
	const realtimeReadyRef = useRef(false);
	const recordingStartedAtRef = useRef<number | null>(null);

	const { toast } = useToast();

	// Realtime ASR helpers
	const formatBeijingTimestamp = () =>
		new Intl.DateTimeFormat("zh-CN", {
			timeZone: "Asia/Shanghai",
			hour12: false,
			hour: "2-digit",
			minute: "2-digit",
			second: "2-digit",
		}).format(new Date());

	const getBlobDurationMs = async (blob: Blob): Promise<number> => {
		const objectUrl = URL.createObjectURL(blob);
		try {
			const durationSeconds = await new Promise<number>((resolve, reject) => {
				const audio = document.createElement("audio");
				audio.preload = "metadata";
				audio.src = objectUrl;
				audio.onloadedmetadata = () => resolve(audio.duration);
				audio.onerror = () => reject(new Error("Failed to read audio metadata"));
			});
			if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
				return 0;
			}
			return Math.round(durationSeconds * 1000);
		} catch {
			return 0;
		} finally {
			URL.revokeObjectURL(objectUrl);
		}
	};

	const downsampleTo16k = (input: Float32Array, sourceSampleRate: number): Float32Array => {
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
	};

	const cleanupRealtimeASR = () => {
		realtimeReadyRef.current = false;
		if (realtimeProcessorRef.current) { realtimeProcessorRef.current.onaudioprocess = null; realtimeProcessorRef.current.disconnect(); realtimeProcessorRef.current = null; }
		if (realtimeSourceRef.current) { realtimeSourceRef.current.disconnect(); realtimeSourceRef.current = null; }
		if (realtimeSinkRef.current) { realtimeSinkRef.current.disconnect(); realtimeSinkRef.current = null; }
		if (realtimeAudioContextRef.current) { void realtimeAudioContextRef.current.close(); realtimeAudioContextRef.current = null; }
		const socket = realtimeSocketRef.current;
		realtimeSocketRef.current = null;
		if (socket && socket.readyState === WebSocket.OPEN) { socket.close(1000, "recording_stopped"); }
		setRealtimeStatus("idle");
	};

	const startRealtimeASR = (audioStream: MediaStream) => {
		setRealtimeError("");
		setLiveTranscript("");
		setRealtimeStatus("connecting");

		const protocol = window.location.protocol === "https:" ? "wss" : "ws";
		const url = `${protocol}://${window.location.host}/api/v1/transcription/realtime/ws`;
		const socket = new WebSocket(url);
		realtimeSocketRef.current = socket;

		socket.onmessage = (event) => {
			if (typeof event.data !== "string") return;
			let payload: { type: string; text?: string; message?: string };
			try { payload = JSON.parse(event.data); } catch { return; }
			if (!payload?.type) return;

			if (payload.type === "ready") {
				setRealtimeStatus("ready");
				realtimeReadyRef.current = true;
				// Start audio capture pipeline
				const ctx = new AudioContext();
				realtimeAudioContextRef.current = ctx;
				const source = ctx.createMediaStreamSource(audioStream);
				realtimeSourceRef.current = source;
				const processor = ctx.createScriptProcessor(4096, 1, 1);
				realtimeProcessorRef.current = processor;
				const sink = ctx.createGain();
				sink.gain.value = 0;
				realtimeSinkRef.current = sink;

				processor.onaudioprocess = (e) => {
					if (!realtimeReadyRef.current) return;
					const ws = realtimeSocketRef.current;
					if (!ws || ws.readyState !== WebSocket.OPEN) return;
					const data = e.inputBuffer.getChannelData(0);
					if (!data || data.length === 0) return;
					const pcm16k = downsampleTo16k(new Float32Array(data), ctx.sampleRate);
					if (pcm16k.length > 0) ws.send(pcm16k.buffer);
				};

				source.connect(processor);
				processor.connect(sink);
				sink.connect(ctx.destination);
				} else if (payload.type === "text" && payload.text) {
					const line = `[${formatBeijingTimestamp()}] ${payload.text}`;
					setLiveTranscript((prev) => (prev ? `${prev}\n${line}` : line));
				} else if (payload.type === "error") {
					setRealtimeStatus("error");
					setRealtimeError(payload.message || "Realtime transcription failed.");
				}
			};

		socket.onerror = () => { setRealtimeStatus("error"); setRealtimeError("Failed to connect realtime ASR."); };
		socket.onclose = () => { realtimeReadyRef.current = false; if (realtimeSocketRef.current) setRealtimeStatus("error"); };
	};

	// Browser compatibility check - only Chromium browsers supported
	const checkCompatibility = (): { supported: boolean; error?: string } => {
		// Check if browser supports getDisplayMedia at all
		if (!navigator.mediaDevices?.getDisplayMedia) {
			return {
				supported: false,
				error:
					"Your browser doesn't support screen capture. Please use Chrome, Edge, or Brave.",
			};
		}

		// Check if it's a Chromium-based browser
		const userAgent = navigator.userAgent.toLowerCase();
		const isChromium = userAgent.includes('chrome') ||
		                   userAgent.includes('chromium') ||
		                   userAgent.includes('edg/') ||
		                   userAgent.includes('brave');

		if (!isChromium) {
			return {
				supported: false,
				error:
					"System audio recording is only supported on Chromium-based browsers (Chrome, Edge, Brave). " +
					"Please switch to one of these browsers to use this feature.",
			};
		}

		return { supported: true };
	};

	// Load initial blob from floating window recording
	useEffect(() => {
		if (isOpen && initialBlob) {
			setRecordedBlob(initialBlob);
			setIsRecording(false);
			const fallbackDuration = typeof initialDurationMs === "number" && Number.isFinite(initialDurationMs)
				? Math.max(0, Math.round(initialDurationMs))
				: 0;
			if (fallbackDuration > 0) {
				setRecordingTime(fallbackDuration);
			}
			let cancelled = false;
			void (async () => {
				const durationMs = await getBlobDurationMs(initialBlob);
				if (!cancelled) {
					setRecordingTime((prev) => Math.max(prev, durationMs, fallbackDuration));
				}
			})();
			return () => {
				cancelled = true;
			};
		}
	}, [isOpen, initialBlob, initialDurationMs]);

	useEffect(() => {
		if (isOpen) {
			void refreshOpenClawProfiles();
		}
	}, [isOpen, refreshOpenClawProfiles]);

	// Initialize microphone device list when dialog opens
	useEffect(() => {
		if (!isOpen) return;

		let activeStream: MediaStream | null = null;

		const init = async () => {
			try {
				// Check browser compatibility
				const compatibility = checkCompatibility();
				if (!compatibility.supported) {
					setCompatibilityError(compatibility.error || null);
					return;
				}

				// Request permission to get device labels
				activeStream = await navigator.mediaDevices.getUserMedia({
					audio: true,
				});

				// Get available microphones
				const devices = await navigator.mediaDevices.enumerateDevices();
				const audioDevices = devices.filter((d) => d.kind === "audioinput");
				setAvailableDevices(audioDevices);

				// Set default device if none selected
				if (audioDevices.length > 0) {
					const deviceExists = audioDevices.some(
						(d) => d.deviceId === selectedDevice,
					);
					if (!selectedDevice || !deviceExists) {
						setSelectedDevice(audioDevices[0].deviceId);
					}
				}
			} catch (error) {
				console.error("Failed to enumerate devices:", error);
				toast({
					title: "Initialization Error",
					description: "Failed to get microphone devices. You can still record system audio only.",
				});
			} finally {
				// Stop the temporary stream used for permissions
				if (activeStream) {
					activeStream.getTracks().forEach((track) => track.stop());
				}
			}
		};

		init();
	}, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

	// Simple timer that increments every second while recording
	useEffect(() => {
		if (isRecording) {
			timerIntervalRef.current = window.setInterval(() => {
				setRecordingTime((prev) => prev + 1000);
			}, 1000);
		} else {
			if (timerIntervalRef.current) {
				clearInterval(timerIntervalRef.current);
				timerIntervalRef.current = null;
			}
		}

		return () => {
			if (timerIntervalRef.current) {
				clearInterval(timerIntervalRef.current);
			}
		};
	}, [isRecording]);

	// Handle background recording - prevent page unload warnings during recording
	useEffect(() => {
		const originalTitle = document.title;

		const handleBeforeUnload = (e: BeforeUnloadEvent) => {
			if (isRecording) {
				e.preventDefault();
				e.returnValue =
					"Recording in progress. Are you sure you want to leave?";
				return e.returnValue;
			}
		};

		if (isRecording) {
			// Update page title to show recording status
			document.title = `🔴 Recording System Audio... - ${originalTitle}`;
			window.addEventListener("beforeunload", handleBeforeUnload);
		} else {
			// Restore original title
			document.title = originalTitle;
		}

		return () => {
			document.title = originalTitle;
			window.removeEventListener("beforeunload", handleBeforeUnload);
		};
	}, [isRecording]);

	// Create mixed audio stream using Web Audio API
	const createMixedAudioStream = (
		sysStream: MediaStream,
		mStream: MediaStream,
	): MediaStream => {
		try {
			// Create audio context
			const ctx = new AudioContext();
			setAudioContext(ctx);

			// Create source nodes from MediaStreams
			const systemSource = ctx.createMediaStreamSource(sysStream);
			const micSource = ctx.createMediaStreamSource(mStream);

			// Create gain nodes for volume control
			const systemGain = ctx.createGain();
			const micGain = ctx.createGain();

			// Set initial volumes
			systemGain.gain.value = clampGain((systemVolume / 100) * SYSTEM_MIX_MULTIPLIER);
			micGain.gain.value = clampGain((micVolume / 100) * MIC_MIX_MULTIPLIER);

			// Store gain nodes for real-time control
			setSystemGainNode(systemGain);
			setMicGainNode(micGain);

			// Create destination for mixed output
			const destination = ctx.createMediaStreamDestination();
			const masterCompressor = ctx.createDynamicsCompressor();
			masterCompressor.threshold.value = -16;
			masterCompressor.knee.value = 20;
			masterCompressor.ratio.value = 3;
			masterCompressor.attack.value = 0.003;
			masterCompressor.release.value = 0.25;

			// Connect: sources → gains → destination
			systemSource.connect(systemGain);
			micSource.connect(micGain);
			systemGain.connect(masterCompressor);
			micGain.connect(masterCompressor);
			masterCompressor.connect(destination);

			return destination.stream;
		} catch (error) {
			console.error("Audio mixing failed:", error);
			toast({
				title: "Audio Mixing Unavailable",
				description: "Recording system audio only. Browser doesn't support mixing.",
			});
			// Fallback: return system stream only
			return sysStream;
		}
	};

	// Start recording
	const startRecording = async () => {
		try {
			setPermissionDenied(false);

			// Step 1: Request system audio via getDisplayMedia
			// Note: video is required by the API, we'll stop it immediately
			const displayStream = await navigator.mediaDevices.getDisplayMedia({
				video: true,
				audio: {
					echoCancellation: false,
					noiseSuppression: false,
					autoGainControl: false,
				},
			});

			// Debug: Log what tracks we got
			console.info("Display stream tracks:", {
				video: displayStream.getVideoTracks().length,
				audio: displayStream.getAudioTracks().length,
				allTracks: displayStream.getTracks().map(t => ({ kind: t.kind, label: t.label }))
			});

			// Stop the video track immediately since we only want audio
			const videoTrack = displayStream.getVideoTracks()[0];
			if (videoTrack) {
				videoTrack.stop();
				displayStream.removeTrack(videoTrack);
			}

			// Create a new MediaStream with only audio tracks
			const audioTracks = displayStream.getAudioTracks();
			if (audioTracks.length === 0) {
				alert(
					"No audio track found!\\n\\n" +
					"Make sure to:\\n" +
					"1. Select a Chrome TAB (not window or screen)\\n" +
					"2. Check the 'Share tab audio' checkbox\\n" +
					"3. Choose a tab that's actually playing audio"
				);
				cleanupStreams();
				return;
			}
			const sysStream = new MediaStream(audioTracks);

			setSystemStream(sysStream);

			// Handle stream end (user stops sharing via browser UI)
			sysStream.getAudioTracks()[0].addEventListener("ended", () => {
				if (isRecording) {
					stopRecording();
					toast({
						title: "Screen Sharing Stopped",
						description: "Recording has been saved.",
					});
				}
			});

			// Step 2: Request microphone
			let mStream: MediaStream | null = null;
			try {
				const preferredDeviceConstraint = selectedDevice ? { exact: selectedDevice } : undefined;
				const micConstraintCandidates: Array<{
					label: string;
					constraints: MediaTrackConstraints;
				}> = [
					{
						label: "selected-device-clean",
						constraints: {
							deviceId: preferredDeviceConstraint,
							echoCancellation: false,
							noiseSuppression: false,
							autoGainControl: autoGainControl,
						},
					},
					{
						label: "selected-device-processed",
						constraints: {
							deviceId: preferredDeviceConstraint,
							echoCancellation: true,
							noiseSuppression: true,
							autoGainControl: autoGainControl,
						},
					},
					{
						label: "default-device-clean",
						constraints: {
							echoCancellation: false,
							noiseSuppression: false,
							autoGainControl: autoGainControl,
						},
					},
				];

				let lastMicError: unknown = null;
				for (const candidate of micConstraintCandidates) {
					try {
						mStream = await navigator.mediaDevices.getUserMedia({
							audio: candidate.constraints,
						});
						console.info("Microphone capture initialized", {
							attempt: candidate.label,
							settings: mStream.getAudioTracks()[0]?.getSettings(),
						});
						break;
					} catch (candidateError) {
						lastMicError = candidateError;
						console.warn("Microphone capture attempt failed", {
							attempt: candidate.label,
							error: candidateError,
						});
					}
				}

				if (!mStream && lastMicError) {
					throw lastMicError;
				}
				if (!mStream) {
					throw new Error("Unable to initialize microphone stream");
				}

				setMicStream(mStream);
				setMicAvailable(true);
			} catch (micError) {
				console.error("Microphone permission denied:", micError);
				toast({
					title: "Microphone Unavailable",
					description: "Recording system audio only.",
				});
				setMicAvailable(false);
			}

			// Step 3: Mix streams or use system-only
			let streamToRecord: MediaStream;
			if (mStream) {
				streamToRecord = createMixedAudioStream(sysStream, mStream);
			} else {
				streamToRecord = sysStream;
			}

			// Step 4: Create MediaRecorder directly
			const recorder = new MediaRecorder(streamToRecord);
			recordingChunksRef.current = [];

			recorder.ondataavailable = (e) => {
				if (e.data.size > 0) {
					recordingChunksRef.current.push(e.data);
				}
			};

			recorder.onstop = () => {
				const blob = new Blob(recordingChunksRef.current, {
					type: recordingChunksRef.current[0]?.type || 'audio/webm'
				});
				const elapsedMs = recordingStartedAtRef.current ? Date.now() - recordingStartedAtRef.current : 0;
				setRecordingTime((prev) => Math.max(prev, elapsedMs));
				recordingStartedAtRef.current = null;
				setRecordedBlob(blob);
				setIsRecording(false);
			};

			recorder.start(1000); // Capture in 1-second chunks
			setMediaRecorder(recorder);

			setIsRecording(true);
			setRecordingTime(0);
			setRecordedBlob(null);
			recordingStartedAtRef.current = Date.now();

			// Step 5: Start realtime ASR on the system audio stream
			void startRealtimeASR(sysStream);
		} catch (error) {
			console.error("Failed to start recording:", error);

			// Handle specific errors
			if (error instanceof Error && error.name === "NotAllowedError") {
				setPermissionDenied(true);
			} else if (error instanceof Error && error.name === "NotFoundError") {
				alert(
					"The selected source doesn't support audio sharing. Please choose a tab or window with audio.",
				);
			} else {
				alert("Failed to start screen sharing. Please try again.");
			}

			// Cleanup if failed
			cleanupStreams();
		}
	};

	// Stop recording
	const stopRecording = () => {
		if (recordingStartedAtRef.current) {
			const elapsedMs = Date.now() - recordingStartedAtRef.current;
			setRecordingTime((prev) => Math.max(prev, elapsedMs));
		}
		if (mediaRecorder && mediaRecorder.state !== 'inactive') {
			mediaRecorder.stop();
		}
		cleanupRealtimeASR();
		cleanupStreams();
	};

	// Update system volume in real-time
	const updateSystemVolume = (value: number[]) => {
		const vol = value[0];
		setSystemVolume(vol);
		if (systemGainNode && isRecording) {
			systemGainNode.gain.value = clampGain((vol / 100) * SYSTEM_MIX_MULTIPLIER);
		}
	};

	// Update microphone volume in real-time
	const updateMicVolume = (value: number[]) => {
		const vol = value[0];
		setMicVolume(vol);
		if (micGainNode && isRecording) {
			micGainNode.gain.value = clampGain((vol / 100) * MIC_MIX_MULTIPLIER);
		}
	};

	// Cleanup streams and audio context
	const cleanupStreams = () => {
		if (systemStream) {
			systemStream.getTracks().forEach((track) => track.stop());
			setSystemStream(null);
		}
		if (micStream) {
			micStream.getTracks().forEach((track) => track.stop());
			setMicStream(null);
		}
		if (audioContext && audioContext.state !== "closed") {
			audioContext.close();
			setAudioContext(null);
		}
		setSystemGainNode(null);
		setMicGainNode(null);
		setMediaRecorder(null);
	};

	// Format time in mm:ss
	const formatTime = (timeMs: number) => {
		const minutes = Math.floor(timeMs / 60000);
		const seconds = Math.floor((timeMs % 60000) / 1000);
		return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
	};

	// Handle upload
	const handleUpload = async () => {
		if (!recordedBlob) return;

		setIsUploading(true);
		try {
			await onRecordingComplete(
				recordedBlob,
				title || `System Recording ${new Date().toISOString()}`,
				"system_audio_recording",
			);
			// Reset state
			setRecordedBlob(null);
			setTitle("");
			setRecordingTime(0);
			setSendToOpenClawAfterTranscription(false);
			onClose();
		} catch (error) {
			console.error("Failed to upload recording:", error);
			alert("Failed to upload recording");
		} finally {
			setIsUploading(false);
		}
	};

	// Handle dialog close
	const handleClose = () => {
		if (isRecording) {
			stopRecording();
		}
		cleanupRealtimeASR();
		cleanupStreams();
		setRecordedBlob(null);
		setTitle("");
		setRecordingTime(0);
		setIsRecording(false);
		setPermissionDenied(false);
		setCompatibilityError(null);
		setLiveTranscript("");
		setRealtimeStatus("idle");
		setRealtimeError("");
		recordingStartedAtRef.current = null;
		setSendToOpenClawAfterTranscription(false);
		onClose();
	};

	// Don't render anything if there's a compatibility error - show it in a separate dialog
	if (compatibilityError) {
		return (
			<Dialog open={true} onOpenChange={(open) => {
				if (!open) {
					setCompatibilityError(null);
					onClose();
				}
			}}>
				<DialogContent className="sm:max-w-[600px]">
					<DialogHeader>
						<DialogTitle className="flex items-center gap-2">
							<MonitorSpeaker className="h-5 w-5 text-[var(--brand-solid)]" />
							Record System Audio
						</DialogTitle>
					</DialogHeader>

					<div className="flex items-center gap-3 p-4 bg-[var(--error)]/10 border border-[var(--error)]/20 rounded-[var(--radius-card)]">
						<XCircle className="h-6 w-6 text-[var(--error)] flex-shrink-0" />
						<div>
							<h3 className="font-semibold mb-2 text-[var(--text-primary)]">
								Browser Not Supported
							</h3>
							<p className="text-sm mb-3 text-[var(--text-secondary)]">
								{compatibilityError}
							</p>
							<p className="text-xs text-[var(--text-tertiary)]">
								You can use "Record Audio" for microphone-only recording.
							</p>
						</div>
					</div>

					<div className="flex justify-end">
						<Button variant="outline" onClick={() => {
							setCompatibilityError(null);
							onClose();
						}}>
							Close
						</Button>
					</div>
				</DialogContent>
			</Dialog>
		);
	}

	// If dialog not open, don't render anything
	if (!isOpen) {
		return null;
	}

	// Render permission denied error
	if (permissionDenied && !isRecording) {
		return (
			<Dialog open={isOpen} onOpenChange={handleClose}>
				<DialogContent className="sm:max-w-[600px]">
					<DialogHeader>
						<DialogTitle className="flex items-center gap-2">
							<MonitorSpeaker className="h-5 w-5 text-[var(--brand-solid)]" />
							Record System Audio
						</DialogTitle>
					</DialogHeader>

					<div className="flex items-center gap-3 p-4 bg-[var(--warning-translucent)] border border-[var(--warning-solid)]/20 rounded-[var(--radius-card)]">
						<AlertCircle className="h-6 w-6 text-[var(--warning-solid)] flex-shrink-0" />
						<div>
							<h3 className="font-semibold mb-2 text-[var(--text-primary)]">
								Screen Sharing Permission Required
							</h3>
							<p className="text-sm mb-3 text-[var(--text-secondary)]">
								You denied screen sharing permission. Please click "Try Again"
								and allow access when prompted.
							</p>
							<p className="text-xs font-medium text-[var(--warning-solid)]">
								Make sure to check "Share system audio" or "Share tab audio"
								in the browser picker!
							</p>
						</div>
					</div>

					<div className="flex justify-end gap-3">
						<Button variant="outline" onClick={handleClose}>
							Cancel
						</Button>
						<Button
							onClick={() => {
								setPermissionDenied(false);
								startRecording();
							}}
						>
							Try Again
						</Button>
					</div>
				</DialogContent>
			</Dialog>
		);
	}

	// Render recording complete state
	if (recordedBlob && !isRecording) {
		return (
			<Dialog open={isOpen} onOpenChange={handleClose}>
				<DialogContent className="sm:max-w-[600px]">
					<DialogHeader>
						<DialogTitle className="flex items-center gap-2">
							<MonitorSpeaker className="h-5 w-5 text-[var(--brand-solid)]" />
							Recording Complete
						</DialogTitle>
					</DialogHeader>

					<div className="space-y-6 py-4">
						{/* Success Message */}
						<div className="flex items-center gap-3 p-4 bg-[var(--success-translucent)] border border-[var(--success-solid)]/20 rounded-[var(--radius-card)]">
							<CheckCircle className="h-5 w-5 text-[var(--success-solid)] flex-shrink-0" />
							<div>
								<h3 className="font-semibold text-[var(--text-primary)]">
									Recording Complete!
								</h3>
								<p className="text-sm text-[var(--text-secondary)]">
									Duration: {formatTime(recordingTime)}
								</p>
							</div>
						</div>

						{/* Title Input */}
						<div className="space-y-2">
							<label className="text-sm font-medium text-[var(--text-primary)]">
								Recording Title
							</label>
							<Input
								value={title}
								onChange={(e) => setTitle(e.target.value)}
								placeholder="Enter a title for your recording..."
							/>
						</div>

						<div className="space-y-3 p-4 border border-[var(--border-subtle)] rounded-[var(--radius-card)] bg-[var(--bg-card)]">
							<div className="flex items-start gap-3">
								<Checkbox
									id="system-recorder-auto-send-openclaw"
									checked={sendToOpenClawAfterTranscription}
									onCheckedChange={(checked) => setSendToOpenClawAfterTranscription(checked === true)}
									className="mt-0.5"
								/>
								<div>
									<Label htmlFor="system-recorder-auto-send-openclaw" className="text-sm font-medium text-[var(--text-primary)] cursor-pointer">
										Send to OpenClaw after transcription
									</Label>
									<p className="text-xs text-[var(--text-secondary)] mt-1">
										Upload completes first, then system will auto-send to OpenClaw after ASR finishes.
									</p>
								</div>
							</div>

							<div className="flex items-center gap-2">
								<Select
									value={selectedOpenClawProfileId}
									onValueChange={setSelectedOpenClawProfileId}
									disabled={!sendToOpenClawAfterTranscription || openClawProfilesLoading || openClawProfiles.length === 0}
								>
									<SelectTrigger className="w-full">
										<SelectValue placeholder={openClawProfilesLoading ? "Loading profiles..." : "Select OpenClaw profile"} />
									</SelectTrigger>
									<SelectContent>
										{openClawProfiles.map((profile) => (
											<SelectItem key={profile.id} value={profile.id}>
												{profile.name}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
								<Button variant="outline" size="sm" onClick={() => void refreshOpenClawProfiles()}>
									Refresh
								</Button>
							</div>

							{openClawProfilesError && (
								<p className="text-xs text-[var(--error)]">{openClawProfilesError}</p>
							)}
							{sendToOpenClawAfterTranscription && !openClawProfilesLoading && openClawProfiles.length === 0 && !openClawProfilesError && (
								<p className="text-xs text-[var(--warning-solid)]">
									No OpenClaw profiles found. Create one in Settings &gt; OpenClaw.
								</p>
							)}
						</div>

						{/* Upload Button */}
						<Button
							onClick={handleUpload}
							disabled={isUploading}
							className="w-full rounded-xl text-white cursor-pointer bg-gradient-to-r from-[#FFAB40] to-[#FF3D00] hover:opacity-90 active:scale-[0.98] transition-all shadow-lg shadow-orange-500/20"
						>
							{isUploading ? (
								<>
									<Loader2 className="h-5 w-5 mr-2 animate-spin" />
									Uploading...
								</>
							) : (
								<>
									<Upload className="h-5 w-5 mr-2" />
									Upload Recording
								</>
							)}
						</Button>
					</div>
				</DialogContent>
			</Dialog>
		);
	}

	// Render active recording state
	if (isRecording) {
		return (
			<Dialog open={isOpen} onOpenChange={handleClose}>
				<DialogContent className="sm:max-w-[700px]">
					<DialogHeader>
						<DialogTitle className="flex items-center gap-2">
							<MonitorSpeaker className="h-5 w-5 text-[var(--brand-solid)]" />
							Recording System Audio
						</DialogTitle>
					</DialogHeader>

					<div className="space-y-6 py-4">
						{/* Recording Status Banner */}
						<div className="flex items-center gap-3 p-4 bg-[var(--brand-light)] border border-[var(--brand-solid)]/20 rounded-[var(--radius-card)]">
							<div className="h-3 w-3 bg-[var(--error)] rounded-full animate-pulse flex-shrink-0" />
							<div>
								<h3 className="font-semibold text-[var(--text-primary)]">
									Recording System Audio{micAvailable ? " + Microphone" : " Only"}
								</h3>
								<p className="text-xs text-[var(--text-secondary)]">
									Recording continues even if you switch tabs
								</p>
							</div>
						</div>

						{/* Recording Time */}
						<div className="text-center">
							<div className="text-6xl font-mono font-bold text-[var(--text-primary)] mb-2">
								{formatTime(recordingTime)}
							</div>
							<div className="flex items-center justify-center gap-2 text-sm text-[var(--text-secondary)]">
								<div className="h-2 w-2 bg-[var(--error)] rounded-full animate-pulse" />
								<span>Recording...</span>
							</div>
						</div>

						{/* Volume Controls */}
						{micAvailable && (
							<div className="grid grid-cols-2 gap-4">
								<div className="space-y-2">
									<div className="flex items-center gap-2">
										<MonitorSpeaker className="h-4 w-4 text-[var(--brand-solid)]" />
										<label className="text-sm font-medium text-[var(--text-primary)]">
											System Audio
										</label>
									</div>
									<Slider
										min={0}
										max={100}
										step={1}
										value={[systemVolume]}
										onValueChange={updateSystemVolume}
										className="cursor-pointer"
									/>
									<span className="text-xs text-[var(--text-tertiary)]">
										{systemVolume}%
									</span>
								</div>
								<div className="space-y-2">
									<div className="flex items-center gap-2">
										<Mic className="h-4 w-4 text-[var(--brand-solid)]" />
										<label className="text-sm font-medium text-[var(--text-primary)]">
											Microphone
										</label>
									</div>
									<Slider
										min={0}
										max={100}
										step={1}
										value={[micVolume]}
										onValueChange={updateMicVolume}
										className="cursor-pointer"
									/>
									<span className="text-xs text-[var(--text-tertiary)]">
										{micVolume}%
									</span>
								</div>
							</div>
						)}

						{/* Live Transcript */}
						<div className="space-y-2">
							<div className="text-sm font-medium text-[var(--text-primary)]">
								Live Transcript (FireRedVAD)
							</div>
							<div className="rounded-lg border border-[var(--border-primary)] bg-[var(--bg-secondary)] p-3 min-h-[90px] max-h-[170px] overflow-y-auto text-sm text-[var(--text-primary)] whitespace-pre-wrap">
								{liveTranscript || "Realtime text will appear here after speech segments are detected."}
							</div>
							<div className="text-xs text-[var(--text-tertiary)]">
								{realtimeStatus === "connecting" ? "Initializing FireRed realtime ASR..." :
								 realtimeStatus === "ready" ? "Realtime transcription running (FireRedVAD)" :
								 realtimeStatus === "error" ? "Realtime transcription unavailable" :
								 "Realtime transcription idle"}
							</div>
							{realtimeError && <div className="text-xs text-[var(--error)]">{realtimeError}</div>}
						</div>

						{/* Recording Controls */}
						<div className="flex justify-center">
							<Button
								onClick={stopRecording}
								size="lg"
								variant="secondary"
							>
								<Square className="h-5 w-5 mr-2" />
								Stop Recording
							</Button>
						</div>
					</div>
				</DialogContent>
			</Dialog>
		);
	}

	// Render initial instructions state
	return (
		<Dialog open={isOpen} onOpenChange={handleClose}>
			<DialogContent className="sm:max-w-[700px]">
				<DialogHeader>
					<DialogTitle className="flex items-center gap-2">
						<MonitorSpeaker className="h-5 w-5 text-[var(--brand-solid)]" />
						Record System Audio
					</DialogTitle>
					<DialogDescription>
						Capture system audio from your screen/tab along with your microphone
						for meeting recordings.
					</DialogDescription>
				</DialogHeader>

				<div className="space-y-6 py-4">
					{/* Instructions Card */}
					<div className="p-4 bg-[var(--brand-light)] border border-[var(--brand-solid)]/20 rounded-[var(--radius-card)]">
						<h3 className="font-semibold mb-3 text-[var(--text-primary)]">
							How it works:
						</h3>
						<ol className="space-y-3 text-sm text-[var(--text-secondary)]">
							<li className="flex gap-3">
								<span className="font-bold text-[var(--brand-solid)] flex-shrink-0">
									1.
								</span>
								<span>Click "Start Recording" below</span>
							</li>
							<li className="flex gap-3">
								<span className="font-bold text-[var(--brand-solid)] flex-shrink-0">
									2.
								</span>
								<span>
									Select a <strong>Chrome Tab</strong> from the browser picker (not window or screen)
								</span>
							</li>
							<li className="flex gap-3">
								<span className="font-bold text-[var(--brand-solid)] flex-shrink-0">
									3.
								</span>
								<span>
									<strong>Check "Share tab audio"</strong> checkbox at the bottom
								</span>
							</li>
							<li className="flex gap-3">
								<span className="font-bold text-[var(--brand-solid)] flex-shrink-0">
									4.
								</span>
								<span>Allow microphone access when prompted (optional)</span>
							</li>
						</ol>
						<div className="mt-4 p-3 bg-[var(--warning-translucent)] border border-[var(--warning-solid)]/20 rounded-[var(--radius-btn)]">
							<p className="text-xs text-[var(--text-secondary)]">
								<strong>💡 Tip:</strong> Use headphones to prevent echo and ensure the best recording quality!
							</p>
						</div>
					</div>

					{/* Title Input */}
					<div className="space-y-2">
						<label className="text-sm font-medium text-[var(--text-primary)]">
							Recording Title (Optional)
						</label>
						<Input
							value={title}
							onChange={(e) => setTitle(e.target.value)}
							placeholder="Enter a title for your recording..."
							disabled={isRecording}
						/>
					</div>

					{/* Microphone Selection */}
					{availableDevices.length > 1 && (
						<div className="space-y-2">
							<label className="text-sm font-medium text-[var(--text-primary)]">
								Microphone
							</label>
							<DropdownMenu>
								<DropdownMenuTrigger asChild disabled={isRecording}>
									<Button
										variant="outline"
										className="w-full justify-between"
									>
										<div className="flex items-center gap-2">
											<Settings className="h-4 w-4" />
											<span className="truncate">
												{availableDevices.find(
													(d) => d.deviceId === selectedDevice,
												)?.label || `Microphone ${selectedDevice.slice(0, 8)}`}
											</span>
										</div>
										<ChevronDown className="h-4 w-4 opacity-50" />
									</Button>
								</DropdownMenuTrigger>
								<DropdownMenuContent className="w-full min-w-[400px]">
									{availableDevices.map((device) => (
										<DropdownMenuItem
											key={device.deviceId}
											onClick={() => setSelectedDevice(device.deviceId)}
											className="flex items-center gap-3 px-3 py-2 cursor-pointer"
										>
											<Mic className="h-4 w-4 text-[var(--text-tertiary)]" />
											<div className="flex-1">
												<div className="text-sm font-medium text-[var(--text-primary)]">
													{device.label ||
														`Microphone ${device.deviceId.slice(0, 8)}`}
												</div>
												<div className="text-xs text-[var(--text-tertiary)]">
													Device ID: {device.deviceId.slice(0, 20)}...
												</div>
											</div>
											{selectedDevice === device.deviceId && (
												<div className="h-2 w-2 bg-[var(--brand-solid)] rounded-full"></div>
											)}
										</DropdownMenuItem>
									))}
								</DropdownMenuContent>
							</DropdownMenu>
						</div>
					)}

					{/* Audio Settings */}
					<div className="space-y-3">
						<label className="text-sm font-medium text-[var(--text-primary)]">
							Audio Settings
						</label>
						<div className="flex items-center justify-between p-3 bg-[var(--bg-card)] border border-[var(--border-subtle)] rounded-[var(--radius-card)]">
							<div className="flex-1">
								<div className="text-sm font-medium text-[var(--text-primary)]">
									Automatic Gain Control
								</div>
								<p className="text-xs text-[var(--text-tertiary)] mt-1">
									Automatically adjusts microphone volume for consistent audio levels
								</p>
							</div>
							<Switch
								id="agc-toggle"
								checked={autoGainControl}
								onCheckedChange={setAutoGainControl}
							/>
						</div>
					</div>

					{/* Start Button */}
					<Button
						onClick={startRecording}
						size="lg"
						className="w-full rounded-xl text-white cursor-pointer bg-gradient-to-r from-[#FFAB40] to-[#FF3D00] hover:opacity-90 active:scale-[0.98] transition-all shadow-lg shadow-orange-500/20"
					>
						<MonitorSpeaker className="h-5 w-5 mr-2" />
						Start Recording
					</Button>
				</div>
			</DialogContent>
		</Dialog>
	);
}
