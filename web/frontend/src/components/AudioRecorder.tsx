import { useState, useEffect, useRef } from "react";
import WaveSurfer from "wavesurfer.js";
import RecordPlugin from "wavesurfer.js/dist/plugins/record.js";
import {
	Mic,
	Square,
	Play,
	Pause,
	Upload,
	Loader2,
	ChevronDown,
	Settings,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
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

interface AudioRecorderProps {
	isOpen: boolean;
	onClose: () => void;
	onRecordingComplete: (blob: Blob, title: string) => void;
}

type RealtimeStatus = "idle" | "connecting" | "ready" | "error";

interface RealtimeEvent {
	type: string;
	text?: string;
	message?: string;
}

function downsampleTo16k(input: Float32Array, sourceSampleRate: number): Float32Array {
	if (sourceSampleRate === 16000) {
		return input;
	}

	const ratio = sourceSampleRate / 16000;
	const outputLength = Math.max(1, Math.round(input.length / ratio));
	const output = new Float32Array(outputLength);
	let offset = 0;

	for (let i = 0; i < outputLength; i++) {
		const nextOffset = Math.min(input.length, Math.round((i + 1) * ratio));
		let total = 0;
		let count = 0;
		for (let j = offset; j < nextOffset; j++) {
			total += input[j];
			count++;
		}
		output[i] = count > 0 ? total / count : 0;
		offset = nextOffset;
	}

	return output;
}

export function AudioRecorder({
	isOpen,
	onClose,
	onRecordingComplete,
}: AudioRecorderProps) {
	const [wavesurfer, setWavesurfer] = useState<WaveSurfer | null>(null);
	const [record, setRecord] = useState<RecordPlugin | null>(null);
	const [isRecording, setIsRecording] = useState(false);
	const [isPaused, setIsPaused] = useState(false);
	const [recordingTime, setRecordingTime] = useState(0);
	const [title, setTitle] = useState("");
	const [availableDevices, setAvailableDevices] = useState<MediaDeviceInfo[]>([]);
	const [selectedDevice, setSelectedDevice] = useState("");
	const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
	const [isUploading, setIsUploading] = useState(false);
	const [realtimeStatus, setRealtimeStatus] = useState<RealtimeStatus>("idle");
	const [realtimeError, setRealtimeError] = useState<string>("");
	const [liveTranscript, setLiveTranscript] = useState("");

	const micContainerRef = useRef<HTMLDivElement>(null);
	const realtimeSocketRef = useRef<WebSocket | null>(null);
	const realtimeAudioContextRef = useRef<AudioContext | null>(null);
	const realtimeProcessorRef = useRef<ScriptProcessorNode | null>(null);
	const realtimeSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
	const realtimeSinkNodeRef = useRef<GainNode | null>(null);
	const realtimeStreamRef = useRef<MediaStream | null>(null);
	const realtimeReadyRef = useRef(false);

	const cleanupRealtimeAudioGraph = () => {
		realtimeReadyRef.current = false;

		if (realtimeProcessorRef.current) {
			realtimeProcessorRef.current.onaudioprocess = null;
			realtimeProcessorRef.current.disconnect();
			realtimeProcessorRef.current = null;
		}

		if (realtimeSourceNodeRef.current) {
			realtimeSourceNodeRef.current.disconnect();
			realtimeSourceNodeRef.current = null;
		}

		if (realtimeSinkNodeRef.current) {
			realtimeSinkNodeRef.current.disconnect();
			realtimeSinkNodeRef.current = null;
		}

		if (realtimeAudioContextRef.current) {
			void realtimeAudioContextRef.current.close();
			realtimeAudioContextRef.current = null;
		}

		if (realtimeStreamRef.current) {
			realtimeStreamRef.current.getTracks().forEach((track) => track.stop());
			realtimeStreamRef.current = null;
		}
	};

	const stopRealtimeTranscription = () => {
		cleanupRealtimeAudioGraph();

		const socket = realtimeSocketRef.current;
		realtimeSocketRef.current = null;
		if (socket && socket.readyState === WebSocket.OPEN) {
			socket.close(1000, "recording_stopped");
		}
		setRealtimeStatus("idle");
	};

	const startRealtimeCapture = async (constraints: MediaTrackConstraints) => {
		const stream = await navigator.mediaDevices.getUserMedia({ audio: constraints });
		realtimeStreamRef.current = stream;

		const audioContext = new AudioContext();
		realtimeAudioContextRef.current = audioContext;

		const source = audioContext.createMediaStreamSource(stream);
		realtimeSourceNodeRef.current = source;

		const processor = audioContext.createScriptProcessor(4096, 1, 1);
		realtimeProcessorRef.current = processor;

		const sink = audioContext.createGain();
		sink.gain.value = 0;
		realtimeSinkNodeRef.current = sink;

		processor.onaudioprocess = (event) => {
			if (!realtimeReadyRef.current) return;
			const socket = realtimeSocketRef.current;
			if (!socket || socket.readyState !== WebSocket.OPEN) return;

			const channelData = event.inputBuffer.getChannelData(0);
			if (!channelData || channelData.length === 0) return;

			const copy = new Float32Array(channelData);
			const pcm16k = downsampleTo16k(copy, audioContext.sampleRate);
			if (pcm16k.length > 0) {
				socket.send(pcm16k.buffer);
			}
		};

		source.connect(processor);
		processor.connect(sink);
		sink.connect(audioContext.destination);
	};

	const startRealtimeTranscription = async (constraints: MediaTrackConstraints) => {
		cleanupRealtimeAudioGraph();
		setRealtimeError("");
		setLiveTranscript("");
		setRealtimeStatus("connecting");

		const protocol = window.location.protocol === "https:" ? "wss" : "ws";
		const url = `${protocol}://${window.location.host}/api/v1/transcription/realtime/ws`;
		const socket = new WebSocket(url);
		realtimeSocketRef.current = socket;

		socket.onmessage = async (event) => {
			if (typeof event.data !== "string") {
				return;
			}

			let payload: RealtimeEvent | null = null;
			try {
				payload = JSON.parse(event.data) as RealtimeEvent;
			} catch {
				return;
			}

			if (!payload?.type) return;

			if (payload.type === "ready") {
				setRealtimeStatus("ready");
				realtimeReadyRef.current = true;
				try {
					await startRealtimeCapture(constraints);
				} catch (error) {
					console.error("Failed to start realtime capture:", error);
					setRealtimeStatus("error");
					setRealtimeError("Failed to capture audio for realtime transcription.");
				}
				return;
			}

			if (payload.type === "text" && payload.text) {
				setLiveTranscript((prev) => (prev ? `${prev}\n${payload.text}` : payload.text ?? ""));
				return;
			}

			if (payload.type === "error") {
				setRealtimeStatus("error");
				realtimeReadyRef.current = false;
				setRealtimeError(payload.message || "Realtime transcription failed.");
			}
		};

		socket.onerror = () => {
			realtimeReadyRef.current = false;
			setRealtimeStatus("error");
			setRealtimeError("Failed to connect realtime transcription channel.");
		};

		socket.onclose = () => {
			realtimeReadyRef.current = false;
			cleanupRealtimeAudioGraph();
			if (isRecording) {
				setRealtimeStatus("error");
				setRealtimeError("Realtime transcription connection closed.");
			}
		};
	};

	useEffect(() => {
		if (!isOpen) {
			stopRealtimeTranscription();
		}
	}, [isOpen]);

	useEffect(() => {
		return () => {
			stopRealtimeTranscription();
		};
	}, []);

	// Initialize WaveSurfer and RecordPlugin when dialog opens
	useEffect(() => {
		if (!isOpen) return;

		let activeStream: MediaStream | null = null;
		let ws: WaveSurfer | null = null;

		const init = async () => {
			try {
				activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });

				const devices = await RecordPlugin.getAvailableAudioDevices();
				setAvailableDevices(devices);

				if (devices.length > 0) {
					const deviceExists = devices.some((d) => d.deviceId === selectedDevice);
					if (!selectedDevice || !deviceExists) {
						setSelectedDevice(devices[0].deviceId);
					}
				}

				if (!micContainerRef.current) return;

				ws = WaveSurfer.create({
					container: micContainerRef.current,
					waveColor: "rgb(168, 85, 247)",
					progressColor: "rgb(147, 51, 234)",
					height: 80,
					normalize: true,
					interact: false,
				});

				setWavesurfer(ws);

				const recordPlugin = ws.registerPlugin(
					RecordPlugin.create({
						renderRecordedAudio: false,
						scrollingWaveform: true,
						continuousWaveform: true,
						continuousWaveformDuration: 30,
						mediaRecorderTimeslice: 1000,
					}),
				);

				recordPlugin.on("record-end", (blob: Blob) => {
					setRecordedBlob(blob);
					setIsRecording(false);
					setIsPaused(false);
					stopRealtimeTranscription();
				});

				recordPlugin.on("record-progress", (time: number) => {
					setRecordingTime(time);
				});

				setRecord(recordPlugin);
			} catch (error) {
				console.error("Failed to initialize recorder:", error);
			} finally {
				if (activeStream) {
					activeStream.getTracks().forEach((track) => track.stop());
				}
			}
		};

		const timeoutId = setTimeout(init, 100);

		return () => {
			clearTimeout(timeoutId);
			if (ws) {
				ws.destroy();
			}
			if (activeStream) {
				activeStream.getTracks().forEach((track) => track.stop());
			}
		};
	}, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

	useEffect(() => {
		const originalTitle = document.title;

		const handleBeforeUnload = (e: BeforeUnloadEvent) => {
			if (isRecording) {
				e.preventDefault();
				e.returnValue = "Recording in progress. Are you sure you want to leave?";
				return e.returnValue;
			}
		};

		const handleVisibilityChange = () => {};

		if (isRecording) {
			document.title = `Recording... - ${originalTitle}`;
			window.addEventListener("beforeunload", handleBeforeUnload);
			document.addEventListener("visibilitychange", handleVisibilityChange);
		} else {
			document.title = originalTitle;
		}

		return () => {
			document.title = originalTitle;
			window.removeEventListener("beforeunload", handleBeforeUnload);
			document.removeEventListener("visibilitychange", handleVisibilityChange);
		};
	}, [isRecording]);

	const startRecording = async () => {
		if (!record) {
			alert("Recorder not initialized. Please close and reopen the dialog.");
			return;
		}

		try {
			const constraints: MediaTrackConstraints = {
				deviceId: selectedDevice ? { exact: selectedDevice } : undefined,
				echoCancellation: false,
				noiseSuppression: false,
				autoGainControl: true,
				channelCount: 1,
			};

			await record.startRecording(constraints);
			setIsRecording(true);
			setIsPaused(false);
			setRecordingTime(0);
			setRecordedBlob(null);
			void startRealtimeTranscription(constraints);
		} catch (error) {
			console.error("Failed to start recording:", error);
			alert("Failed to start recording. Please check microphone permissions and try again.");
		}
	};

	const stopRecording = () => {
		if (!record) return;
		record.stopRecording();
	};

	const togglePauseRecording = () => {
		if (!record) return;

		if (isPaused) {
			record.resumeRecording();
			setIsPaused(false);
		} else {
			record.pauseRecording();
			setIsPaused(true);
		}
	};

	const formatTime = (timeMs: number) => {
		const minutes = Math.floor(timeMs / 60000);
		const seconds = Math.floor((timeMs % 60000) / 1000);
		return `${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
	};

	const handleUpload = async () => {
		if (!recordedBlob) return;

		setIsUploading(true);
		try {
			await onRecordingComplete(recordedBlob, title || `Recording ${new Date().toISOString()}`);
			setRecordedBlob(null);
			setTitle("");
			setRecordingTime(0);
			setLiveTranscript("");
			onClose();
		} catch (error) {
			console.error("Failed to upload recording:", error);
			alert("Failed to upload recording");
		} finally {
			setIsUploading(false);
		}
	};

	const handleClose = () => {
		if (isRecording) {
			stopRecording();
		}
		stopRealtimeTranscription();
		setRecordedBlob(null);
		setTitle("");
		setRecordingTime(0);
		setIsRecording(false);
		setIsPaused(false);
		setLiveTranscript("");
		onClose();
	};

	const realtimeStatusText =
		realtimeStatus === "connecting"
			? "Initializing FireRed realtime ASR..."
			: realtimeStatus === "ready"
				? "Realtime transcription running (FireRedVAD)"
				: realtimeStatus === "error"
					? "Realtime transcription unavailable"
					: "Realtime transcription idle";

	return (
		<Dialog open={isOpen} onOpenChange={handleClose}>
			<DialogContent className="sm:max-w-[700px] bg-white dark:bg-carbon-800 border-carbon-200 dark:border-carbon-700">
				<DialogHeader>
					<DialogTitle className="text-carbon-900 dark:text-carbon-100 text-xl font-bold">
						Record Audio
					</DialogTitle>
					<DialogDescription className="text-carbon-600 dark:text-carbon-400">
						Record audio directly from your microphone, view realtime transcript, then upload.
					</DialogDescription>
				</DialogHeader>

				<div className="space-y-6 py-4">
					<div className="space-y-2">
						<label className="text-sm font-medium text-carbon-700 dark:text-carbon-300">
							Recording Title (Optional)
						</label>
						<Input
							value={title}
							onChange={(e) => setTitle(e.target.value)}
							placeholder="Enter a title for your recording..."
							className="bg-white dark:bg-carbon-800 border-carbon-300 dark:border-carbon-600 text-carbon-900 dark:text-carbon-100"
							disabled={isRecording}
						/>
					</div>

					{availableDevices.length > 1 && (
						<div className="space-y-2">
							<label className="text-sm font-medium text-carbon-700 dark:text-carbon-300">
								Microphone
							</label>
							<DropdownMenu>
								<DropdownMenuTrigger asChild disabled={isRecording}>
									<Button
										variant="outline"
										className="w-full justify-between bg-white dark:bg-carbon-800 border-carbon-300 dark:border-carbon-600 hover:bg-carbon-50 dark:hover:bg-carbon-700"
									>
										<div className="flex items-center gap-2">
											<Settings className="h-4 w-4" />
											<span className="truncate">
												{availableDevices.find((d) => d.deviceId === selectedDevice)?.label ||
													`Microphone ${selectedDevice.slice(0, 8)}`}
											</span>
										</div>
										<ChevronDown className="h-4 w-4 opacity-50" />
									</Button>
								</DropdownMenuTrigger>
								<DropdownMenuContent className="w-full min-w-[400px] bg-white dark:bg-carbon-900 border-carbon-200 dark:border-carbon-700">
									{availableDevices.map((device) => (
										<DropdownMenuItem
											key={device.deviceId}
											onClick={() => setSelectedDevice(device.deviceId)}
											className="flex items-center gap-3 px-3 py-2 cursor-pointer hover:bg-carbon-100 dark:hover:bg-carbon-700"
										>
											<Mic className="h-4 w-4 text-carbon-500" />
											<div className="flex-1">
												<div className="text-sm font-medium text-carbon-900 dark:text-carbon-100">
													{device.label || `Microphone ${device.deviceId.slice(0, 8)}`}
												</div>
												<div className="text-xs text-carbon-500 dark:text-carbon-400">
													Device ID: {device.deviceId.slice(0, 20)}...
												</div>
											</div>
											{selectedDevice === device.deviceId && (
												<div className="h-2 w-2 bg-brand-500 rounded-full"></div>
											)}
										</DropdownMenuItem>
									))}
								</DropdownMenuContent>
							</DropdownMenu>
						</div>
					)}

					<div className="text-center">
						<div className="text-3xl font-mono font-bold text-carbon-900 dark:text-carbon-100 mb-2">
							{formatTime(recordingTime)}
						</div>
						<div className="flex items-center justify-center gap-2 text-sm text-carbon-600 dark:text-carbon-400">
							{isRecording && !isPaused && (
								<div className="h-2 w-2 bg-red-500 rounded-full animate-pulse"></div>
							)}
							<span>
								{isRecording ? (isPaused ? "Recording Paused" : "Recording...") : "Ready to Record"}
							</span>
						</div>
						{isRecording && (
							<div className="text-xs text-brand-600 dark:text-brand-400 mt-1">
								Recording continues even if you switch tabs
							</div>
						)}
					</div>

					<div className="relative">
						<div
							ref={micContainerRef}
							className="w-full rounded-lg p-4 bg-carbon-50 dark:bg-carbon-800/50 min-h-[120px]"
						/>
						{!isRecording && !recordedBlob && (
							<div className="absolute inset-0 flex items-center justify-center pointer-events-none">
								<div className="text-carbon-400 dark:text-carbon-500 text-sm text-center">
									<Mic className="h-8 w-8 mx-auto mb-2 opacity-50" />
									<div>Waveform will appear here during recording</div>
									{!wavesurfer && <div className="text-xs text-red-400 mt-1">Initializing recorder...</div>}
									{wavesurfer && !record && <div className="text-xs text-yellow-400 mt-1">Recorder plugin loading...</div>}
									{wavesurfer && record && <div className="text-xs text-green-400 mt-1">Ready to record</div>}
								</div>
							</div>
						)}
					</div>

					<div className="space-y-2">
						<div className="text-sm font-medium text-carbon-700 dark:text-carbon-300">
							Live Transcript (FireRedVAD)
						</div>
						<div className="rounded-lg border border-carbon-200 dark:border-carbon-700 bg-carbon-50/70 dark:bg-carbon-900/40 p-3 min-h-[90px] max-h-[170px] overflow-y-auto text-sm text-carbon-800 dark:text-carbon-200 whitespace-pre-wrap">
							{liveTranscript || "Realtime text will appear here after speech segments are detected."}
						</div>
						<div className="text-xs text-carbon-500 dark:text-carbon-400">{realtimeStatusText}</div>
						{realtimeError && <div className="text-xs text-red-500">{realtimeError}</div>}
					</div>

					<div className="flex justify-center gap-4">
						{!isRecording && !recordedBlob && (
							<Button
								onClick={startRecording}
								size="lg"
								className="bg-red-500 hover:bg-red-600 text-white px-8 py-3 rounded-xl font-medium transition-all duration-300 hover:scale-105"
							>
								<Mic className="h-5 w-5 mr-2" />
								Start Recording
							</Button>
						)}

						{isRecording && (
							<>
								<Button
									onClick={togglePauseRecording}
									size="lg"
									variant="outline"
									className="border-carbon-300 dark:border-carbon-600 hover:bg-carbon-100 dark:hover:bg-carbon-700 px-6 py-3 rounded-xl"
								>
									{isPaused ? (
										<>
											<Play className="h-5 w-5 mr-2" />
											Resume
										</>
									) : (
										<>
											<Pause className="h-5 w-5 mr-2" />
											Pause
										</>
									)}
								</Button>
								<Button
									onClick={stopRecording}
									size="lg"
									className="bg-carbon-600 hover:bg-carbon-700 text-white px-6 py-3 rounded-xl"
								>
									<Square className="h-5 w-5 mr-2" />
									Stop
								</Button>
							</>
						)}

						{recordedBlob && (
							<Button
								onClick={handleUpload}
								size="lg"
								disabled={isUploading}
								className="bg-brand-500 hover:bg-brand-600 text-white px-8 py-3 rounded-xl font-medium transition-all duration-300 hover:scale-105"
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
						)}
					</div>

					{recordedBlob && (
						<div className="text-center text-sm text-green-600 dark:text-green-400">
							Recording completed. Upload when ready.
						</div>
					)}
				</div>
			</DialogContent>
		</Dialog>
	);
}
