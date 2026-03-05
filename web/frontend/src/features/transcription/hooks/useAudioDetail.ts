import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAuth } from "@/features/auth/hooks/useAuth";


// Types
export interface MultiTrackFile {
    id: number;
    file_name: string;
    file_path: string;
    track_index: number;
}

export interface MultiTrackTiming {
    track_name: string;
    start_time: string;
    end_time: string;
    duration: number; // milliseconds
}

export interface ExecutionData {
    id?: string;
    transcription_job_id: string;
    started_at?: string;
    completed_at?: string | null;
    processing_duration?: number | null; // milliseconds
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    actual_parameters?: any;
    status?: string;
    error_message?: string | null;
    created_at?: string;
    updated_at?: string;
    // Multi-track specific fields
    is_multi_track?: boolean;
    multi_track_timings?: MultiTrackTiming[];
    merge_start_time?: string | null;
    merge_end_time?: string | null;
    merge_duration?: number | null; // milliseconds
    multi_track_files?: MultiTrackFile[];
    // Graceful empty response fields
    available?: boolean;
    message?: string;
}

export interface LogsData {
    job_id: string;
    available: boolean;
    content: string;
    message?: string;
}

export interface AudioFile {
    id: string;
    title?: string;
    status: "uploaded" | "pending" | "processing" | "completed" | "failed";
    transcription_progress?: number;
    transcription_stage?: string;
    transcription_stage_progress?: number;
    openclaw_sent_at?: string | null;
    openclaw_profile_name?: string | null;
    created_at: string;
    audio_path: string;
    diarization?: boolean;
    is_multi_track?: boolean;
    multi_track_files?: MultiTrackFile[];
    merged_audio_path?: string;
    merge_status?: string;
    merge_error?: string;
    parameters?: {
        diarize?: boolean;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        [key: string]: any;
    };
}

export interface WordSegment {
    start: number;
    end: number;
    word: string;
    score: number;
    speaker?: string;
}

export interface TranscriptSegment {
    start: number;
    end: number;
    text: string;
    speaker?: string;
}

export interface Transcript {
    text: string;
    segments?: TranscriptSegment[];
    word_segments?: WordSegment[];
}

type ProgressFields = Pick<AudioFile, "transcription_progress" | "transcription_stage" | "transcription_stage_progress">;

const hasProgressData = (audio: ProgressFields): boolean =>
    typeof audio.transcription_progress === "number" ||
    typeof audio.transcription_stage === "string" ||
    typeof audio.transcription_stage_progress === "number";

const mergeProgressIfMissing = (incoming: AudioFile, cached: ProgressFields | null): AudioFile => {
    if (!cached || !hasProgressData(cached)) return incoming;

    const isActive = incoming.status === "processing" || incoming.status === "pending";
    if (!isActive) return incoming;

    return {
        ...incoming,
        transcription_progress:
            typeof incoming.transcription_progress === "number"
                ? incoming.transcription_progress
                : cached.transcription_progress,
        transcription_stage: incoming.transcription_stage || cached.transcription_stage,
        transcription_stage_progress:
            typeof incoming.transcription_stage_progress === "number"
                ? incoming.transcription_stage_progress
                : cached.transcription_stage_progress,
    };
};

const findCachedProgress = (queryClient: ReturnType<typeof useQueryClient>, audioId: string): ProgressFields | null => {
    const detailCache = queryClient.getQueryData<AudioFile>(["audio", audioId]);
    if (detailCache && hasProgressData(detailCache)) {
        return {
            transcription_progress: detailCache.transcription_progress,
            transcription_stage: detailCache.transcription_stage,
            transcription_stage_progress: detailCache.transcription_stage_progress,
        };
    }

    const listCaches = queryClient.getQueriesData({ queryKey: ["audioFiles"] });
    for (const [, data] of listCaches) {
        if (!data || typeof data !== "object") continue;

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const listData = data as any;

        if (Array.isArray(listData.pages)) {
            for (const page of listData.pages) {
                const job = page?.jobs?.find((item: AudioFile) => item.id === audioId);
                if (job && hasProgressData(job)) {
                    return {
                        transcription_progress: job.transcription_progress,
                        transcription_stage: job.transcription_stage,
                        transcription_stage_progress: job.transcription_stage_progress,
                    };
                }
            }
        }

        if (Array.isArray(listData.jobs)) {
            const job = listData.jobs.find((item: AudioFile) => item.id === audioId);
            if (job && hasProgressData(job)) {
                return {
                    transcription_progress: job.transcription_progress,
                    transcription_stage: job.transcription_stage,
                    transcription_stage_progress: job.transcription_stage_progress,
                };
            }
        }
    }

    return null;
};

export function useAudioDetail(audioId: string) {
    const { getAuthHeaders } = useAuth();
    const queryClient = useQueryClient();

    return useQuery({
        queryKey: ["audio", audioId],
        queryFn: async () => {
            const response = await fetch(`/api/v1/transcription/${audioId}`, {
                headers: getAuthHeaders(),
            });
            if (!response.ok) throw new Error("Failed to fetch audio details");
            const data = await response.json() as AudioFile;
            return mergeProgressIfMissing(data, findCachedProgress(queryClient, audioId));
        },
        // Poll while processing or pending
        refetchInterval: (query) => {
            const status = query.state.data?.status;
            if (status === "processing" || status === "pending") {
                return 3000;
            }
            return false;
        },
    });
}

export function useTranscript(audioId: string, enabled: boolean) {
    const { getAuthHeaders } = useAuth();

    return useQuery({
        queryKey: ["transcript", audioId],
        queryFn: async () => {
            const response = await fetch(`/api/v1/transcription/${audioId}/transcript`, {
                headers: getAuthHeaders(),
            });
            if (!response.ok) throw new Error("Failed to fetch transcript");
            const data = await response.json();

            // Handle graceful empty responses (available=false)
            if (data.available === false || !data.transcript) {
                return null; // Return null to indicate no transcript
            }

            // Normalize transcript structure
            if (typeof data.transcript === "string") {
                return { text: data.transcript } as Transcript;
            } else if (data.transcript.text) {
                return {
                    text: data.transcript.text,
                    segments: data.transcript.segments,
                    word_segments: data.transcript.word_segments,
                } as Transcript;
            } else if (data.transcript.segments) {
                const fullText = data.transcript.segments
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    .map((segment: any) => segment.text)
                    .join(" ");
                return {
                    text: fullText,
                    segments: data.transcript.segments,
                    word_segments: data.transcript.word_segments,
                } as Transcript;
            }

            return { text: "" } as Transcript;
        },
        enabled: enabled,
    });
}

export function useExecutionData(audioId: string) {
    const { getAuthHeaders } = useAuth();
    return useQuery({
        queryKey: ["executionData", audioId],
        queryFn: async () => {
            const response = await fetch(`/api/v1/transcription/${audioId}/execution`, {
                headers: getAuthHeaders(),
            });
            if (!response.ok) throw new Error("Failed to fetch execution data");
            return response.json() as Promise<ExecutionData>;
        },
        enabled: !!audioId,
    });
}

export function useLogs(audioId: string) {
    const { getAuthHeaders } = useAuth();
    return useQuery({
        queryKey: ["logs", audioId],
        queryFn: async () => {
            const response = await fetch(`/api/v1/transcription/${audioId}/logs`, {
                headers: getAuthHeaders(),
            });
            if (!response.ok) throw new Error("Failed to fetch logs");
            return response.json() as Promise<LogsData>;
        },
        enabled: !!audioId,
    });
}

export function useUpdateTitle(audioId: string) {
    const { getAuthHeaders } = useAuth();
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: async (newTitle: string) => {
            const response = await fetch(`/api/v1/transcription/${audioId}/title`, {
                method: "PUT",
                headers: {
                    "Content-Type": "application/json",
                    ...getAuthHeaders(),
                },
                body: JSON.stringify({ title: newTitle }),
            });
            if (!response.ok) {
                const msg = await response.text();
                throw new Error(msg || "Failed to update title");
            }
            return response.json();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["audio", audioId] });
            queryClient.invalidateQueries({ queryKey: ["audioFiles"] }); // Update list too
        },
    });
}
