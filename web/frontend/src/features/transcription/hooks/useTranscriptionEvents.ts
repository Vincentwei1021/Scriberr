import { useEffect, useRef } from 'react';
import { useAuth } from '@/features/auth/hooks/useAuth';
import { useQueryClient } from '@tanstack/react-query';
import type { AudioFile } from '@/features/transcription/hooks/useAudioFiles';

interface JobUpdateEvent {
    type: string;
    payload: {
        job_id: string;
        status: string;
        error?: string;
        progress?: number;
        stage?: string;
        stage_progress?: number;
    };
}

const resolveStageProgress = (
    currentStage?: string,
    currentStageProgress?: number,
    incomingStage?: string,
    incomingStageProgress?: number,
): number | undefined => {
    if (typeof incomingStageProgress === "number") {
        return incomingStageProgress;
    }

    // Clear stale percentage when stage changed but no numeric progress is provided.
    if (incomingStage && incomingStage !== currentStage) {
        return undefined;
    }

    return currentStageProgress;
};

export const useTranscriptionEvents = (jobId: string | null) => {
    const { getAuthHeaders } = useAuth();
    const queryClient = useQueryClient();
    const abortControllerRef = useRef<AbortController | null>(null);

    useEffect(() => {
        if (!jobId) return;

        // Cleanup previous connection if any
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }

        const abortController = new AbortController();
        abortControllerRef.current = abortController;

        const connect = async () => {
            try {
                // Use trailing slash to avoid Gin's redirect on /events -> /events/.
                const response = await fetch(`/api/v1/events/?job_id=${jobId}`, {
                    headers: getAuthHeaders(),
                    signal: abortController.signal,
                });

                if (!response.ok) {
                    throw new Error(`SSE connection failed: ${response.status}`);
                }

                if (!response.body) {
                    throw new Error('No response body');
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                const processBuffer = () => {
                    // Normalize line endings so both LF and CRLF streams are parsed consistently.
                    buffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

                    const blocks = buffer.split('\n\n');
                    buffer = blocks.pop() || '';

                    for (const block of blocks) {
                        const lines = block.split('\n');
                        const dataLines: string[] = [];

                        for (const rawLine of lines) {
                            const line = rawLine.trimEnd();
                            if (!line || line.startsWith(':')) continue;
                            if (line.startsWith('data:')) {
                                dataLines.push(line.slice(5).trimStart());
                            }
                        }

                        if (dataLines.length === 0) continue;

                        const data = dataLines.join('\n');
                        try {
                            const event = JSON.parse(data);
                            handleEvent(event);
                        } catch (e) {
                            console.error('Failed to parse SSE data:', e);
                        }
                    }
                };

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;
                    processBuffer();
                }

                // Try parsing any final buffered frame.
                if (buffer.trim()) {
                    processBuffer();
                }
            } catch (error) {
                if ((error as Error).name !== 'AbortError') {
                    // Ignore "Error in input stream" which happens on abort/close in some browsers
                    const errorMsg = (error as Error).message;
                    if (!errorMsg.includes('Error in input stream')) {
                        console.error('SSE connection error, reconnecting in 5s...', error);
                        setTimeout(() => {
                            if (!abortController.signal.aborted) {
                                connect();
                            }
                        }, 5000);
                    }
                }
            }
        };

        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const handleEvent = (event: any) => {
            if (event.type === 'job_update') {
                const payload = event.payload as JobUpdateEvent['payload'];

                // Optimistically update the list
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                queryClient.setQueriesData({ queryKey: ['audioFiles'] }, (oldData: any) => {
                    if (!oldData) return oldData;

                    // Handle generic infinite query structure
                    if (oldData.pages) {
                        return {
                            ...oldData,
                            // eslint-disable-next-line @typescript-eslint/no-explicit-any
                            pages: oldData.pages.map((page: any) => ({
                                ...page,
                                jobs: page.jobs.map((job: AudioFile) => {
                                    if (job.id === payload.job_id) {
                                        const nextStage = payload.stage || job.transcription_stage;
                                        return {
                                            ...job,
                                            status: payload.status,
                                            error_message: payload.error || job.error_message,
                                            transcription_progress: typeof payload.progress === "number"
                                                ? payload.progress
                                                : job.transcription_progress,
                                            transcription_stage: nextStage,
                                            transcription_stage_progress: resolveStageProgress(
                                                job.transcription_stage,
                                                job.transcription_stage_progress,
                                                payload.stage,
                                                payload.stage_progress,
                                            ),
                                        };
                                    }
                                    return job;
                                }),
                            })),
                        };
                    }

                    // Handle standard query structure (if used elsewhere)
                    if (oldData.jobs) {
                        return {
                            ...oldData,
                            jobs: oldData.jobs.map((job: AudioFile) => {
                                if (job.id === payload.job_id) {
                                    const nextStage = payload.stage || job.transcription_stage;
                                    return {
                                        ...job,
                                        status: payload.status,
                                        error_message: payload.error || job.error_message,
                                        transcription_progress: typeof payload.progress === "number"
                                            ? payload.progress
                                            : job.transcription_progress,
                                        transcription_stage: nextStage,
                                        transcription_stage_progress: resolveStageProgress(
                                            job.transcription_stage,
                                            job.transcription_stage_progress,
                                            payload.stage,
                                            payload.stage_progress,
                                        ),
                                    };
                                }
                                return job;
                            }),
                        };
                    }

                    return oldData;
                });

                // Keep audio detail query in sync so detail page can show progress too
                queryClient.setQueryData(['audio', payload.job_id], (oldData: AudioFile | undefined) => {
                    if (!oldData) return oldData;
                    const nextStage = payload.stage || oldData.transcription_stage;
                    return {
                        ...oldData,
                        status: payload.status as AudioFile['status'],
                        error_message: payload.error || oldData.error_message,
                        transcription_progress: typeof payload.progress === "number"
                            ? payload.progress
                            : oldData.transcription_progress,
                        transcription_stage: nextStage,
                        transcription_stage_progress: resolveStageProgress(
                            oldData.transcription_stage,
                            oldData.transcription_stage_progress,
                            payload.stage,
                            payload.stage_progress,
                        ),
                    };
                });
            }
        };

        connect();

        return () => {
            abortController.abort();
        };
    }, [getAuthHeaders, queryClient, jobId]);
};
