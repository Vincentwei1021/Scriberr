const STAGE_LABELS: Record<string, string> = {
    starting: "Initializing",
    initializing_asr: "Initializing ASR",
    preprocessing: "Preprocessing",
    transcribing: "Transcribing",
    finalizing_transcription: "Finalizing Transcription",
    transcription_completed: "Transcription Completed",
    diarizing: "Diarizing",
    finalizing_diarization: "Finalizing Diarization",
    diarization_completed: "Diarization Completed",
    saving: "Saving",
    completed: "Completed",
    failed: "Failed",
};

const NUMERIC_STAGE_SET = new Set(["transcribing", "diarizing"]);

export const clampPercent = (value: number): number => Math.max(0, Math.min(100, Math.round(value)));

export const formatStageLabel = (stage?: string): string => {
    if (!stage) return "Processing";
    return STAGE_LABELS[stage] || stage.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
};

export const stageSupportsNumericProgress = (stage?: string): boolean => {
    if (!stage) return false;
    return NUMERIC_STAGE_SET.has(stage);
};

export const formatStageWithProgress = (stage?: string, stageProgress?: number): string => {
    const label = formatStageLabel(stage);
    if (stageSupportsNumericProgress(stage) && typeof stageProgress === "number") {
        return `${label} (${clampPercent(stageProgress)}%)`;
    }
    return label;
};
