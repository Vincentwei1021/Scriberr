import {
    createContext,
    useContext,
    useState,
    useCallback,
    useEffect,
    useRef,
    type PropsWithChildren,
} from "react";
import { useLocation } from "react-router-dom";
import { useAudioUpload, useMultiTrackUpload } from "@/features/transcription/hooks/useAudioFiles";
import { useToast } from "@/components/ui/toast";
import { MultiTrackUploadDialog } from "@/features/transcription/components/MultiTrackUploadDialog";
import { useAuth } from "@/features/auth/hooks/useAuth";

// Types
interface FileWithType {
    file: File;
    isVideo: boolean;
}

interface UploadProgress {
    fileName: string;
    status: "uploading" | "success" | "error";
    error?: string;
}

interface OpenClawProfileSummary {
    id: string;
    name: string;
}

interface PendingOpenClawSend {
    jobId: string;
    profileId: string;
    title?: string;
}

interface GlobalUploadContextValue {
    // File upload
    handleFileSelect: (
        files: File | File[] | FileWithType | FileWithType[],
        source?: string
    ) => Promise<void>;
    // Multi-track
    handleMultiTrackUpload: (
        files: File[],
        aupFile: File,
        title: string
    ) => Promise<void>;
    openMultiTrackDialog: () => void;
    // Recording completion
    handleRecordingComplete: (blob: Blob, title: string, source?: string) => Promise<void>;
    // State
    isUploading: boolean;
    uploadProgress: UploadProgress[];
    // For Dashboard to render its own progress bar
    isOnDashboard: boolean;
    sendToOpenClawAfterTranscription: boolean;
    setSendToOpenClawAfterTranscription: (enabled: boolean) => void;
    selectedOpenClawProfileId: string;
    setSelectedOpenClawProfileId: (profileId: string) => void;
    openClawProfiles: OpenClawProfileSummary[];
    openClawProfilesLoading: boolean;
    openClawProfilesError: string;
    refreshOpenClawProfiles: () => Promise<void>;
}

const GlobalUploadContext = createContext<GlobalUploadContextValue | null>(
    null
);

export function GlobalUploadProvider({ children }: PropsWithChildren) {
    const { mutateAsync: uploadFile } = useAudioUpload();
    const { mutateAsync: uploadMultiTrack } = useMultiTrackUpload();
    const { toast } = useToast();
    const { getAuthHeaders } = useAuth();
    const location = useLocation();

    // Check if we're on the dashboard (home page)
    const isOnDashboard = location.pathname === "/" || location.pathname === "";

    // Upload state
    const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
    const [isUploading, setIsUploading] = useState(false);
    const [sendToOpenClawAfterTranscription, setSendToOpenClawAfterTranscription] = useState(false);
    const [selectedOpenClawProfileId, setSelectedOpenClawProfileId] = useState("");
    const [openClawProfiles, setOpenClawProfiles] = useState<OpenClawProfileSummary[]>([]);
    const [openClawProfilesLoading, setOpenClawProfilesLoading] = useState(false);
    const [openClawProfilesError, setOpenClawProfilesError] = useState("");
    const [pendingOpenClawSends, setPendingOpenClawSends] = useState<PendingOpenClawSend[]>([]);

    // Multi-track dialog state
    const [isMultiTrackDialogOpen, setIsMultiTrackDialogOpen] = useState(false);
    const [multiTrackPreview, setMultiTrackPreview] = useState<{
        audioFiles: File[];
        aupFile: File;
        title: string;
    } | null>(null);

    const refreshOpenClawProfiles = useCallback(async () => {
        try {
            setOpenClawProfilesLoading(true);
            setOpenClawProfilesError("");
            const res = await fetch("/api/v1/openclaw/profiles", {
                headers: getAuthHeaders(),
            });
            if (!res.ok) {
                throw new Error("Failed to load OpenClaw profiles");
            }
            const data = await res.json();
            const nextProfiles = Array.isArray(data)
                ? data.filter((item): item is OpenClawProfileSummary => {
                    return !!item && typeof item.id === "string" && typeof item.name === "string";
                })
                : [];
            setOpenClawProfiles(nextProfiles);
            setSelectedOpenClawProfileId((prev) => {
                if (prev && nextProfiles.some((profile) => profile.id === prev)) {
                    return prev;
                }
                return nextProfiles[0]?.id ?? "";
            });
        } catch (error) {
            setOpenClawProfilesError(error instanceof Error ? error.message : "Failed to load OpenClaw profiles");
            setOpenClawProfiles([]);
            setSelectedOpenClawProfileId("");
        } finally {
            setOpenClawProfilesLoading(false);
        }
    }, [getAuthHeaders]);

    useEffect(() => {
        if (!isOnDashboard) return;
        void refreshOpenClawProfiles();
    }, [isOnDashboard, refreshOpenClawProfiles]);

    const queueOpenClawSend = useCallback((jobId: string, profileId: string, title?: string) => {
        if (!jobId || !profileId) return;
        setPendingOpenClawSends((prev) => {
            if (prev.some((item) => item.jobId === jobId)) {
                return prev;
            }
            return [...prev, { jobId, profileId, title }];
        });
    }, []);

    const pendingOpenClawSendsRef = useRef<PendingOpenClawSend[]>([]);
    useEffect(() => {
        pendingOpenClawSendsRef.current = pendingOpenClawSends;
    }, [pendingOpenClawSends]);

    useEffect(() => {
        if (pendingOpenClawSends.length === 0) return;

        let cancelled = false;
        let polling = false;

        const removePendingJob = (jobId: string) => {
            setPendingOpenClawSends((prev) => prev.filter((item) => item.jobId !== jobId));
        };

        const pollOpenClawQueue = async () => {
            if (cancelled || polling) return;
            polling = true;
            try {
                const currentQueue = pendingOpenClawSendsRef.current;
                for (const pendingItem of currentQueue) {
                    if (cancelled) break;

                    const jobRes = await fetch(`/api/v1/transcription/${pendingItem.jobId}`, {
                        headers: getAuthHeaders(),
                    });

                    if (jobRes.status === 404) {
                        removePendingJob(pendingItem.jobId);
                        continue;
                    }

                    if (!jobRes.ok) {
                        continue;
                    }

                    const jobData = await jobRes.json() as { status?: string; title?: string };
                    if (jobData.status === "completed") {
                        const sendRes = await fetch(`/api/v1/transcription/${pendingItem.jobId}/send-openclaw`, {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json",
                                ...getAuthHeaders(),
                            },
                            body: JSON.stringify({ profile_id: pendingItem.profileId }),
                        });

                        if (sendRes.ok) {
                            toast({
                                title: "Sent to OpenClaw",
                                description: `${jobData.title || pendingItem.title || "Transcript"} sent automatically after transcription.`,
                            });
                        } else {
                            const errText = await sendRes.text();
                            toast({
                                title: "OpenClaw Send Failed",
                                description: errText || "Failed to send transcript to OpenClaw automatically.",
                            });
                        }

                        removePendingJob(pendingItem.jobId);
                        continue;
                    }

                    if (jobData.status === "failed") {
                        toast({
                            title: "Transcription Failed",
                            description: `${jobData.title || pendingItem.title || "Transcript"} failed, skipped automatic OpenClaw send.`,
                        });
                        removePendingJob(pendingItem.jobId);
                    }
                }
            } finally {
                polling = false;
            }
        };

        void pollOpenClawQueue();
        const timer = window.setInterval(() => {
            void pollOpenClawQueue();
        }, 5000);

        return () => {
            cancelled = true;
            window.clearInterval(timer);
        };
    }, [pendingOpenClawSends.length, getAuthHeaders, toast]);

    const handleFileSelect = useCallback(
        async (files: File | File[] | FileWithType | FileWithType[], source?: string) => {
            // Normalize input to an array of FileWithType objects
            const fileArray = Array.isArray(files) ? files : [files];
            const processedFiles = fileArray.map((item) => {
                if ("file" in item && "isVideo" in item) {
                    return item;
                } else {
                    return { file: item as File, isVideo: false };
                }
            });

            if (processedFiles.length === 0) return;

            const autoSendEnabled = sendToOpenClawAfterTranscription && !!selectedOpenClawProfileId;
            if (sendToOpenClawAfterTranscription && !selectedOpenClawProfileId) {
                toast({
                    title: "OpenClaw Profile Required",
                    description: "Enable auto-send requires selecting an OpenClaw profile first.",
                });
            }

            setIsUploading(true);

            // If on dashboard, use progress bar; otherwise use toasts
            if (isOnDashboard) {
                setUploadProgress(
                    processedFiles.map((item) => ({
                        fileName: item.file.name,
                        status: "uploading",
                    }))
                );
            } else {
                toast({
                    title: "Uploading...",
                    description: `Uploading ${processedFiles.length} file(s)`,
                });
            }

            let successCount = 0;

            // Upload files sequentially
            for (let i = 0; i < processedFiles.length; i++) {
                const fileItem = processedFiles[i];
                const file = fileItem.file;
                const isVideo = fileItem.isVideo;

                try {
                    const uploadResult = await uploadFile({ file, isVideo, source }) as { id?: string; title?: string };

                    if (isOnDashboard) {
                        setUploadProgress((prev) =>
                            prev.map((item, index) =>
                                index === i
                                    ? { ...item, status: "success", error: undefined }
                                    : item
                            )
                        );
                    }
                    successCount++;

                    if (autoSendEnabled && typeof uploadResult?.id === "string") {
                        queueOpenClawSend(uploadResult.id, selectedOpenClawProfileId, uploadResult.title || file.name);
                    }
                } catch (error) {
                    if (isOnDashboard) {
                        setUploadProgress((prev) =>
                            prev.map((item, index) =>
                                index === i
                                    ? {
                                        ...item,
                                        status: "error",
                                        error:
                                            error instanceof Error
                                                ? error.message
                                                : "Upload failed",
                                    }
                                    : item
                            )
                        );
                    } else {
                        toast({
                            title: "Upload Failed",
                            description: `Failed to upload ${file.name}`,
                        });
                    }
                }
            }

            setIsUploading(false);

            // Show success toast if not on dashboard
            if (!isOnDashboard && successCount > 0) {
                toast({
                    title: "Upload Complete",
                    description: `Successfully uploaded ${successCount} file(s)`,
                });
            }

            // Auto-hide progress after 3 seconds if all succeeded (for dashboard)
            if (isOnDashboard && successCount === fileArray.length) {
                setTimeout(() => setUploadProgress([]), 3000);
            }
        },
        [
            isOnDashboard,
            uploadFile,
            toast,
            sendToOpenClawAfterTranscription,
            selectedOpenClawProfileId,
            queueOpenClawSend,
        ]
    );

    const handleMultiTrackUpload = useCallback(
        async (files: File[], aupFile: File, title: string) => {
            setIsUploading(true);

            if (isOnDashboard) {
                setUploadProgress([
                    {
                        fileName: `${title} (${files.length} tracks)`,
                        status: "uploading",
                    },
                ]);
            } else {
                toast({
                    title: "Uploading Multi-Track...",
                    description: `Uploading ${title} with ${files.length} tracks`,
                });
            }

            try {
                const uploadResult = await uploadMultiTrack({ files, aupFile, title }) as { id?: string; title?: string };

                if (isOnDashboard) {
                    setUploadProgress([
                        {
                            fileName: `${title} (${files.length} tracks)`,
                            status: "success",
                        },
                    ]);
                    setTimeout(() => setUploadProgress([]), 3000);
                } else {
                    toast({
                        title: "Upload Complete",
                        description: `Successfully uploaded ${title}`,
                    });
                }

                if (sendToOpenClawAfterTranscription && selectedOpenClawProfileId && typeof uploadResult?.id === "string") {
                    queueOpenClawSend(uploadResult.id, selectedOpenClawProfileId, uploadResult.title || title);
                } else if (sendToOpenClawAfterTranscription && !selectedOpenClawProfileId) {
                    toast({
                        title: "OpenClaw Profile Required",
                        description: "Enable auto-send requires selecting an OpenClaw profile first.",
                    });
                }
            } catch (error) {
                if (isOnDashboard) {
                    setUploadProgress([
                        {
                            fileName: `${title} (${files.length} tracks)`,
                            status: "error",
                            error:
                                error instanceof Error ? error.message : "Upload failed",
                        },
                    ]);
                } else {
                    toast({
                        title: "Upload Failed",
                        description: `Failed to upload ${title}`,
                    });
                }
            } finally {
                setIsUploading(false);
            }
        },
        [
            isOnDashboard,
            uploadMultiTrack,
            toast,
            sendToOpenClawAfterTranscription,
            selectedOpenClawProfileId,
            queueOpenClawSend,
        ]
    );

    const openMultiTrackDialog = useCallback(() => {
        setMultiTrackPreview(null);
        setIsMultiTrackDialogOpen(true);
    }, []);

    const handleRecordingComplete = useCallback(
        async (blob: Blob, title: string, source?: string) => {
            const file = new File([blob], `${title}.webm`, { type: blob.type });
            await handleFileSelect(file, source);
        },
        [handleFileSelect]
    );

    const handleMultiTrackDialogClose = useCallback(() => {
        setIsMultiTrackDialogOpen(false);
        setMultiTrackPreview(null);
    }, []);

    const handleMultiTrackConfirm = useCallback(
        async (files: File[], aupFile: File, title: string) => {
            await handleMultiTrackUpload(files, aupFile, title);
            handleMultiTrackDialogClose();
        },
        [handleMultiTrackUpload, handleMultiTrackDialogClose]
    );


    const value: GlobalUploadContextValue = {
        handleFileSelect,
        handleMultiTrackUpload,
        openMultiTrackDialog,
        handleRecordingComplete,
        isUploading,
        uploadProgress,
        isOnDashboard,
        sendToOpenClawAfterTranscription,
        setSendToOpenClawAfterTranscription,
        selectedOpenClawProfileId,
        setSelectedOpenClawProfileId,
        openClawProfiles,
        openClawProfilesLoading,
        openClawProfilesError,
        refreshOpenClawProfiles,
    };

    return (
        <GlobalUploadContext.Provider value={value}>
            {children}

            {/* Multi-track Upload Dialog (global) */}
            <MultiTrackUploadDialog
                open={isMultiTrackDialogOpen}
                onOpenChange={handleMultiTrackDialogClose}
                onMultiTrackUpload={handleMultiTrackConfirm}
                prePopulatedFiles={multiTrackPreview?.audioFiles}
                prePopulatedAupFile={multiTrackPreview?.aupFile}
                prePopulatedTitle={multiTrackPreview?.title}
            />
        </GlobalUploadContext.Provider>
    );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useGlobalUpload() {
    const ctx = useContext(GlobalUploadContext);
    if (!ctx) {
        throw new Error(
            "useGlobalUpload must be used within GlobalUploadProvider"
        );
    }
    return ctx;
}
