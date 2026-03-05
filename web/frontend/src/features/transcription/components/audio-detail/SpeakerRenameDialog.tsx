import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent } from '@/components/ui/card';
import { Loader2, Users, Save, X, Play, Pause } from 'lucide-react';
import { useAuth } from "@/features/auth/hooks/useAuth";

interface SpeakerMapping {
  id?: number;
  original_speaker: string;
  custom_name: string;
}

interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  speaker?: string;
}

interface SpeakerRenameDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  transcriptionId: string;
  onSpeakerMappingsUpdate: (mappings: SpeakerMapping[]) => void;
  initialSpeakers?: string[];
  transcriptSegments?: TranscriptSegment[];
}

const SpeakerRenameDialog: React.FC<SpeakerRenameDialogProps> = ({
  open,
  onOpenChange,
  transcriptionId,
  onSpeakerMappingsUpdate,
  initialSpeakers = [],
  transcriptSegments = [],
}) => {
  const { getAuthHeaders } = useAuth();
  const [speakerMappings, setSpeakerMappings] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resolvedTranscriptSegments, setResolvedTranscriptSegments] = useState<TranscriptSegment[]>([]);
  const [isLoadingTranscriptSegments, setIsLoadingTranscriptSegments] = useState(false);

  const [previewAudioUrl, setPreviewAudioUrl] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [isPreviewLoading, setIsPreviewLoading] = useState(false);
  const [isPreviewReady, setIsPreviewReady] = useState(false);
  const [activePreviewSpeaker, setActivePreviewSpeaker] = useState<string | null>(null);
  const previewAudioRef = useRef<HTMLAudioElement | null>(null);
  const previewStopAtRef = useRef<number | null>(null);

  const speakerSamples = useMemo(() => {
    const samples: Record<string, TranscriptSegment> = {};
    resolvedTranscriptSegments.forEach((segment) => {
      if (!segment.speaker || samples[segment.speaker]) {
        return;
      }
      if (typeof segment.start !== 'number' || typeof segment.end !== 'number' || segment.end <= segment.start) {
        return;
      }
      samples[segment.speaker] = segment;
    });
    return samples;
  }, [resolvedTranscriptSegments]);

  const stopPreview = useCallback(() => {
    const audio = previewAudioRef.current;
    if (audio) {
      audio.pause();
    }
    previewStopAtRef.current = null;
    setActivePreviewSpeaker(null);
  }, []);

  const fetchSpeakerMappings = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/transcription/' + transcriptionId + '/speakers', {
        headers: { ...getAuthHeaders() },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch speaker mappings: ' + response.statusText);
      }

      const existingMappings: SpeakerMapping[] = await response.json();

      const mappingObj: Record<string, string> = {};
      existingMappings.forEach(mapping => {
        mappingObj[mapping.original_speaker] = mapping.custom_name;
      });

      initialSpeakers.forEach(speaker => {
        if (!mappingObj[speaker]) {
          mappingObj[speaker] = speaker;
        }
      });

      setSpeakerMappings(mappingObj);
    } catch (err) {
      console.error('Error fetching speaker mappings:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch speaker mappings');

      const defaultMappings: Record<string, string> = {};
      initialSpeakers.forEach(speaker => {
        defaultMappings[speaker] = speaker;
      });
      setSpeakerMappings(defaultMappings);
    } finally {
      setIsLoading(false);
    }
  }, [transcriptionId, getAuthHeaders, initialSpeakers]);

  const fetchPreviewAudio = useCallback(async () => {
    if (!open || !transcriptionId) {
      return;
    }

    setIsPreviewLoading(true);
    setIsPreviewReady(false);
    setPreviewError(null);
    try {
      const response = await fetch('/api/v1/transcription/' + transcriptionId + '/audio', {
        headers: { ...getAuthHeaders() },
      });
      if (!response.ok) {
        throw new Error('Audio load failed: ' + response.status);
      }

      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);

      setPreviewAudioUrl((prev) => {
        if (prev) {
          URL.revokeObjectURL(prev);
        }
        return objectUrl;
      });
    } catch (err) {
      console.error('Error loading preview audio:', err);
      setPreviewError('Unable to load speaker preview audio.');
      setIsPreviewReady(false);
      setPreviewAudioUrl((prev) => {
        if (prev) {
          URL.revokeObjectURL(prev);
        }
        return null;
      });
    } finally {
      setIsPreviewLoading(false);
    }
  }, [open, transcriptionId, getAuthHeaders]);

  const fetchTranscriptSegments = useCallback(async () => {
    if (!open || !transcriptionId) {
      return;
    }

    const hasIncomingSpeakerSegments = transcriptSegments.some((segment) => !!segment.speaker);
    if (hasIncomingSpeakerSegments) {
      setResolvedTranscriptSegments(transcriptSegments);
      return;
    }

    setIsLoadingTranscriptSegments(true);
    try {
      const response = await fetch('/api/v1/transcription/' + transcriptionId + '/transcript', {
        headers: { ...getAuthHeaders() },
      });
      if (!response.ok) {
        throw new Error('Failed to fetch transcript: ' + response.statusText);
      }

      const data = await response.json();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const candidateSegments: any[] = data?.transcript?.segments || [];
      if (Array.isArray(candidateSegments)) {
        const normalized = candidateSegments
          .filter((segment) =>
            segment &&
            typeof segment.start === 'number' &&
            typeof segment.end === 'number' &&
            typeof segment.text === 'string'
          )
          .map((segment) => ({
            start: segment.start,
            end: segment.end,
            text: segment.text,
            speaker: segment.speaker,
          }));
        setResolvedTranscriptSegments(normalized);
      } else {
        setResolvedTranscriptSegments([]);
      }
    } catch (err) {
      console.error('Error loading transcript segments for speaker preview:', err);
      setResolvedTranscriptSegments([]);
    } finally {
      setIsLoadingTranscriptSegments(false);
    }
  }, [open, transcriptionId, transcriptSegments, getAuthHeaders]);

  useEffect(() => {
    if (open && transcriptionId) {
      fetchSpeakerMappings();
      fetchPreviewAudio();
      fetchTranscriptSegments();
    }
  }, [open, transcriptionId, fetchSpeakerMappings, fetchPreviewAudio, fetchTranscriptSegments]);

  useEffect(() => {
    return () => {
      if (previewAudioUrl) {
        URL.revokeObjectURL(previewAudioUrl);
      }
    };
  }, [previewAudioUrl]);

  useEffect(() => {
    if (!open) {
      stopPreview();
    }
  }, [open, stopPreview]);

  useEffect(() => {
    if (!open) {
      setIsPreviewReady(false);
      return;
    }

    const hasIncomingSpeakerSegments = transcriptSegments.some((segment) => !!segment.speaker);
    if (hasIncomingSpeakerSegments) {
      setResolvedTranscriptSegments(transcriptSegments);
    }
  }, [open, transcriptSegments]);

  const handleSpeakerNameChange = (originalSpeaker: string, customName: string) => {
    setSpeakerMappings(prev => ({
      ...prev,
      [originalSpeaker]: customName,
    }));
  };

  const handlePreviewTimeUpdate = () => {
    if (!activePreviewSpeaker || previewStopAtRef.current == null || !previewAudioRef.current) {
      return;
    }

    if (previewAudioRef.current.currentTime >= previewStopAtRef.current) {
      stopPreview();
    }
  };

  const toggleSpeakerPreview = async (speaker: string) => {
    if (!previewAudioUrl || !previewAudioRef.current) {
      return;
    }
    if (!isPreviewReady) {
      setPreviewError('Preview audio is still loading. Please try again.');
      return;
    }

    const sample = speakerSamples[speaker];
    if (!sample) {
      setPreviewError('No sample segment found for ' + speaker + '.');
      return;
    }

    if (activePreviewSpeaker === speaker && !previewAudioRef.current.paused) {
      stopPreview();
      return;
    }

    setPreviewError(null);

    try {
      const duration = Number.isFinite(previewAudioRef.current.duration)
        ? previewAudioRef.current.duration
        : undefined;
      const safeStart = duration != null
        ? Math.max(0, Math.min(sample.start, Math.max(0, duration - 0.05)))
        : Math.max(0, sample.start);
      const safeEnd = duration != null
        ? Math.max(safeStart, Math.min(sample.end, duration))
        : sample.end;

      previewStopAtRef.current = safeEnd;
      previewAudioRef.current.currentTime = safeStart;
      await previewAudioRef.current.play();
      setActivePreviewSpeaker(speaker);
    } catch (err) {
      console.error('Error playing speaker preview:', err);
      setPreviewError('Failed to play speaker preview.');
      stopPreview();
    }
  };

  const saveSpeakerMappings = async () => {
    setIsSaving(true);
    setError(null);

    try {
      const mappingsArray = Object.entries(speakerMappings).map(([original_speaker, custom_name]) => ({
        original_speaker,
        custom_name,
      }));

      const response = await fetch('/api/v1/transcription/' + transcriptionId + '/speakers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...getAuthHeaders() },
        body: JSON.stringify({
          mappings: mappingsArray,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save speaker mappings: ' + response.statusText);
      }

      const updatedMappings: SpeakerMapping[] = await response.json();
      onSpeakerMappingsUpdate(updatedMappings);
      stopPreview();
      onOpenChange(false);
    } catch (err) {
      console.error('Error saving speaker mappings:', err);
      setError(err instanceof Error ? err.message : 'Failed to save speaker mappings');
    } finally {
      setIsSaving(false);
    }
  };

  const speakers = Object.keys(speakerMappings).sort();

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Rename Speakers
          </DialogTitle>
        </DialogHeader>

        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span className="ml-2 text-sm text-muted-foreground">Loading speakers...</span>
          </div>
        ) : (
          <div className="space-y-4">
            {error && (
              <div className="p-3 rounded-md bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
                <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
              </div>
            )}

            {previewError && (
              <div className="p-3 rounded-md bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800">
                <p className="text-sm text-amber-700 dark:text-amber-300">{previewError}</p>
              </div>
            )}

            {(isPreviewLoading || isLoadingTranscriptSegments) && (
              <div className="p-3 rounded-md bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
                <p className="text-sm text-blue-700 dark:text-blue-300">
                  Loading speaker preview resources...
                </p>
              </div>
            )}

            {speakers.length === 0 ? (
              <Card>
                <CardContent className="pt-6 text-center text-muted-foreground">
                  <Users className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No speakers found with diarization enabled.</p>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-3 max-h-72 overflow-y-auto">
                {speakers.map((speaker) => {
                  const sample = speakerSamples[speaker];
                  const canPreview = !!previewAudioUrl && !!sample && isPreviewReady && !isPreviewLoading;

                  return (
                    <div
                      key={speaker}
                      className="space-y-1"
                    >
                      <Label htmlFor={'speaker-' + speaker} className="text-xs font-medium text-muted-foreground">
                        {speaker}
                      </Label>
                      <div className="flex items-center gap-2">
                        <Input
                          id={'speaker-' + speaker}
                          value={speakerMappings[speaker] || ''}
                          onChange={(e) => handleSpeakerNameChange(speaker, e.target.value)}
                          placeholder={'Enter custom name for ' + speaker}
                          className="transition-all duration-200 focus:ring-2 focus:ring-primary/20"
                        />
                        <Button
                          type="button"
                          variant="outline"
                          size="icon"
                          onClick={() => { void toggleSpeakerPreview(speaker); }}
                          disabled={!canPreview}
                          title={canPreview ? ('Preview ' + speaker) : ('No sample available for ' + speaker)}
                        >
                          {activePreviewSpeaker === speaker ? (
                            <Pause className="h-4 w-4" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                      {sample?.text && (
                        <p className="text-[11px] text-muted-foreground line-clamp-1">
                          Sample: {sample.text}
                        </p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        <DialogFooter className="gap-2">
          <Button variant="outline" onClick={() => { stopPreview(); onOpenChange(false); }} disabled={isSaving}>
            <X className="h-4 w-4 mr-1" />
            Cancel
          </Button>
          <Button
            onClick={saveSpeakerMappings}
            disabled={isSaving || speakers.length === 0}
            className="min-w-[100px]"
          >
            {isSaving ? (
              <>
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                Saving...
              </>
            ) : (
              <>
                <Save className="h-4 w-4 mr-1" />
                Save
              </>
            )}
          </Button>
        </DialogFooter>

        <audio
          ref={previewAudioRef}
          src={previewAudioUrl || undefined}
          preload="metadata"
          onLoadedMetadata={() => setIsPreviewReady(true)}
          onCanPlay={() => setIsPreviewReady(true)}
          onTimeUpdate={handlePreviewTimeUpdate}
          onEnded={stopPreview}
          className="hidden"
        />
      </DialogContent>
    </Dialog>
  );
};

export default SpeakerRenameDialog;
