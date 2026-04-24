'use client';

import { useRef, useCallback, useEffect, useState } from 'react';
import { useVoiceRecorder } from '@/lib/hooks';

export function VoiceRecorder({
  onRecordingComplete,
  disabled,
}: {
  onRecordingComplete?: (audioBase64: string) => Promise<void>;
  disabled?: boolean;
}) {
  const {
    isRecording,
    duration,
    audioUrl,
    error,
    startRecording,
    stopRecording,
    getAudioBase64,
    reset,
    formatDuration,
  } = useVoiceRecorder();

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const [showResult, setShowResult] = useState(false);

  // Waveform visualization
  useEffect(() => {
    if (!isRecording) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Setup audio analysis
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext({ sampleRate: 16000 });
    }

    const drawWaveform = () => {
      if (!canvas || !ctx) return;

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      // Generate animated waveform
      const bars = 40;
      const barWidth = width / bars;

      for (let i = 0; i < bars; i++) {
        const amplitude = Math.random() * height * 0.6 + height * 0.1;
        const barHeight = amplitude * (Math.sin(Date.now() / 200 + i * 0.5) * 0.5 + 0.5);
        const x = i * barWidth + barWidth / 2;
        const y = height / 2;

        const gradient = ctx.createLinearGradient(x, y - barHeight / 2, x, y + barHeight / 2);
        gradient.addColorStop(0, '#8b5cf6');
        gradient.addColorStop(0.5, '#6366f1');
        gradient.addColorStop(1, '#8b5cf6');

        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.roundRect(x - barWidth * 0.3, y - barHeight / 2, barWidth * 0.6, barHeight, 2);
        ctx.fill();
      }

      animationRef.current = requestAnimationFrame(drawWaveform);
    };

    drawWaveform();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isRecording]);

  const handleToggleRecording = useCallback(async () => {
    if (isRecording) {
      stopRecording();
      setShowResult(true);
    } else {
      setShowResult(false);
      await startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const handleSendRecording = useCallback(async () => {
    if (disabled) return;
    try {
      const base64 = await getAudioBase64();
      if (onRecordingComplete) {
        await onRecordingComplete(base64);
      }
      reset();
      setShowResult(false);
    } catch (err) {
      console.error('Failed to process recording:', err);
    }
  }, [getAudioBase64, onRecordingComplete, reset, disabled]);

  const handleDiscard = useCallback(() => {
    reset();
    setShowResult(false);
  }, [reset]);

  return (
    <div className="flex flex-col items-center gap-3">
      {/* Waveform canvas */}
      <div className="w-full h-20 bg-gray-50 rounded-xl border border-gray-100 overflow-hidden">
        <canvas
          ref={canvasRef}
          width={400}
          height={80}
          className="w-full h-full"
        />
        {!isRecording && !showResult && (
          <div className="absolute inset-0 flex items-center justify-center text-gray-400 text-sm">
            Press the microphone to start recording
          </div>
        )}
      </div>

      {/* Timer */}
      {isRecording && (
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
          <span className="text-sm font-mono text-red-500 font-medium">
            {formatDuration(duration)}
          </span>
        </div>
      )}

      {/* Recording controls */}
      <div className="flex items-center gap-3">
        {showResult && audioUrl ? (
          <>
            {/* Playback */}
            <audio controls src={audioUrl} className="h-8 max-w-[200px]" />
            {/* Send button */}
            <button
              onClick={handleSendRecording}
              disabled={disabled}
              className="flex items-center gap-1.5 px-4 py-2 bg-gradient-to-r from-violet-500 to-indigo-600 text-white text-sm font-medium rounded-lg shadow-md hover:shadow-lg transition-all disabled:opacity-40"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
              Send
            </button>
            {/* Discard button */}
            <button
              onClick={handleDiscard}
              className="flex items-center gap-1.5 px-4 py-2 bg-gray-100 text-gray-600 text-sm font-medium rounded-lg hover:bg-gray-200 transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Discard
            </button>
          </>
        ) : (
          <button
            onClick={handleToggleRecording}
            disabled={disabled}
            className={`w-14 h-14 rounded-full flex items-center justify-center shadow-lg transition-all ${
              isRecording
                ? 'bg-red-500 hover:bg-red-600 animate-pulse'
                : 'bg-gradient-to-r from-violet-500 to-indigo-600 hover:shadow-xl hover:scale-105'
            } text-white disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100`}
          >
            {isRecording ? (
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="2" />
              </svg>
            ) : (
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 10v2a7 7 0 01-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
            )}
          </button>
        )}
      </div>

      {/* Error message */}
      {error && (
        <p className="text-sm text-red-500 text-center max-w-xs">{error}</p>
      )}

      {/* Status text */}
      {!isRecording && !showResult && !error && (
        <p className="text-xs text-gray-400">
          Tap to speak in Vietnamese, English, or Chinese
        </p>
      )}
    </div>
  );
}
