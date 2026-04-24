'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { Message, ChatResponse, VoiceResponse } from './types';
import { generateId } from './utils';

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addMessage = useCallback((role: 'user' | 'agent', text: string, extras?: Partial<Message>) => {
    const message: Message = {
      id: generateId(),
      role,
      text,
      timestamp: new Date(),
      ...extras,
    };
    setMessages((prev) => [...prev, message]);
    return message;
  }, []);

  const sendMessage = useCallback(async (text: string, language?: string) => {
    if (!text.trim() || isLoading) return;

    addMessage('user', text.trim());
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text.trim(), language }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      addMessage('agent', data.reply, {
        pinyin: data.pinyin,
        vietnamese: data.vietnamese,
        chinese: data.chinese,
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Something went wrong';
      setError(errorMessage);
      console.error('Chat error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading, addMessage]);

  const sendVoiceMessage = useCallback(async (audioBase64: string, language?: string) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_base64: audioBase64, language }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data: VoiceResponse = await response.json();

      // Add user transcript
      if (data.transcript) {
        addMessage('user', data.transcript);
      }

      // Add agent reply with audio
      const agentMsg = addMessage('agent', data.reply, {
        audioUrl: `data:audio/wav;base64,${data.audio_base64}`,
      });

      return agentMsg;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Voice processing failed';
      setError(errorMessage);
      console.error('Voice error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [addMessage]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    sendVoiceMessage,
    clearMessages,
    addMessage,
  };
}

export function useVoiceRecorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopTimer();
      stopStream();
      revokeAudioUrl();
    };
  }, []);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const stopStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }, []);

  const revokeAudioUrl = useCallback(() => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
  }, [audioUrl]);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });

      streamRef.current = stream;

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      });

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        revokeAudioUrl();
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setDuration(0);

      timerRef.current = setInterval(() => {
        setDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      const message =
        err instanceof DOMException && err.name === 'NotAllowedError'
          ? 'Microphone access denied. Please allow microphone access and try again.'
          : 'Failed to start recording. Please check your microphone.';
      setError(message);
      console.error('Recording error:', err);
    }
  }, [revokeAudioUrl]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
    stopTimer();
    stopStream();
    setIsRecording(false);
  }, [stopTimer, stopStream]);

  const getAudioBase64 = useCallback((): Promise<string> => {
    return new Promise((resolve, reject) => {
      if (!audioBlob) {
        reject(new Error('No audio available'));
        return;
      }

      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        // Remove the data URL prefix
        const base64Data = base64.split(',')[1];
        resolve(base64Data);
      };
      reader.onerror = () => reject(new Error('Failed to read audio blob'));
      reader.readAsDataURL(audioBlob);
    });
  }, [audioBlob]);

  const reset = useCallback(() => {
    setAudioBlob(null);
    revokeAudioUrl();
    setDuration(0);
    setError(null);
  }, [revokeAudioUrl]);

  const formatDuration = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }, []);

  return {
    isRecording,
    duration,
    audioBlob,
    audioUrl,
    error,
    startRecording,
    stopRecording,
    getAudioBase64,
    reset,
    formatDuration,
  };
}
