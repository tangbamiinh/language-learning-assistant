import type {
  ChatRequest,
  ChatResponse,
  VoiceRequest,
  VoiceResponse,
  HealthStatus,
  CognateResult,
} from './types';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3000/api';

/**
 * Send a chat message to the backend via the Next.js API route.
 */
export async function sendChat(
  message: string,
  options?: { language?: string; session_id?: string }
): Promise<ChatResponse> {
  const body: ChatRequest = {
    message,
    language: options?.language,
    session_id: options?.session_id,
  };

  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-store',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `Chat API returned ${response.status}`);
  }

  return response.json();
}

/**
 * Send audio for voice processing (STT → LLM → TTS) via the Next.js API route.
 */
export async function sendVoice(
  audioBase64: string,
  options?: { language?: string; session_id?: string }
): Promise<VoiceResponse> {
  const body: VoiceRequest = {
    audio_base64: audioBase64,
    language: options?.language,
    session_id: options?.session_id,
  };

  const response = await fetch(`${API_BASE}/voice`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    cache: 'no-store',
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `Voice API returned ${response.status}`);
  }

  return response.json();
}

/**
 * Check the health of the Python backend services.
 */
export async function checkHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/health`, {
    method: 'GET',
    cache: 'no-store',
  });

  if (!response.ok) {
    throw new Error(`Health check failed: ${response.status}`);
  }

  return response.json();
}

/**
 * Look up a Sino-Vietnamese cognate by word.
 */
export async function lookupCognate(word: string): Promise<CognateResult> {
  const response = await fetch(
    `${API_BASE}/cognates?word=${encodeURIComponent(word)}`,
    {
      method: 'GET',
      cache: 'no-store',
    }
  );

  if (!response.ok) {
    throw new Error(`Cognate lookup failed: ${response.status}`);
  }

  return response.json();
}
