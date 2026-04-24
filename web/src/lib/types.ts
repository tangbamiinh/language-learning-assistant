export interface Message {
  id: string;
  role: 'user' | 'agent';
  text: string;
  pinyin?: string;
  vietnamese?: string;
  chinese?: string;
  timestamp: Date;
  audioUrl?: string;
}

export interface ChatRequest {
  message: string;
  language?: string;
  session_id?: string;
}

export interface ChatResponse {
  reply: string;
  language: string;
  pinyin?: string;
  vietnamese?: string;
  chinese?: string;
}

export interface VoiceRequest {
  audio_base64: string;
  language?: string;
  session_id?: string;
}

export interface VoiceResponse {
  transcript: string;
  reply: string;
  audio_base64: string;
  language: string;
}

export interface HealthStatus {
  status: string;
  stt_available: boolean;
  tts_available: boolean;
}

export interface Cognate {
  vietnamese: string;
  chinese_traditional: string;
  chinese_simplified: string;
  pinyin: string;
  english: string;
  hsk_level: number | null;
  category: string;
}

export interface CognateResult {
  found: boolean;
  cognate?: Cognate;
  message?: string;
}
