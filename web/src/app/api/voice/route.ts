import { NextRequest, NextResponse } from 'next/server';

interface VoiceRequest {
  audio_base64: string;
  language?: string;
  session_id?: string;
}

interface VoiceApiResponse {
  transcript: string;
  reply: string;
  audio_base64: string;
  language: string;
}

const BACKEND_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

/**
 * Generate a mock voice response when the backend is unavailable.
 * The UI can still display a transcript and reply.
 */
function generateMockVoiceResponse(language: string): VoiceApiResponse {
  return {
    transcript: '(Audio received — connect Python backend for STT transcription)',
    reply: language === 'chinese'
      ? '我听到了你的语音！连接 Python 后端以启用完整的语音识别和合成功能。'
      : 'Tôi nghe thấy giọng nói của bạn! Kết nối Python backend để kích hoạt chức năng nhận diện và tổng hợp giọng nói.',
    audio_base64: '',
    language: language || 'chinese',
  };
}

export async function POST(request: NextRequest) {
  // Prevent Next.js from caching API responses
  NextResponse.next();

  try {
    const body: VoiceRequest = await request.json();

    if (!body.audio_base64) {
      return NextResponse.json(
        { error: 'Audio data (audio_base64) is required' },
        {
          status: 400,
          headers: {
            'Cache-Control': 'no-store',
          },
        }
      );
    }

    const language = body.language || 'chinese';

    // Try to forward to Python backend
    try {
      const response = await fetch(`${BACKEND_URL}/api/voice`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio_base64: body.audio_base64,
          language,
          session_id: body.session_id,
        }),
        // Voice processing can take longer - 60s timeout
        signal: AbortSignal.timeout(60_000),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Python API error:', response.status, errorText);
        throw new Error(`Backend returned ${response.status}`);
      }

      const data: VoiceApiResponse = await response.json();
      return NextResponse.json(data, {
        headers: {
          'Cache-Control': 'no-store, must-revalidate',
        },
      });
    } catch (backendError) {
      console.warn('Backend unavailable, returning mock voice response:', backendError);
      const mockData = generateMockVoiceResponse(language);
      return NextResponse.json(mockData, {
        headers: {
          'Cache-Control': 'no-store, must-revalidate',
        },
      });
    }
  } catch (error) {
    console.error('Voice route error:', error);
    return NextResponse.json(
      { error: 'Failed to process voice request' },
      {
        status: 500,
        headers: {
          'Cache-Control': 'no-store',
        },
      }
    );
  }
}
