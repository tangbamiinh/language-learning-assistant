import { NextRequest, NextResponse } from 'next/server';

interface ChatRequest {
  message: string;
  language?: string;
  session_id?: string;
}

interface ChatApiResponse {
  reply: string;
  language: string;
  pinyin?: string;
  vietnamese?: string;
  chinese?: string;
}

const BACKEND_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

/**
 * Mock tutor responses for when the backend is unavailable.
 * Keeps the UI functional during development.
 */
function generateMockReply(message: string, language: string): ChatApiResponse {
  const lowerMsg = message.toLowerCase().trim();

  // Greetings
  if (/^(hi|hello|你好|xin chao|chào)/i.test(lowerMsg)) {
    return {
      reply: language === 'chinese' ? '你好！我是你的汉语老师。让我们开始学习吧！' : 'Xin chào! Tôi là giáo viên Hán ngữ của bạn. Chúng ta bắt đầu học nhé!',
      language: language || 'chinese',
      pinyin: 'Nǐ hǎo! Wǒ shì nǐ de hànyǔ lǎoshī. Ràng wǒmen kāishǐ xuéxí ba!',
      chinese: '你好！我是你的汉语老师。让我们开始学习吧！',
      vietnamese: 'Xin chào! Tôi là giáo viên Hán ngữ của bạn.',
    };
  }

  // Thank you
  if (/^(thank|thanks|谢谢|cảm ơn|cảmơn)/i.test(lowerMsg)) {
    return {
      reply: language === 'chinese' ? '不客气！继续加油！' : 'Không vấn đề gì! Tiếp tục cố gắng nhé!',
      language: language || 'chinese',
      pinyin: 'Bú kèqì! Jìxù jiāyóu!',
      chinese: '不客气！继续加油！',
      vietnamese: 'Không vấn đề gì! Tiếp tục cố gắng nhé!',
    };
  }

  // Default helpful response
  return {
    reply: language === 'chinese'
      ? '这是一个很好的问题。在汉语中，很多词和越南语有相似之处，因为我们叫做"汉越词"。你可以尝试说说你想学什么话题？比如：问候、数字、家庭、食物。'
      : 'Đây là một câu hỏi tốt! Trong tiếng Trung, nhiều từ có điểm chung với tiếng Việt vì chúng là từ Hán Việt. Bạn muốn học chủ đề gì? Ví dụ: chào hỏi, số đếm, gia đình, hay đồ ăn?',
    language: language || 'chinese',
    pinyin: 'Zhè shì yí gè hěn hǎo de wèntí.',
    chinese: '这是一个很好的问题。',
    vietnamese: 'Đây là một câu hỏi tốt!',
  };
}

export async function POST(request: NextRequest) {
  // Prevent Next.js from caching API responses
  NextResponse.next();

  try {
    const body: ChatRequest = await request.json();

    if (!body.message || !body.message.trim()) {
      return NextResponse.json(
        { error: 'Message is required' },
        { status: 400 }
      );
    }

    const language = body.language || 'chinese';

    // Try to forward to Python backend
    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: body.message.trim(),
          language,
          session_id: body.session_id,
        }),
        // Don't wait forever - 30s timeout
        signal: AbortSignal.timeout(30_000),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Python API error:', response.status, errorText);
        throw new Error(`Backend returned ${response.status}`);
      }

      const data: ChatApiResponse = await response.json();
      return NextResponse.json(data, {
        headers: {
          'Cache-Control': 'no-store, must-revalidate',
        },
      });
    } catch (backendError) {
      console.warn('Backend unavailable, returning mock response:', backendError);
      const mockData = generateMockReply(body.message, language);
      return NextResponse.json(mockData, {
        headers: {
          'Cache-Control': 'no-store, must-revalidate',
        },
      });
    }
  } catch (error) {
    console.error('Chat route error:', error);
    return NextResponse.json(
      { error: 'Failed to process chat request' },
      {
        status: 500,
        headers: {
          'Cache-Control': 'no-store',
        },
      }
    );
  }
}
