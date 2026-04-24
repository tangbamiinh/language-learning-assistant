'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { Message, detectLanguage } from '@/lib';
import { useChat } from '@/lib/hooks';
import { LiveKitVoiceRoom } from '@/components/LiveKitVoiceRoom';
import { formatTimestamp } from '@/lib/utils';

function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  const lang = detectLanguage(message.text);

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[80%] rounded-2xl px-4 py-3 shadow-sm ${
          isUser
            ? 'bg-indigo-600 text-white rounded-br-md'
            : 'bg-white text-gray-800 rounded-bl-md border border-gray-100'
        }`}
      >
        <p className="text-base leading-relaxed whitespace-pre-wrap break-words">
          {message.text}
        </p>

        {/* Show pinyin for Chinese text */}
        {message.pinyin && (
          <p className={`text-sm mt-1 italic ${isUser ? 'text-indigo-200' : 'text-gray-500'}`}>
            {message.pinyin}
          </p>
        )}

        {/* Show Vietnamese translation */}
        {message.vietnamese && (
          <p className={`text-sm mt-1 ${isUser ? 'text-indigo-200' : 'text-violet-600'}`}>
            {message.vietnamese}
          </p>
        )}

        {/* Show Chinese characters */}
        {message.chinese && (
          <p className={`text-sm mt-1 font-medium ${isUser ? 'text-indigo-100' : 'text-gray-600'}`}>
            {message.chinese}
          </p>
        )}

        {/* Audio playback for agent messages */}
        {message.audioUrl && (
          <div className="mt-2 flex items-center gap-2">
            <audio controls src={message.audioUrl} className="h-8 max-w-full">
              Your browser does not support audio playback.
            </audio>
          </div>
        )}

        <p className={`text-xs mt-2 ${isUser ? 'text-indigo-200' : 'text-gray-400'}`}>
          {formatTimestamp(message.timestamp)}
        </p>
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex justify-start mb-4">
      <div className="bg-white border border-gray-100 rounded-2xl rounded-bl-md px-4 py-3 shadow-sm">
        <div className="flex items-center gap-1.5">
          <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
          <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
          <span className="w-2 h-2 bg-violet-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
}

export function ChatUI() {
  const { messages, isLoading, error, sendMessage, clearMessages } = useChat();
  const [input, setInput] = useState('');
  const [mode, setMode] = useState<'text' | 'voice'>('text');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, scrollToBottom]);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isLoading) return;

    await sendMessage(input.trim());
    setInput('');
    inputRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleVoiceDisconnect = () => {
    setMode('text');
  };

  return (
    <div className="flex flex-col h-full bg-gradient-to-br from-violet-50 via-white to-indigo-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-white border-b border-gray-100 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-white font-bold text-sm shadow-md">
            AI
          </div>
          <div>
            <h2 className="font-semibold text-gray-800 text-sm">Chinese Tutor</h2>
            <p className="text-xs text-gray-500">
              {mode === 'voice' ? (
                <span className="text-violet-500">🎙️ Voice Call</span>
              ) : isLoading ? (
                <span className="text-violet-500">Thinking...</span>
              ) : (
                'Online · Hán Việt Bridge'
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Mode toggle */}
          <div className="flex bg-gray-100 rounded-lg p-0.5">
            <button
              onClick={() => setMode('text')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                mode === 'text'
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
              disabled={mode === 'voice'}
            >
              💬 Text
            </button>
            <button
              onClick={() => setMode('voice')}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                mode === 'voice'
                  ? 'bg-white text-indigo-600 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              🎙️ Voice
            </button>
          </div>

          {/* Clear button */}
          <button
            onClick={clearMessages}
            className="p-2 text-gray-400 hover:text-red-500 transition-colors rounded-lg hover:bg-red-50"
            title="Clear chat"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && mode === 'text' ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-3xl mb-4 shadow-lg">
              🎓
            </div>
            <h3 className="text-lg font-semibold text-gray-800 mb-2">
              Welcome to Chinese Learning
            </h3>
            <p className="text-sm text-gray-500 max-w-sm leading-relaxed">
              Start a conversation in Vietnamese, English, or Chinese. I&apos;ll help you learn Mandarin using the
              Hán Việt bridge — leveraging the 60% vocabulary overlap between Vietnamese and Chinese.
            </p>
            <div className="flex flex-wrap gap-2 mt-6 justify-center">
              {[
                { text: 'Hello! Let me learn Chinese', emoji: '👋' },
                { text: 'Giúp tôi học từ vựng Hán Việt', emoji: '📚' },
                { text: 'How do I say "thank you" in Chinese?', emoji: '🙏' },
              ].map((suggestion) => (
                <button
                  key={suggestion.text}
                  onClick={() => {
                    setInput(suggestion.text);
                    inputRef.current?.focus();
                  }}
                  className="flex items-center gap-1.5 px-4 py-2 text-sm bg-white border border-gray-200 rounded-full text-gray-600 hover:border-violet-300 hover:text-violet-600 hover:shadow-sm transition-all"
                >
                  <span>{suggestion.emoji}</span>
                  {suggestion.text}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg) => (
              <MessageBubble key={msg.id} message={msg} />
            ))}
            {isLoading && mode === 'text' && <TypingIndicator />}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Error display */}
      {error && (
        <div className="mx-4 mb-2 px-4 py-2 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">
          {error}
        </div>
      )}

      {/* Input area */}
      <div className="px-4 pb-4 pt-2 bg-white border-t border-gray-100">
        {mode === 'text' ? (
          <form onSubmit={handleSubmit} className="flex items-end gap-2">
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type in Vietnamese, English, or Chinese..."
                rows={1}
                disabled={isLoading}
                className="w-full resize-none rounded-xl border border-gray-200 bg-gray-50 px-4 py-3 pr-12 text-sm text-gray-800 placeholder-gray-400 focus:border-violet-400 focus:ring-2 focus:ring-violet-100 focus:bg-white focus:outline-none transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ minHeight: '44px', maxHeight: '120px' }}
              />
            </div>
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="flex-shrink-0 w-11 h-11 rounded-xl bg-gradient-to-r from-violet-500 to-indigo-600 text-white flex items-center justify-center shadow-md hover:shadow-lg transition-all disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-md"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </form>
        ) : (
          <LiveKitVoiceRoom onDisconnect={handleVoiceDisconnect} />
        )}
      </div>
    </div>
  );
}
