import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Navigation */}
      <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-violet-100">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center gap-2">
              <span className="text-xl font-bold gradient-text">Language Learning Assistant</span>
            </Link>
            <div className="flex items-center gap-3">
              <Link
                href="/chat"
                className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-violet-500 to-indigo-600 rounded-lg shadow-sm hover:shadow-md transition-all"
              >
                Start Chat
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-8 py-16 sm:py-24">
        <div className="max-w-4xl mx-auto text-center">
          {/* Flag badges */}
          <div className="flex items-center justify-center gap-4 mb-8">
            <span className="text-5xl animate-float" style={{ animationDelay: "0ms" }}>🇻🇳</span>
            <svg className="w-8 h-8 text-violet-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
            <span className="text-5xl animate-float" style={{ animationDelay: "150ms" }}>🇨🇳</span>
          </div>

          {/* Main heading */}
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight mb-6">
            <span className="gradient-text">Language Learning</span>
            <br />
            <span className="text-gray-800 dark:text-gray-100">Assistant</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto mb-4 leading-relaxed">
            Learn Mandarin Chinese through the{" "}
            <span className="font-semibold text-violet-600 dark:text-violet-400">Hán Việt Bridge</span> —
            leveraging the ~60% Sino-Vietnamese vocabulary overlap between Vietnamese and Chinese.
          </p>
          <p className="text-base text-gray-500 dark:text-gray-400 max-w-xl mx-auto mb-10">
            AI-powered speech-to-speech conversations that help Vietnamese speakers master Mandarin
            faster by recognizing familiar cognates and building on what you already know.
          </p>

          {/* Quick action buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
            <Link
              href="/chat"
              className="w-full sm:w-auto px-8 py-4 text-base font-semibold text-white bg-gradient-to-r from-violet-500 to-indigo-600 rounded-xl shadow-lg hover:shadow-xl hover:scale-105 transition-all flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              Start Chat
            </Link>
            <Link
              href="/chat"
              className="w-full sm:w-auto px-8 py-4 text-base font-semibold text-violet-700 dark:text-violet-300 bg-white dark:bg-gray-800 border-2 border-violet-200 dark:border-violet-700 rounded-xl hover:border-violet-400 hover:shadow-md transition-all flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 10v2a7 7 0 01-14 0v-2" />
                <line x1="12" y1="19" x2="12" y2="23" />
                <line x1="8" y1="23" x2="16" y2="23" />
              </svg>
              Voice Practice
            </Link>
            <button
              className="w-full sm:w-auto px-8 py-4 text-base font-semibold text-indigo-700 dark:text-indigo-300 bg-white dark:bg-gray-800 border-2 border-indigo-200 dark:border-indigo-700 rounded-xl hover:border-indigo-400 hover:shadow-md transition-all flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              Learn Cognates
            </button>
          </div>

          {/* How it works cards */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 max-w-3xl mx-auto">
            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-violet-100 dark:border-gray-700 text-left hover:shadow-md transition-shadow">
              <div className="w-10 h-10 rounded-xl bg-violet-100 dark:bg-violet-900/30 flex items-center justify-center mb-4">
                <span className="text-xl">🗣️</span>
              </div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-2">
                Speak Naturally
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Talk in Vietnamese or Chinese. Our AI understands both languages and adapts to your level.
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-violet-100 dark:border-gray-700 text-left hover:shadow-md transition-shadow">
              <div className="w-10 h-10 rounded-xl bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center mb-4">
                <span className="text-xl">🔗</span>
              </div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-2">
                Hán Việt Bridge
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Discover cognates like 學生 (học sinh → xuéshēng) and learn Chinese faster with familiar words.
              </p>
            </div>

            <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-sm border border-violet-100 dark:border-gray-700 text-left hover:shadow-md transition-shadow">
              <div className="w-10 h-10 rounded-xl bg-violet-100 dark:bg-violet-900/30 flex items-center justify-center mb-4">
                <span className="text-xl">📈</span>
              </div>
              <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-2">
                Track Progress
              </h3>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                AI remembers your weak areas and builds personalized lessons to accelerate your learning.
              </p>
            </div>
          </div>

          {/* Example cognates */}
          <div className="mt-16 bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-2xl border border-violet-100 dark:border-gray-700 p-6 max-w-2xl mx-auto">
            <h3 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-4">
              Example: Sino-Vietnamese Cognates
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                { vn: "Học sinh", cn: "学生", py: "xuéshēng", en: "Student" },
                { vn: "Giáo viên", cn: "教師", py: "jiàoshī", en: "Teacher" },
                { vn: "Trường học", cn: "学校", py: "xuéxiào", en: "School" },
                { vn: "Quốc gia", cn: "国家", py: "guójiā", en: "Country" },
              ].map((item) => (
                <div key={item.vn} className="text-center p-3 rounded-xl bg-violet-50 dark:bg-violet-900/20">
                  <p className="text-sm font-medium text-violet-700 dark:text-violet-300">{item.cn}</p>
                  <p className="text-xs text-gray-500 dark:text-gray-400 italic">{item.py}</p>
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">{item.vn}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-violet-100 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm py-6">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <p className="text-sm text-gray-400">
            Built with Qwen3 AI · Next.js 16 · Powered by the Hán Việt bridge
          </p>
        </div>
      </footer>
    </div>
  );
}
