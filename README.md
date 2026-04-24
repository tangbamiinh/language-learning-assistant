# Language Learning Assistant 🇻🇳 → 🇨🇳

An AI-powered Speech-to-Speech agent that helps Vietnamese people learn Mandarin Chinese, leveraging the ~60% Sino-Vietnamese vocabulary overlap as a learning bridge.

## Vision

Every Vietnamese speaker already knows the roots of thousands of Chinese words — they're just spoken differently. "Quốc gia" is 國家 (guójiā). "Giáo dục" is 教育 (jiàoyù). "Kinh tế" is 经济 (jīngjì). This project makes that connection explicit, turning a historical linguistic advantage into an accelerated learning experience powered by real-time AI voice conversation.

## Core Concept: The Hán Việt Bridge

| Vietnamese (Hán Việt) | Chinese (汉语) | Pinyin | English |
|---|---|---|---|
| quốc gia | 國家 | guójiā | country |
| giáo dục | 教育 | jiàoyù | education |
| kinh tế | 经济 | jīngjì | economy |
| đại học | 大學 | dàxué | university |
| điện thoại | 電話 | diànhuà | telephone |
| gia đình | 家庭 | jiātíng | family |
| sinh nhật | 生日 | shēngrì | birthday |
| bệnh viện | 病院 | bìngyuàn | hospital |

~60% of Vietnamese vocabulary has Chinese origins. This project leverages that overlap to accelerate Mandarin learning.

## Architecture

### Phase 1 — Text + Voice Messages (Hermes Agent Foundation)
- **Hermes Agent** as the brain: skills system, persistent memory, session management, messaging gateway
- **Web App** (Next.js): microphone recording → Qwen3-ASR STT → Agent reasoning → TTS audio reply
- Specialized learning skills: Hán Việt cognates, pronunciation guide, grammar, conversation practice
- Memory tracks student progress, weak areas, and learned vocabulary across sessions
- Available via Telegram/Discord gateway

### Phase 2 — Real-Time Speech-to-Speech (LiveKit)
- **LiveKit Agents** framework for full-duplex voice conversations
- Streaming pipeline: `VAD → STT → LLM → TTS`
- Vietnamese + Chinese auto-detection and code-switching
- Real-time pronunciation feedback with tone correction
- Barge-in / interruption support for natural conversation
- Sub-500ms end-to-end latency target

## Tech Stack

### Backend
| Component | Choice | Why |
|---|---|---|
| **Agent Framework** | Hermes Agent (Phase 1) + LiveKit (Phase 2) | Skills, memory, multi-platform + real-time voice |
| **Language** | Python 3.12 | Stable, all packages confirmed compatible (Hermes, qwen-asr, qwen-tts) |
| **LLM** | Qwen/Qwen3.5-9B | Hybrid DeltaNet+MoE architecture, 201 languages, 262K context, Apache 2.0 |
| **STT** | Qwen/Qwen3-ASR-1.7B | 52 languages/dialects including Vietnamese, streaming + offline, Apache 2.0 |
| **TTS Chinese** | Qwen/Qwen3-TTS-12Hz-1.7B-Base | 10 languages, 97ms streaming latency, voice cloning, Apache 2.0 |
| **TTS Vietnamese** | VieNeu-TTS v2 Turbo | Bilingual VN/EN, instant voice cloning, CPU-inference, Apache 2.0 |
| **VAD** | Silero VAD | Lightweight, industry standard |
| **Transport** | WebRTC via LiveKit | Low-latency real-time audio |

### Frontend
| Component | Choice | Why |
|---|---|---|
| **Framework** | Next.js 16 (App Router) | Turbopack default, Cache Components, React 19.2, React Compiler |
| **Audio** | LiveKit WebRTC client | Real-time bidirectional audio streams |
| **Styling** | Tailwind CSS + shadcn/ui | Fast, consistent, accessible |
| **State** | Zustand | Lightweight, simple |
| **Voice UI** | Custom Web Audio API | VAD visualization, waveform, tone display |

### Infrastructure
| Component | Choice | Why |
|---|---|---|
| **Package Manager** | uv | Fast Python package management |
| **Node Package Manager** | pnpm | Fast, disk-efficient |
| **Deployment** | LiveKit Cloud (voice) + Vercel (web) | Easy scaling, free tiers |

## Project Structure

```
language-learning-assistant/
├── README.md                  # This file
├── AGENTS.md                  # Agent context and instructions
├── .env.example               # Environment variable template
├── .gitignore
├── pyproject.toml             # Python project config
├── src/
│   ├── __init__.py
│   ├── prompts.py             # System prompts for Chinese tutor persona
│   ├── cognates.json          # Sino-Vietnamese ↔ 汉语 database
│   ├── scenarios.json         # Role-play conversation scenarios
│   ├── voice_agent.py         # LiveKit voice agent (Phase 2)
│   └── skills/
│       ├── hanviet/           # Sino-Vietnamese vocabulary skill
│       │   └── SKILL.md
│       ├── pronunciation/     # Tone & pronunciation guide skill
│       │   └── SKILL.md
│       ├── grammar/           # Chinese grammar lessons skill
│       │   └── SKILL.md
│       └── conversation/      # Role-play scenarios skill
│           └── SKILL.md
├── web/                       # Next.js frontend
│   ├── app/
│   ├── components/
│   ├── lib/
│   ├── public/
│   ├── package.json
│   └── ...
└── tests/
```

## Key Learning Features

### 1. Hán Việt Cognate Discovery
The agent identifies Sino-Vietnamese words the student already knows and reveals their Chinese counterparts, creating "aha!" moments that anchor new vocabulary to existing knowledge.

### 2. Tone Training
Visual tone comparison between Vietnamese (6 tones) and Mandarin (4 tones + neutral). Audio playback with side-by-side waveform visualization. Real-time tone accuracy scoring during practice.

### 3. Scenario-Based Practice
Role-play real-world situations: ordering food at a restaurant, shopping at a market, asking for directions, business meetings. Each scenario targets specific vocabulary and grammar patterns.

### 4. Progressive Curriculum (HSK-Aligned)
Structured learning path aligned with HSK levels (1-6), from basic greetings to fluent conversation. The agent adapts difficulty based on student performance tracked in memory.

### 5. Pronunciation Feedback
Phoneme-level analysis comparing student's pronunciation to native speakers. Specific feedback on initials, finals, and tones. Vietnamese-specific common mistakes highlighted (e.g., zh/z, ch/c, sh/s distinctions).

### 6. Persistent Progress Tracking
Memory system tracks: vocabulary learned, weak areas, tone accuracy scores, conversation fluency, and personalized learning recommendations.

## Why This Works

- **Linguistic advantage**: Vietnamese speakers have a ~60% head start on Chinese vocabulary through Hán Việt cognates
- **Tone familiarity**: Vietnamese speakers are already comfortable with tonal languages (6 tones vs 4)
- **SVO word order**: Both languages share Subject-Verb-Object sentence structure
- **Cultural proximity**: Shared historical, cultural, and Confucian context makes content relatable
- **AI voice practice**: Real-time voice practice without embarrassment — the AI never judges

## Inspiration & References

| Project | What We Learn |
|---|---|
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Skills system, memory, multi-platform messaging, voice mode |
| [LiveKit Agents](https://github.com/livekit/agents) | Real-time voice AI framework, WebRTC streaming pipeline |
| [CleanS2S](https://github.com/opendilab/CleanS2S) | Single-file S2S pipeline, full-duplex with interruption |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Hybrid DeltaNet+MoE LLM, 201 languages, efficient inference |
| [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 52-language ASR with streaming + offline modes |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 10-language TTS, 97ms streaming latency, voice cloning |
| [VieNeu-TTS](https://github.com/pnnbao97/VieNeu-TTS) | Bilingual VN/EN TTS, instant voice cloning, CPU inference |

## Getting Started

### Prerequisites
- Python 3.12+
- Node.js 20.9+ (required by Next.js 16)
- pnpm
- uv (Python package manager)
- CUDA-compatible GPU (recommended for local model inference; cloud API fallback available)

### Phase 1: Text + Voice Messages
```bash
# Set up Python environment
uv venv --python 3.12
source .venv/bin/activate

# Install Hermes Agent
pip install "hermes-agent[all]"

# Install Qwen models
pip install -U qwen-asr          # STT (Qwen3-ASR-1.7B)
pip install -U qwen-tts          # TTS (Qwen3-TTS-12Hz)
pip install -U vieneu            # Vietnamese TTS (VieNeu-TTS)

# Configure model
hermes model
hermes model set openrouter:qwen/qwen3.5-9b

# Start the agent
hermes

# Enable voice mode
/voice on
```

### Phase 2: Real-Time Voice
```bash
# Set up LiveKit project
lk cloud auth
lk agent init . --template agent-starter-python

# Install dependencies
uv pip install -e ".[openai,silero,turn-detector]"
uv pip install -U qwen-asr qwen-tts vieneu

# Download models (auto-downloaded on first run, or manually):
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./models/Qwen3-TTS-12Hz-1.7B-Base

# Run the voice agent
uv run python src/voice_agent.py dev
```

### Web Frontend
```bash
cd web
pnpm create next-app@latest . --typescript --tailwind --app
pnpm dev  # Runs with Turbopack (default in Next.js 16)
```

## Roadmap

```
Phase 1 (Foundation)
  □ Hermes Agent setup with Chinese tutor persona
  □ Hán Việt cognate database (500+ words)
  □ Specialized learning skills
  □ Web frontend with text + voice message mode
  □ Student progress memory

Phase 2 (Speech-to-Speech)
  □ LiveKit Agents integration
  □ Real-time voice pipeline (VAD → STT → LLM → TTS)
  □ Multilingual auto-detection (Vietnamese + Chinese)
  □ Real-time pronunciation feedback
  □ WebRTC web frontend

Phase 3 (Advanced)
  □ HSK-aligned curriculum with progressive difficulty
  □ Character writing practice with stroke order
  □ Voice cloning for personalized TTS
  □ Multiplayer practice (student ↔ student via AI matchmaker)
  □ Mobile app (React Native / Flutter)
```

## License

MIT
