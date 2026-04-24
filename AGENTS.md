# Language Learning Assistant - Agent Context

## Quick Start
```bash
make setup      # first time: creates .venv (Python 3.12), installs all deps
make dev        # runs Python bridge (localhost:8000) + Next.js web (localhost:3000)
make stop       # kill background Python server
```
Always use `make` commands. Never run `python` directly — the venv is at `.venv/bin/python3`.
`source .env` does nothing — `python-dotenv` loads `.env` automatically on import.

## Architecture
Two processes, two ports, one data flow:

```
Web (localhost:3000)  --proxy-->  Python Bridge (localhost:8000)  --API-->  OpenRouter (Qwen3.5-9B)
    Next.js 16 + React 19.2            FastAPI server                   OPENROUTER_API_KEY required
    └── /api/chat routes to :8000      └── /api/chat, /api/voice, /api/cognates
    └── /api/voice routes to :8000     └── in-memory session history (20 msgs max)
```

Web API routes proxy to the Python backend. If backend is down, they return mock responses — UI works standalone.

## Directory Map
- **`src/server.py`** — FastAPI bridge (entrypoint). Connects web → OpenRouter LLM. Lazy-loads STT/TTS.
- **`src/stt_service.py`** — Qwen3-ASR wrapper. Transformers backend (GPU) or DashScope (cloud fallback).
- **`src/tts_service.py`** — Dual backend: Qwen3-TTS (Chinese, 10 langs) + VieNeu-TTS (Vietnamese, CPU).
- **`src/voice_agent.py`** — Phase 2 placeholder. LiveKit voice agent pattern (commented, ready to uncomment).
- **`src/cognates.json`** — 308 Sino-Vietnamese cognates. Loaded at server startup, injected as LLM context.
- **`src/scenarios.json`** — 10 role-play scenarios (HSK 1-5). Used by conversation skill.
- **`src/prompts.py`** — System prompts + prompt builders. `SYSTEM_PROMPT` (text), `SYSTEM_PROMPT_VOICE` (voice).
- **`src/skills/`** — 4 Hermes-compatible SKILL.md files. Context injected into LLM prompts by `server.py`, NOT via Hermes Agent.
- **`web/`** — Next.js 16 app. `src/app/api/` routes proxy to Python. `src/components/ChatUI.tsx` + `VoiceRecorder.tsx`.
- **`web/src/lib/api.ts`** — API client used by hooks. Points to relative `/api/*` (proxied by Next.js).

## Important Gotchas
- **`src/__init__.py` has lazy imports** — importing `src` won't load torch/numpy. This is intentional to avoid GPU deps at import time.
- **Cognates are injected as text context**, not queried from a database. `server.py` does substring matching on user messages.
- **STT/TTS degrade gracefully** — if no GPU, server returns text-only responses. `make health` checks availability.
- **`.venv` must exist** or `make` will error. Run `make setup` on a fresh clone.
- **`OPENROUTER_API_KEY` is required** — server won't start a meaningful conversation without it.
- **Python tests don't need GPU** — `test_stt.py` and `test_tts.py` mock heavy imports. Run with `make test`.

## Commands Reference
| Command | What it does |
|---|---|
| `make setup` | Create `.venv`, install Python + web deps |
| `make dev` | Start Python bridge (bg) + web app (fg) |
| `make run-server` | Python bridge only (`:8000`) |
| `make run-web` | Web app only (`:3000`) |
| `make stop` | Kill background Python server |
| `make test` | Run all Python tests |
| `make web-build` | Production Next.js build |
| `make web-lint` | ESLint |
| `make health` | Curl `/api/health` |
| `make check` | Verify env + deps are ready |
| `make clean` | Remove build artifacts |

## Phase 2 (LiveKit)
`src/voice_agent.py` has the full implementation pattern commented in. To activate: uncomment the code, install `livekit-agents[silero,turn-detector]`, and set `LIVEKIT_*` env vars. This is a separate process from Phase 1 — it doesn't use `server.py`.
