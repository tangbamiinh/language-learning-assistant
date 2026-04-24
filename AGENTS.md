# Language Learning Assistant - Agent Context

## Quick Start
```bash
make setup      # first time: creates .venv (Python 3.12), installs all deps
make dev        # runs Python bridge (localhost:8000) + Next.js web (localhost:3000)
make stop       # kill background Python server
make docker-up  # start LiveKit server + voice agent (Docker)
```
Always use `make` commands. Never run `python` directly — the venv is at `.venv/bin/python3`.
`source .env` does nothing — `python-dotenv` loads `.env` automatically on import.

## Architecture
Two modes, three services:

```
TEXT MODE (Phase 1):
Web (localhost:3000)  --proxy-->  Python Bridge (localhost:8000)  --API-->  OpenRouter (Qwen3.5-9B)
    Next.js 16 + React 19.2            FastAPI server                   OPENROUTER_API_KEY required
    └── /api/chat routes to :8000      └── /api/chat, /api/voice, /api/cognates
    └── /api/voice routes to :8000     └── in-memory session history (20 msgs max)

VOICE MODE (Phase 2):
Browser WebRTC ──> LiveKit SFU (localhost:7880) ──> voice_agent (Docker)
    └── LiveKitVoiceRoom.tsx              └── silero VAD + MLX STT + LLM + BilingualTTS
```

Web API routes proxy to the Python backend. If backend is down, they return mock responses — UI works standalone.
Voice mode is a separate process — it doesn't use `server.py`.

## Directory Map
- **`src/server.py`** — FastAPI bridge (Phase 1 entrypoint). Connects web → OpenRouter LLM. Lazy-loads STT/TTS.
- **`src/stt_service.py`** — Qwen3-ASR wrapper. Transformers backend (GPU) or DashScope (cloud fallback).
- **`src/tts_service.py`** — Dual backend: Qwen3-TTS (Chinese, 10 langs) + VieNeu-TTS (Vietnamese, CPU). Device: CUDA > MPS > CPU.
- **`src/livekit_stt.py`** — LiveKit-compatible STT wrappers. `MLXSTT` (local Qwen3-ASR, Apple Silicon GPU) + `DashScopeSTT` (cloud). `create_stt()` auto-selects. Non-streaming — use with `StreamAdapter + VAD`.
- **`src/livekit_tts.py`** — `BilingualTTS` for LiveKit. Routes Vietnamese → VieNeu-TTS (CPU streaming), Chinese/English → Qwen3-TTS (MPS/CPU).
- **`src/voice_agent.py`** — Fully implemented LiveKit voice agent. `ChineseTutorAgent` with `tts_node()` override. Run via `make docker-up` or `python -m src.voice_agent dev`.
- **`src/cognates.json`** — 308 Sino-Vietnamese cognates. Loaded at server startup, injected as LLM context.
- **`src/scenarios.json`** — 10 role-play scenarios (HSK 1-5). Used by conversation skill.
- **`src/prompts.py`** — System prompts + prompt builders. `SYSTEM_PROMPT` (text), `SYSTEM_PROMPT_VOICE` (voice).
- **`src/skills/`** — 4 Hermes-compatible SKILL.md files. Context injected into LLM prompts by `server.py`, NOT via Hermes Agent.
- **`web/`** — Next.js 16 app. `src/app/api/` routes proxy to Python. `ChatUI.tsx` + `VoiceRecorder.tsx` + `LiveKitVoiceRoom.tsx`.
- **`web/src/lib/api.ts`** — API client used by hooks. Points to relative `/api/*` (proxied by Next.js).

## Important Gotchas
- **`src/__init__.py` has lazy imports** — importing `src` won't load torch/numpy. Intentional to avoid GPU deps at import time.
- **Cognates are injected as text context**, not queried from a database. `server.py` does substring matching on user messages.
- **STT/TTS degrade gracefully** — if no GPU, server returns text-only responses. `make health` checks availability.
- **`.venv` must exist** or `make` will error. Run `make setup` on a fresh clone.
- **`OPENROUTER_API_KEY` is required** — server won't start a meaningful conversation without it.
- **`DASHSCOPE_API_KEY`** needed for cloud STT fallback (set in `.env`). Not required if MLX (Apple Silicon) is available.
- **Python tests don't need GPU** — `test_stt.py` and `test_tts.py` mock heavy imports. Run with `make test`. Some STT/TTS tests are stale (outdated API names) — 84/91 tests currently pass.
- **MLX Metal streams are thread-local** — MLX inference must run on the main thread, NOT in `run_in_executor`. `livekit_stt.py` handles this correctly.
- **`qwen-tts` requires `transformers==4.57.3`**, but `mlx-audio` needs `>=5.0.0`. We use `4.57.3` — MLX STT still works despite pip warnings.

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
| `make docker-up` | Start LiveKit server + voice agent (Docker) |
| `make docker-down` | Stop LiveKit + voice agent |
| `make docker-logs` | View LiveKit + voice agent logs |
| `make docker-rebuild` | Rebuild and restart voice agent container |
| `make docker-agent-dev` | Run voice agent in dev mode (hot reload) |
| `make all` | Start everything (LiveKit + web app) |

## LiveKit / Voice Agent
- **Self-hosted** — `docker-compose.yml` runs LiveKit server + voice agent container
- **Dev creds**: `LIVEKIT_API_KEY=devkey`, `LIVEKIT_API_SECRET=secret`, `LIVEKIT_URL=ws://localhost:7880`
- **Web integration**: `LiveKitVoiceRoom.tsx` connects via WebRTC using token from `/api/livekit-token`
- **Pipeline**: Silero VAD → `StreamAdapter(MLXSTT/DashScopeSTT)` → OpenRouter LLM → `BilingualTTS`
- **Voice agent is a separate process** — doesn't share code with `server.py` or the Phase 1 bridge
