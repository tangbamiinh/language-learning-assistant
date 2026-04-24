"""Phase 2: LiveKit Voice Agent - Real-time Speech-to-Speech Mandarin Tutor.

This module implements a real-time voice agent using the LiveKit Agents framework.
The pipeline: VAD (Silero) → STT (MLX Qwen3-ASR / DashScope) → LLM (OpenRouter/Qwen3.5-9B) → TTS (VieNeu/Qwen3-TTS)

Run with:
    python -m src.voice_agent dev          # Dev mode with hot reload
    python -m src.voice_agent console      # Local terminal testing (text-only)
    python -m src.voice_agent download-files  # Download VAD/turn detector models
    python -m src.voice_agent start        # Production mode

Environment variables (from .env):
    LIVEKIT_URL          - LiveKit server WebSocket URL (ws://localhost:7880 for self-hosted)
    LIVEKIT_API_KEY      - LiveKit API key (devkey for dev mode)
    LIVEKIT_API_SECRET   - LiveKit API secret (secret for dev mode)
    OPENROUTER_API_KEY   - OpenRouter API key for LLM
    DASHSCOPE_API_KEY    - DashScope API key for STT fallback (optional if MLX available)
    STT_BACKEND          - Force STT: 'mlx', 'dashscope', or 'auto' (default)

Architecture:
    - STT: MLX Qwen3-ASR-1.7B-8bit (local, Apple Silicon GPU) or DashScope (cloud fallback)
    - TTS: Bilingual — VieNeu-TTS for Vietnamese (CPU streaming), Qwen3-TTS for Chinese (MPS/CPU)
    - LLM: Qwen3.5-9B via OpenRouter
    - VAD: Silero (CPU)
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    stt,
)
from livekit.plugins import silero
from livekit.plugins.openai import LLM

from src.livekit_stt import create_stt
from src.livekit_tts import BilingualTTS

logger = logging.getLogger("voice-agent")

load_dotenv()


# ─── Agent Definition ─────────────────────────────────────────────────────────

class ChineseTutorAgent(Agent):
    """Bilingual Chinese tutor agent with Hán Việt cognate awareness."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Teacher Min (小明), a friendly Mandarin Chinese tutor "
                "for Vietnamese speakers. You are patient, encouraging, and culturally aware.\n\n"
                "Guidelines:\n"
                "- Start conversations in Vietnamese, then gently switch to Chinese as the user learns\n"
                "- Leverage Sino-Vietnamese (Hán Việt) cognates to help the user understand new vocabulary\n"
                "- Keep responses concise and conversational — you're speaking, not writing\n"
                "- Avoid complex formatting, punctuation, emojis, or asterisks\n"
                "- If the user struggles, switch to Vietnamese briefly to explain, then return to Chinese\n"
                "- Praise progress and correct mistakes gently\n"
                "- Use pinyin when introducing new characters\n"
                "- Adapt to the user's HSK level (ask if unsure)"
            ),
        )
        self._tts = BilingualTTS()

    async def tts_node(self, text_stream, model_settings):
        """Override tts_node to use our bilingual TTS.

        Vietnamese text → VieNeu-TTS (CPU, streaming)
        Chinese/English text → Qwen3-TTS (MPS or CPU)
        """
        async for frame in self._tts.tts_node(text_stream, model_settings):
            yield frame


# ─── Server Setup ──────────────────────────────────────────────────────────────

server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    """Load VAD model and warm STT backend before handling sessions."""
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD loaded in prewarm")

    # Warm STT backend (loads MLX model into GPU memory if available)
    stt_service = create_stt()
    proc.userdata["stt"] = stt_service
    logger.info("STT backend warmed: %s", stt_service.provider)


server.setup_fnc = prewarm


@server.rtc_session(agent_name="chinese-tutor")
async def entrypoint(ctx: JobContext) -> None:
    """Handle each new voice session."""
    # Add context to all log entries
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    vad = ctx.proc.userdata["vad"]

    # Set up STT: auto-selects MLX (local) or DashScope (cloud)
    # Already prewarmed in prewarm() — use from proc.userdata
    base_stt = ctx.proc.userdata["stt"]
    stt_stream = stt.StreamAdapter(base_stt, vad.stream())

    # Set up the agent session
    session = AgentSession(
        # ─── LLM: Language Model ──────────────────────────────────────────
        # Using Qwen3.5-9B via OpenRouter (reads OPENROUTER_API_KEY from env)
        llm=LLM.with_openrouter(
            model="qwen/qwen3.5-9b",
            app_name="Language Learning Assistant",
        ),

        # ─── Voice Activity Detection ─────────────────────────────────────
        vad=vad,

        # ─── Speech-to-Text ───────────────────────────────────────────────
        # MLX Qwen3-ASR-1.7B-8bit (local GPU) or DashScope (cloud API)
        stt=stt_stream,
    )

    # Start the session with the Chinese tutor agent
    await session.start(
        agent=ChineseTutorAgent(),
        room=ctx.room,
    )

    # Connect to the room
    await ctx.connect()

    # Generate initial greeting
    await session.generate_reply(
        instructions="Greet the user warmly in Vietnamese, introduce yourself as Teacher Min (小明), and ask what they'd like to practice in Chinese today."
    )


if __name__ == "__main__":
    cli.run_app(server)
