"""Language Learning Assistant - AI-powered Mandarin Chinese tutor for Vietnamese speakers.

Heavy dependencies (torch, numpy, qwen-asr, qwen-tts, vieneu) are imported
lazily to allow lightweight operations like prompt building and server startup
without GPU dependencies.
"""

from __future__ import annotations

from src.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VOICE,
    build_cognate_prompt,
    build_scenario_prompt,
    build_grammar_prompt,
    build_tone_drill_prompt,
)

__version__ = "0.1.0"
__all__ = [
    "STTService",
    "TranscriptionResult",
    "TTSService",
    "SynthesisResult",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_VOICE",
    "build_cognate_prompt",
    "build_scenario_prompt",
    "build_grammar_prompt",
    "build_tone_drill_prompt",
]


def __getattr__(name: str):
    """Lazy-import heavy services to avoid requiring torch/numpy at import time."""
    if name in ("STTService", "TranscriptionResult"):
        from src.stt_service import STTService, TranscriptionResult

        return STTService if name == "STTService" else TranscriptionResult
    if name in ("TTSService", "SynthesisResult"):
        from src.tts_service import SynthesisResult, TTSService

        return TTSService if name == "TTSService" else SynthesisResult
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
