"""Bilingual Text-to-Speech wrapper for LiveKit Agents.

Routes Vietnamese text to VieNeu-TTS (CPU, streaming) and Chinese/English
text to Qwen3-TTS (MPS/GPU or CPU fallback).

Used as a tts_node override in the ChineseTutorAgent.

Usage:
    from src.livekit_tts import BilingualTTS

    class ChineseTutorAgent(Agent):
        def __init__(self):
            super().__init__(instructions="...")
            self._tts = BilingualTTS()

        async def tts_node(self, text_stream, model_settings):
            async for frame in self._tts.tts_node(text_stream, model_settings):
                yield frame

Environment:
    QWEN_TTS_MODEL_PATH  - HuggingFace model ID for Qwen3-TTS (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)
    VIENEU_MODE          - VieNeu-TTS backend mode (default: "turbo" for CPU)
"""

from __future__ import annotations

import io
import logging
from typing import AsyncIterable

import numpy as np
import soundfile as sf
from livekit import rtc
from livekit.agents import ModelSettings

logger = logging.getLogger(__name__)

# Vietnamese diacritics — used for language detection
_VIETNAMESE_CHARS = set(
    "àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩị"
    "òóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊ"
    "ÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ"
)

# CJK Unified Ideographs range
_CJK_START = 0x4E00
_CJK_END = 0x9FFF


class BilingualTTS:
    """Routes Vietnamese → VieNeu-TTS, Chinese/English → Qwen3-TTS.

    Language detection is per-segment: each text chunk is classified by
    character analysis, then routed to the appropriate backend.

    VieNeu-TTS supports streaming (infer_stream) — ideal for Vietnamese.
    Qwen3-TTS does not support streaming — returns full audio chunk.
    """

    def __init__(self) -> None:
        from src.tts_service import TTSService

        self._tts = TTSService()
        logger.info("BilingualTTS initialized")

    async def tts_node(
        self,
        text_stream: AsyncIterable[str],
        model_settings: ModelSettings,
    ) -> AsyncIterable[rtc.AudioFrame]:
        """LiveKit tts_node implementation.

        Consumes text segments from the LLM and yields audio frames.
        Vietnamese text → VieNeu-TTS (streaming PCM chunks).
        Chinese/English text → Qwen3-TTS (full WAV → single frame).

        Args:
            text_stream: Async iterable of text segments from LLM.
            model_settings: Model configuration (unused, for interface compat).

        Yields:
            rtc.AudioFrame: 16-bit PCM audio frames at 24kHz mono.
        """
        from src.tts_service import Backend

        async for text in text_stream:
            if not text or not text.strip():
                continue

            lang = self._detect_language(text)
            logger.debug("TTS segment: lang=%s text=%r", lang, text[:80])

            try:
                if lang == "vi":
                    # VieNeu-TTS: streaming PCM chunks (24kHz, int16)
                    async for chunk_bytes in self._tts.stream_synthesize(
                        text, language="vi"
                    ):
                        frame = self._bytes_to_frame(
                            chunk_bytes, sample_rate=24000
                        )
                        yield frame
                else:
                    # Qwen3-TTS for Chinese or English
                    # Does NOT support streaming — returns full WAV
                    backend = self._tts._select_backend(lang)  # type: ignore[attr-defined]
                    if backend == Backend.VIENEU:
                        # Fallback: VieNeu for English
                        async for chunk_bytes in self._tts.stream_synthesize(
                            text, language=lang
                        ):
                            yield self._bytes_to_frame(chunk_bytes, 24000)
                    else:
                        result = await self._tts.synthesize(text, language=lang)
                        audio = self._wav_to_pcm(result.audio_bytes)
                        frame = rtc.AudioFrame(
                            sample_rate=result.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(audio) // 2,
                            data=audio.tobytes(),
                        )
                        yield frame

            except Exception as e:
                logger.error("TTS synthesis failed for %s: %s", lang, e)
                # Don't break the stream — skip this segment

    # ─── Language Detection ─────────────────────────────────────────────────

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect language from text characters.

        Heuristic: CJK ideographs → zh, Vietnamese diacritics → vi, else en.
        """
        if not text.strip():
            return "en"

        text_len = len(text)
        zh_count = sum(1 for c in text if _CJK_START <= ord(c) <= _CJK_END)
        vi_count = sum(1 for c in text if c in _VIETNAMESE_CHARS)

        # Chinese if >20% CJK characters
        if zh_count > text_len * 0.2:
            return "zh"
        # Vietnamese if >5% diacritics
        if vi_count > text_len * 0.05:
            return "vi"
        return "en"

    # ─── Audio Conversion ───────────────────────────────────────────────────

    @staticmethod
    def _bytes_to_frame(pcm_bytes: bytes, sample_rate: int) -> rtc.AudioFrame:
        """Convert raw PCM int16 bytes to an rtc.AudioFrame."""
        return rtc.AudioFrame(
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(pcm_bytes) // 2,
            data=pcm_bytes,
        )

    @staticmethod
    def _wav_to_pcm(wav_bytes: bytes) -> np.ndarray:
        """Parse WAV bytes and return int16 PCM as numpy array.

        VieNeu stream_synthesize yields raw PCM, but Qwen3 synthesize
        returns WAV-encoded audio. This converts WAV → int16 PCM array.
        """
        with io.BytesIO(wav_bytes) as buf:
            audio, sr = sf.read(buf, dtype="int16", always_2d=False)
        return np.asarray(audio, dtype=np.int16)
