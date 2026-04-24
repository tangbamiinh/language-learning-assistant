"""Speech-to-Text wrappers for LiveKit Agents.

Two backends:
- MLXSTT: Local Qwen3-ASR-1.7B-8bit via MLX (Apple Silicon GPU).
  Fast, offline, no API key needed. ~0.5-1s latency.
- DashScopeSTT: Qwen3-ASR via DashScope cloud API.
  Falls back when MLX is unavailable. ~1-2s latency.

Both are non-streaming STTs — use with StreamAdapter + VAD:

    from livekit.agents import stt
    from src.livekit_stt import create_stt

    stt_service = create_stt()  # auto-selects MLX > DashScope > raises
    session = AgentSession(
        stt=stt.StreamAdapter(stt_service, vad.stream()),
        vad=silero.VAD.load(),
        # ... llm, tts
    )

Environment:
    DASHSCOPE_API_KEY   - Required for DashScopeSTT.
    STT_BACKEND          - Force backend: "mlx", "dashscope", or "auto" (default).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any

import numpy as np
import soundfile as sf
from livekit import rtc
from livekit.agents import APIConnectOptions
from livekit.agents.stt import (
    STT,
    STTCapabilities,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
)
from livekit.agents.types import NOT_GIVEN
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Audio buffer helpers (shared)
# ─────────────────────────────────────────────────────────────────────────────

def buffer_to_numpy(buffer: AudioBuffer) -> np.ndarray:
    """Convert LiveKit AudioBuffer to a flat float32 numpy array.

    Handles both rtc.AudioFrame and raw numpy arrays.
    Normalizes int16 PCM to float32 [-1.0, 1.0].
    """
    frames: list[np.ndarray] = []

    for item in buffer:
        if isinstance(item, rtc.AudioFrame):
            dtype = np.int16  # LiveKit default for mic audio
            arr = np.frombuffer(item.data, dtype=dtype).copy()
            arr = arr.astype(np.float32) / 32768.0
            frames.append(arr)
        elif isinstance(item, np.ndarray):
            frames.append(item.astype(np.float32))

    if not frames:
        return np.array([], dtype=np.float32)

    return np.concatenate(frames)


def detect_sample_rate(buffer: AudioBuffer) -> int:
    """Extract sample rate from the first AudioFrame in the buffer."""
    for item in buffer:
        if isinstance(item, rtc.AudioFrame):
            return item.sample_rate
    return 48000  # LiveKit default


def numpy_to_wav_path(audio_np: np.ndarray, sample_rate: int) -> str:
    """Write a float32 numpy array to a temporary WAV file.

    Returns the file path. Caller is responsible for cleanup.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    # Resample to 16kHz if needed (Qwen3-ASR expects 16kHz mono)
    target_sr = 16000
    if sample_rate != target_sr:
        try:
            import librosa
            audio_np = librosa.resample(
                audio_np, orig_sr=sample_rate, target_sr=target_sr
            )
        except ImportError:
            # Simple decimation if librosa unavailable
            ratio = sample_rate // target_sr
            audio_np = audio_np[::ratio]
        sample_rate = target_sr

    sf.write(tmp.name, audio_np, sample_rate, subtype="FLOAT")
    return tmp.name


# ─────────────────────────────────────────────────────────────────────────────
# MLX STT — Local Qwen3-ASR on Apple Silicon
# ─────────────────────────────────────────────────────────────────────────────

class MLXSTT(STT):
    """Local Qwen3-ASR-1.7B-8bit via MLX (Apple Silicon GPU).

    Runs entirely on-device. No API key, no network.
    Model: mlx-community/Qwen3-ASR-1.7B-8bit (2.46 GB, 8-bit quantized)

    Args:
        model_path: HuggingFace model ID or local path.
        language: Optional ISO 639-1 language code. None for auto-detect.

    Raises:
        ImportError: If mlx-audio is not installed.
    """

    def __init__(
        self,
        *,
        model_path: str = "mlx-community/Qwen3-ASR-1.7B-8bit",
        language: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )
        self._model_path = model_path
        self._language = language
        self._model = None
        self._loaded = False
        logger.info("MLXSTT configured (model=%s, language=%s)", model_path, language)

    def _ensure_loaded(self) -> Any:
        """Lazy-load the MLX model."""
        if self._loaded:
            return self._model

        try:
            from mlx_audio.stt.utils import load_model
        except ImportError as e:
            raise ImportError(
                "mlx-audio is not installed. Run: pip install mlx-audio"
            ) from e

        logger.info("Loading MLX ASR model: %s", self._model_path)
        self._model = load_model(self._model_path)
        self._loaded = True
        logger.info("MLX ASR model loaded successfully")
        return self._model

    @property
    def model(self) -> str:
        return self._model_path

    @property
    def provider(self) -> str:
        return "mlx"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> SpeechEvent:
        """Transcribe an audio buffer via MLX Qwen3-ASR."""
        from mlx_audio.stt.generate import generate_transcription

        lang = language if language != NOT_GIVEN else self._language
        model = self._ensure_loaded()

        audio_np = buffer_to_numpy(buffer)
        sample_rate = detect_sample_rate(buffer)
        wav_path = numpy_to_wav_path(audio_np, sample_rate)

        logger.debug(
            "MLX transcribing %d samples @ %d Hz (language=%s)",
            len(audio_np),
            sample_rate,
            lang,
        )

        try:
            # MLX is GPU-bound — run directly (non-blocking on CPU).
            # Using run_in_executor breaks Metal streams (thread-local).
            kwargs: dict[str, Any] = {
                "model": model,
                "audio": wav_path,
            }
            if lang:
                kwargs["language"] = lang
            result = generate_transcription(**kwargs)

            # STTOutput: text is full transcription, language is a list
            text = getattr(result, "text", "") or ""
            lang_list = getattr(result, "language", []) or []
            detected_lang = lang_list[0] if lang_list else ""

            # Map display language name to ISO code
            iso_lang = _display_to_iso(detected_lang) or (lang or "")

            if not text.strip():
                logger.warning("MLX STT returned empty result")
                return SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechData(text="", language=iso_lang)],
                )

            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    SpeechData(
                        text=text.strip(),
                        language=iso_lang,
                        confidence=0.0,  # MLX doesn't expose confidence
                    )
                ],
            )

        except Exception as e:
            logger.error("MLX STT failed: %s", e)
            raise
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    async def aclose(self) -> None:
        """Release the MLX model."""
        self._model = None
        self._loaded = False


# ─────────────────────────────────────────────────────────────────────────────
# DashScope STT — Cloud Qwen3-ASR API
# ─────────────────────────────────────────────────────────────────────────────

class DashScopeSTT(STT):
    """Wraps STTService(backend='dashscope') as a LiveKit-compatible STT.

    Implements the non-streaming _recognize_impl() interface.
    Use with StreamAdapter for VAD-based buffering.

    Args:
        language: Optional ISO 639-1 language code. None for auto-detect.
        dashscope_api_key: Optional API key. Falls back to DASHSCOPE_API_KEY env var.
        dashscope_model: DashScope model name (default 'qwen3-asr-flash').
    """

    def __init__(
        self,
        *,
        language: str | None = None,
        dashscope_api_key: str | None = None,
        dashscope_model: str = "qwen3-asr-flash",
    ) -> None:
        super().__init__(
            capabilities=STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )
        from src.stt_service import STTService

        self._service = STTService(
            backend="dashscope",
            dashscope_api_key=dashscope_api_key,
            dashscope_model=dashscope_model,
        )
        self._language = language
        logger.info("DashScopeSTT initialized (language=%s)", language)

    @property
    def model(self) -> str:
        return self._service._dashscope_model  # type: ignore[attr-defined]

    @property
    def provider(self) -> str:
        return "dashscope"

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> SpeechEvent:
        """Transcribe an audio buffer via DashScope API."""
        lang = language if language != NOT_GIVEN else self._language

        audio_np = buffer_to_numpy(buffer)
        sample_rate = detect_sample_rate(buffer)
        audio_bytes = audio_np.tobytes()

        logger.debug(
            "Transcribing %d samples @ %d Hz (language=%s)",
            len(audio_np),
            sample_rate,
            lang,
        )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, self._service.transcribe_bytes, audio_bytes, sample_rate, lang
            )

            if not result.text.strip():
                logger.warning("Empty transcription result")
                return SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[SpeechData(text="", language=lang or "")],
                )

            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    SpeechData(
                        text=result.text.strip(),
                        language=result.language or (lang or ""),
                        confidence=result.confidence,
                    )
                ],
            )

        except Exception as e:
            logger.error("DashScope STT failed: %s", e)
            raise


# ─────────────────────────────────────────────────────────────────────────────
# Language mapping helpers
# ─────────────────────────────────────────────────────────────────────────────

_DISPLAY_TO_ISO: dict[str, str] = {
    "Vietnamese": "vi",
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Russian": "ru",
    "Thai": "th",
    "Hindi": "hi",
    "None": "",  # Qwen3-ASR returns "None" for undetected languages
}


def _display_to_iso(display_name: str) -> str | None:
    """Map Qwen3-ASR language display name to ISO 639-1 code."""
    if not display_name:
        return None
    # Try exact match first
    if display_name in _DISPLAY_TO_ISO:
        return _DISPLAY_TO_ISO[display_name]
    # Case-insensitive
    lower = display_name.lower()
    for key, val in _DISPLAY_TO_ISO.items():
        if key.lower() == lower:
            return val
    # Pass through if already an ISO code
    if display_name.lower() in {"vi", "zh", "en", "yue", "ja", "ko"}:
        return display_name.lower()
    return display_name.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect and create best available STT
# ─────────────────────────────────────────────────────────────────────────────

def create_stt(
    *,
    language: str | None = None,
    backend: str | None = None,
) -> STT:
    """Create the best available STT backend.

    Priority: MLX (local, Apple Silicon) > DashScope (cloud API)

    Args:
        language: Optional ISO 639-1 language code.
        backend: Force backend: 'mlx', 'dashscope', or 'auto' (default).

    Returns:
        An STT instance ready to use with StreamAdapter.

    Raises:
        RuntimeError: If no suitable backend is available.
    """
    requested = (backend or os.getenv("STT_BACKEND", "auto")).lower()

    if requested != "auto":
        if requested == "mlx":
            logger.info("STT backend forced to MLX")
            return MLXSTT(language=language)
        elif requested == "dashscope":
            logger.info("STT backend forced to DashScope")
            return DashScopeSTT(language=language)
        else:
            raise ValueError(f"Unknown STT backend: {backend}")

    # Auto-detect: try MLX first, fall back to DashScope
    try:
        import mlx.core as mx  # noqa: F401

        if mx.metal.is_available():
            logger.info("STT: MLX backend available (Apple Silicon GPU)")
            return MLXSTT(language=language)
        else:
            logger.warning("STT: MLX available but Metal not supported")
    except ImportError:
        logger.info("STT: mlx not installed, trying DashScope")

    # Fall back to DashScope
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not dashscope_key:
        logger.error(
            "STT: No backend available. Install mlx-audio for local inference, "
            "or set DASHSCOPE_API_KEY for cloud inference."
        )
        raise RuntimeError(
            "No STT backend available. Either:\n"
            "  1. Install mlx-audio (Apple Silicon): pip install mlx-audio\n"
            "  2. Set DASHSCOPE_API_KEY env variable for cloud STT"
        )

    logger.info("STT: Using DashScope cloud backend")
    return DashScopeSTT(language=language)
