"""Text-to-Speech synthesis service with dual backend support.

Wraps Qwen3-TTS (Chinese + 10 languages) and VieNeu-TTS (Vietnamese + English)
into a unified async API. Auto-selects the optimal backend based on target
language, handles GPU initialization gracefully, and caches voice clone prompts
for repeated use.

Environment variables:
    QWEN_TTS_MODEL_PATH: Path or HuggingFace ID for the Qwen3-TTS model.
        Default: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    VIENEU_MODE: VieNeu-TTS backend mode. "turbo" (default, CPU GGUF) or
        "gpu" (PyTorch GPU). Also accepts "standard", "fast", "remote", "xpu".
    TTS_ENGLISH_BACKEND: Which backend to use for English text. "qwen" (default)
        or "vieneu".
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Optional

import numpy as np
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Lazy imports – avoid importing heavy GPU libs until the model is actually
# needed.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------


class Backend(str, Enum):
    """Supported TTS backends."""

    QWEN = "qwen"
    VIENEU = "vieneu"


@dataclass
class VoiceClonePrompt:
    """Cached voice clone prompt for reuse across synthesis calls.

    Attributes:
        backend: Which TTS backend produced this prompt.
        qwen_prompt: Raw prompt items for Qwen3-TTS (if applicable).
        vieneu_codes: Encoded reference codes for VieNeu-TTS (if applicable).
        ref_text: Transcript of the reference audio (for ICL mode).
    """

    backend: Backend
    qwen_prompt: Any = None
    vieneu_codes: Optional[np.ndarray] = None
    ref_text: Optional[str] = None


@dataclass
class SynthesisResult:
    """Result of a TTS synthesis operation.

    Attributes:
        audio_bytes: WAV-encoded audio data (24 kHz, mono).
        sample_rate: Sample rate of the audio (always 24000).
        duration_seconds: Duration of the synthesized audio in seconds.
        backend: Which backend produced the audio.
    """

    audio_bytes: bytes
    sample_rate: int = 24000
    duration_seconds: float = 0.0
    backend: Backend = Backend.QWEN


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE = 24000  # All output normalized to 24 kHz

# Language code → backend mapping
LANGUAGE_BACKEND: dict[str, Backend] = {
    "zh": Backend.QWEN,
    "vi": Backend.VIENEU,
    "en": Backend.QWEN,  # Overridden by TTS_ENGLISH_BACKEND env var
}

# Qwen3-TTS language name mapping
QWEN_LANGUAGE_MAP: dict[str, str] = {
    "zh": "Chinese",
    "vi": "Auto",  # Vietnamese not natively supported, fallback
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a numpy audio array to WAV bytes.

    Args:
        audio: Mono float32 waveform, values in [-1.0, 1.0].
        sample_rate: Sample rate of the audio.

    Returns:
        WAV-encoded bytes.
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _resample_if_needed(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample audio to TARGET_SAMPLE_RATE if necessary.

    Args:
        audio: Input waveform (float32).
        orig_sr: Original sample rate.

    Returns:
        Resampled waveform at TARGET_SAMPLE_RATE, or the input unchanged.
    """
    if orig_sr == TARGET_SAMPLE_RATE:
        return audio
    try:
        import librosa

        return librosa.resample(
            audio.astype(np.float32),
            orig_sr=orig_sr,
            target_sr=TARGET_SAMPLE_RATE,
        )
    except ImportError:
        logger.warning(
            "librosa not installed; skipping resample from %d to %d Hz. "
            "Install librosa for accurate resampling.",
            orig_sr,
            TARGET_SAMPLE_RATE,
        )
        return audio


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is mono by averaging channels if stereo."""
    if audio.ndim > 1:
        return np.mean(audio, axis=-1).astype(np.float32)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# TTS Service
# ---------------------------------------------------------------------------


class TTSService:
    """Unified async TTS service wrapping Qwen3-TTS and VieNeu-TTS.

    Lazy-loads models on first use. Auto-selects the backend based on
    target language. Supports voice cloning with prompt caching and
    streaming synthesis.

    Example:
        >>> tts = TTSService()
        >>> result = await tts.synthesize("你好世界", language="zh")
        >>> await tts.synthesize("Xin chao the gioi", language="vi")
        >>> # Voice cloning
        >>> prompt = await tts.create_voice_clone_prompt(ref_audio, "transcript")
        >>> result = await tts.synthesize_with_clone("Custom text", "zh", prompt)

    Attributes:
        qwen_model: Loaded Qwen3TTSModel (or None if not yet loaded).
        vieneu_tts: Loaded Vieneu instance (or None if not yet loaded).
    """

    def __init__(
        self,
        qwen_model_path: Optional[str] = None,
        vieneu_mode: Optional[str] = None,
        english_backend: Optional[Backend] = None,
        use_flash_attention: bool = True,
    ) -> None:
        """Initialize the TTS service with lazy-loaded backends.

        Args:
            qwen_model_path: HuggingFace ID or local path for Qwen3-TTS model.
                Falls back to QWEN_TTS_MODEL_PATH env var, then to the default.
            vieneu_mode: VieNeu-TTS mode ("turbo" / "standard" / "fast" / "remote").
                Falls back to VIENEU_MODE env var, then to "turbo".
            english_backend: Which backend to use for English synthesis.
                Falls back to TTS_ENGLISH_BACKEND env var, then to Qwen3-TTS.
            use_flash_attention: Whether to use FlashAttention-2 for Qwen3-TTS.
                Requires flash-attn package and compatible GPU.
        """
        self._qwen_model: Any = None
        self._vieneu_tts: Any = None
        self._qwen_model_path = (
            qwen_model_path
            or os.environ.get("QWEN_TTS_MODEL_PATH")
            or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        self._vieneu_mode = vieneu_mode or os.environ.get("VIENEU_MODE") or "turbo"
        self._english_backend = (
            english_backend
            or (
                Backend(os.environ["TTS_ENGLISH_BACKEND"])
                if "TTS_ENGLISH_BACKEND" in os.environ
                else Backend.QWEN
            )
        )
        self._use_flash_attention = use_flash_attention and torch.cuda.is_available()
        self._voice_clone_cache: dict[str, VoiceClonePrompt] = {}

        # Override English backend mapping
        if self._english_backend != Backend.QWEN:
            LANGUAGE_BACKEND["en"] = self._english_backend

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self._device = "cuda:0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"
        logger.info(
            "TTSService initialized – device=%s, qwen_model=%s, vieneu_mode=%s",
            self._device,
            self._qwen_model_path,
            self._vieneu_mode,
        )

    # -----------------------------------------------------------------------
    # Public async API
    # -----------------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        language: str = "zh",
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize text to speech and return WAV audio bytes.

        Automatically selects the appropriate backend based on the target
        language. For Qwen3-TTS, `voice` selects a preset speaker. For
        VieNeu-TTS, `voice` selects a preset voice by name.

        Args:
            text: The text to synthesize.
            language: Language code ("zh", "vi", "en", etc.). Determines
                which backend is used.
            voice: Optional preset voice/speaker name.
            instruct: Optional style instruction for Qwen3-TTS VoiceDesign
                or CustomVoice models.
            **gen_kwargs: Additional generation parameters forwarded to the
                underlying model (e.g., temperature, top_k, max_new_tokens).

        Returns:
            SynthesisResult containing WAV audio bytes.

        Raises:
            RuntimeError: If the selected backend fails to load or synthesize.
        """
        loop = asyncio.get_running_loop()

        def _run() -> SynthesisResult:
            backend = self._select_backend(language)
            if backend == Backend.QWEN:
                return self._synthesize_qwen(text, language, voice, instruct, **gen_kwargs)
            else:
                return self._synthesize_vieneu(text, voice, **gen_kwargs)

        return await loop.run_in_executor(None, _run)

    async def synthesize_with_clone(
        self,
        text: str,
        language: str,
        voice_prompt: VoiceClonePrompt,
        ref_text: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize text using a cloned voice.

        Args:
            text: The text to synthesize.
            language: Language code for the synthesis.
            voice_prompt: Pre-built VoiceClonePrompt from
                `create_voice_clone_prompt()` or `get_cached_clone_prompt()`.
            ref_text: Reference transcript (required for ICL mode with
                VieNeu-TTS, optional for Qwen3-TTS when already in prompt).
            **gen_kwargs: Additional generation parameters.

        Returns:
            SynthesisResult containing WAV audio bytes.
        """
        loop = asyncio.get_running_loop()

        def _run() -> SynthesisResult:
            if voice_prompt.backend == Backend.QWEN:
                return self._synthesize_qwen_clone(
                    text, language, voice_prompt, **gen_kwargs
                )
            else:
                return self._synthesize_vieneu_clone(
                    text, voice_prompt, ref_text, **gen_kwargs
                )

        return await loop.run_in_executor(None, _run)

    async def stream_synthesize(
        self,
        text: str,
        language: str = "zh",
        voice: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Stream synthesized audio as chunks of WAV data.

        Yields raw PCM audio chunks (16-bit, 24 kHz, mono) for immediate
        playback. Note: Qwen3-TTS does not support true streaming yet,
        so it returns a single chunk. VieNeu-TTS streams in real chunks.

        Args:
            text: The text to synthesize.
            language: Language code.
            voice: Optional preset voice/speaker name.
            **gen_kwargs: Additional generation parameters.

        Yields:
            bytes: Raw PCM audio chunks (16-bit, 24 kHz, mono).
        """
        backend = self._select_backend(language)

        if backend == Backend.QWEN:
            # Qwen3-TTS does not support true streaming; return single chunk
            result = await self.synthesize(text, language, voice, **gen_kwargs)
            wav_data = result.audio_bytes
            yield wav_data
        else:
            # VieNeu-TTS supports real streaming
            async for chunk in self._stream_vieneu(text, voice, **gen_kwargs):
                yield chunk

    async def create_voice_clone_prompt(
        self,
        ref_audio: bytes,
        language: str = "zh",
        ref_text: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> VoiceClonePrompt:
        """Build a reusable voice clone prompt from reference audio.

        The prompt is cached internally for subsequent calls to
        `synthesize_with_clone()`. For Qwen3-TTS, use `create_voice_clone_prompt`
        internally. For VieNeu-TTS, use `encode_reference`.

        Args:
            ref_audio: Reference audio bytes (WAV format, 3–5 seconds).
            language: Language code to determine which backend to use.
            ref_text: Transcript of the reference audio (required for Qwen3-TTS
                ICL mode, optional for VieNeu-TTS v2 Turbo).
            cache_key: Optional key to cache this prompt. If not provided,
                an MD5 hash of the audio is used.

        Returns:
            VoiceClonePrompt that can be passed to `synthesize_with_clone()`.
        """
        loop = asyncio.get_running_loop()

        def _run() -> VoiceClonePrompt:
            backend = self._select_backend(language)
            if backend == Backend.QWEN:
                prompt = self._create_qwen_clone_prompt(ref_audio, ref_text)
            else:
                prompt = self._create_vieneu_clone_prompt(ref_audio)

            key = cache_key or _audio_hash(ref_audio)
            self._voice_clone_cache[key] = prompt
            logger.info("Cached voice clone prompt with key=%s", key)
            return prompt

        return await loop.run_in_executor(None, _run)

    def get_cached_clone_prompt(self, cache_key: str) -> Optional[VoiceClonePrompt]:
        """Retrieve a cached voice clone prompt by key.

        Args:
            cache_key: The cache key used when creating the prompt.

        Returns:
            The cached VoiceClonePrompt, or None if not found.
        """
        return self._voice_clone_cache.get(cache_key)

    async def list_vieneu_voices(self) -> list[tuple[str, str]]:
        """List preset voices available in VieNeu-TTS.

        Returns:
            List of (description, voice_id) tuples.
        """
        loop = asyncio.get_running_loop()
        vieneu = self._ensure_vieneu()

        def _run() -> list[tuple[str, str]]:
            return vieneu.list_preset_voices()

        return await loop.run_in_executor(None, _run)

    async def list_qwen_speakers(self) -> Optional[list[str]]:
        """List preset speakers available in Qwen3-TTS CustomVoice model.

        Returns:
            Sorted list of speaker names, or None if not available.
        """
        loop = asyncio.get_running_loop()
        qwen = self._ensure_qwen()

        def _run() -> Optional[list[str]]:
            return qwen.get_supported_speakers()

        return await loop.run_in_executor(None, _run)

    async def close(self) -> None:
        """Release resources held by both backends."""
        if self._vieneu_tts is not None:
            try:
                self._vieneu_tts.close()
            except Exception as e:
                logger.warning("Error closing VieNeu-TTS: %s", e)
            self._vieneu_tts = None
        self._qwen_model = None
        self._voice_clone_cache.clear()
        logger.info("TTSService closed.")

    # -----------------------------------------------------------------------
    # Backend selection
    # -----------------------------------------------------------------------

    def _select_backend(self, language: str) -> Backend:
        """Select the appropriate backend for a given language code."""
        lang = language.lower()
        backend = LANGUAGE_BACKEND.get(lang)
        if backend is None:
            # Default to Qwen3-TTS for unknown languages (supports 10 languages)
            logger.warning(
                "Unknown language '%s', defaulting to Qwen3-TTS", language
            )
            return Backend.QWEN
        return backend

    # -----------------------------------------------------------------------
    # Qwen3-TTS internal methods
    # -----------------------------------------------------------------------

    def _ensure_qwen(self) -> Any:
        """Lazy-load Qwen3TTSModel."""
        if self._qwen_model is not None:
            return self._qwen_model

        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise ImportError(
                "Qwen3-TTS is not installed. Install it with: pip install qwen-tts"
            ) from e

        logger.info("Loading Qwen3-TTS model: %s (device=%s)", self._qwen_model_path, self._device)

        attn_impl = (
            "flash_attention_2"
            if self._use_flash_attention
            else None
        )

        kwargs: dict[str, Any] = {
            "device_map": self._device,
            "dtype": torch.bfloat16,
        }
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        try:
            self._qwen_model = Qwen3TTSModel.from_pretrained(
                self._qwen_model_path,
                **kwargs,
            )
            logger.info("Qwen3-TTS model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load Qwen3-TTS model: %s", e)
            raise RuntimeError(f"Failed to load Qwen3-TTS model: {e}") from e

        return self._qwen_model

    def _synthesize_qwen(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        instruct: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize using Qwen3-TTS backend."""
        model = self._ensure_qwen()
        qwen_lang = QWEN_LANGUAGE_MAP.get(language.lower(), "Auto")

        # Determine which generation method to use based on model type
        model_type = getattr(model.model, "tts_model_type", "base")

        if model_type == "custom_voice":
            speaker = voice or "Vivian"
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=qwen_lang,
                speaker=speaker,
                instruct=instruct or "",
                **gen_kwargs,
            )
        elif model_type == "voice_design":
            wavs, sr = model.generate_voice_design(
                text=text,
                language=qwen_lang,
                instruct=instruct or "",
                **gen_kwargs,
            )
        else:
            # Base model – use voice clone with default preset or minimal ref
            wavs, sr = self._synthesize_qwen_clone_default(
                text, qwen_lang, voice, **gen_kwargs
            )

        audio = _ensure_mono(wavs[0])
        audio = _resample_if_needed(audio, sr)

        return SynthesisResult(
            audio_bytes=_numpy_to_wav_bytes(audio, TARGET_SAMPLE_RATE),
            sample_rate=TARGET_SAMPLE_RATE,
            duration_seconds=len(audio) / TARGET_SAMPLE_RATE,
            backend=Backend.QWEN,
        )

    def _synthesize_qwen_clone_default(
        self,
        text: str,
        language: str,
        voice: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        """Fallback: synthesize with Qwen3-TTS Base model using x_vector_only.

        When no reference audio is provided, this uses the model's default
        voice cloning with x_vector_only_mode=True.
        """
        model = self._ensure_qwen()

        # Use a minimal internal reference for cloning
        # When no ref_audio is provided, we rely on x_vector_only_mode
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            x_vector_only_mode=True,
            **gen_kwargs,
        )
        return wavs, sr

    def _synthesize_qwen_clone(
        self,
        text: str,
        language: str,
        voice_prompt: VoiceClonePrompt,
        **gen_kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize using Qwen3-TTS with voice cloning."""
        model = self._ensure_qwen()
        qwen_lang = QWEN_LANGUAGE_MAP.get(language.lower(), "Auto")

        prompt_items = voice_prompt.qwen_prompt
        if prompt_items is None:
            raise ValueError("VoiceClonePrompt has no Qwen prompt data.")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=qwen_lang,
            voice_clone_prompt=prompt_items,
            **gen_kwargs,
        )

        audio = _ensure_mono(wavs[0])
        audio = _resample_if_needed(audio, sr)

        return SynthesisResult(
            audio_bytes=_numpy_to_wav_bytes(audio, TARGET_SAMPLE_RATE),
            sample_rate=TARGET_SAMPLE_RATE,
            duration_seconds=len(audio) / TARGET_SAMPLE_RATE,
            backend=Backend.QWEN,
        )

    def _create_qwen_clone_prompt(
        self,
        ref_audio: bytes,
        ref_text: Optional[str] = None,
    ) -> VoiceClonePrompt:
        """Create a voice clone prompt for Qwen3-TTS."""
        model = self._ensure_qwen()

        # Write audio to a temporary tuple for the model
        audio_np, sr = sf.read(io.BytesIO(ref_audio), dtype="float32", always_2d=False)
        audio_np = _ensure_mono(audio_np)

        prompt_items = model.create_voice_clone_prompt(
            ref_audio=(audio_np, sr),
            ref_text=ref_text,
            x_vector_only_mode=(ref_text is None or ref_text == ""),
        )

        return VoiceClonePrompt(
            backend=Backend.QWEN,
            qwen_prompt=prompt_items,
            ref_text=ref_text,
        )

    # -----------------------------------------------------------------------
    # VieNeu-TTS internal methods
    # -----------------------------------------------------------------------

    def _ensure_vieneu(self) -> Any:
        """Lazy-load VieNeu-TTS."""
        if self._vieneu_tts is not None:
            return self._vieneu_tts

        try:
            from vieneu import Vieneu
        except ImportError as e:
            raise ImportError(
                "VieNeu-TTS is not installed. Install it with: pip install vieneu"
            ) from e

        logger.info("Loading VieNeu-TTS in mode: %s", self._vieneu_mode)

        mode = self._vieneu_mode
        # Map our mode names to VieNeu's expected modes
        if mode == "turbo":
            vieneu_kwargs: dict[str, Any] = {"mode": "standard"}
        elif mode == "gpu":
            vieneu_kwargs = {
                "mode": "standard",
                "backbone_repo": "pnnbao-ump/VieNeu-TTS-0.3B",
                "backbone_device": "cuda",
            }
        else:
            vieneu_kwargs = {"mode": mode}

        try:
            self._vieneu_tts = Vieneu(**vieneu_kwargs)
            logger.info("VieNeu-TTS loaded successfully.")
        except Exception as e:
            logger.error("Failed to load VieNeu-TTS: %s", e)
            raise RuntimeError(f"Failed to load VieNeu-TTS: {e}") from e

        return self._vieneu_tts

    def _synthesize_vieneu(
        self,
        text: str,
        voice: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize using VieNeu-TTS backend."""
        vieneu = self._ensure_vieneu()

        voice_data = None
        if voice:
            try:
                voice_data = vieneu.get_preset_voice(voice)
            except Exception as e:
                logger.warning("Could not load preset voice '%s': %s", voice, e)

        infer_kwargs: dict[str, Any] = {}
        if voice_data is not None:
            infer_kwargs["voice"] = voice_data

        # Filter gen_kwargs to only pass supported params
        supported_kwargs = {
            "temperature", "top_k", "max_chars", "skip_normalize"
        }
        for k, v in gen_kwargs.items():
            if k in supported_kwargs:
                infer_kwargs[k] = v

        audio_np = vieneu.infer(text=text, **infer_kwargs)
        audio_np = _ensure_mono(audio_np)
        # VieNeu-TTS outputs at 24 kHz natively

        return SynthesisResult(
            audio_bytes=_numpy_to_wav_bytes(audio_np, TARGET_SAMPLE_RATE),
            sample_rate=TARGET_SAMPLE_RATE,
            duration_seconds=len(audio_np) / TARGET_SAMPLE_RATE,
            backend=Backend.VIENEU,
        )

    def _synthesize_vieneu_clone(
        self,
        text: str,
        voice_prompt: VoiceClonePrompt,
        ref_text: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> SynthesisResult:
        """Synthesize using VieNeu-TTS with voice cloning."""
        vieneu = self._ensure_vieneu()

        infer_kwargs: dict[str, Any] = {}

        if voice_prompt.vieneu_codes is not None:
            infer_kwargs["ref_codes"] = voice_prompt.vieneu_codes
        if ref_text or voice_prompt.ref_text:
            infer_kwargs["ref_text"] = ref_text or voice_prompt.ref_text

        # Filter supported kwargs
        supported_kwargs = {
            "temperature", "top_k", "max_chars", "skip_normalize"
        }
        for k, v in gen_kwargs.items():
            if k in supported_kwargs:
                infer_kwargs[k] = v

        audio_np = vieneu.infer(text=text, **infer_kwargs)
        audio_np = _ensure_mono(audio_np)

        return SynthesisResult(
            audio_bytes=_numpy_to_wav_bytes(audio_np, TARGET_SAMPLE_RATE),
            sample_rate=TARGET_SAMPLE_RATE,
            duration_seconds=len(audio_np) / TARGET_SAMPLE_RATE,
            backend=Backend.VIENEU,
        )

    async def _stream_vieneu(
        self,
        text: str,
        voice: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> AsyncIterator[bytes]:
        """Stream VieNeu-TTS output as raw PCM chunks.

        Yields 16-bit PCM bytes at 24 kHz mono.
        """
        vieneu = self._ensure_vieneu()

        voice_data = None
        if voice:
            try:
                voice_data = vieneu.get_preset_voice(voice)
            except Exception as e:
                logger.warning("Could not load preset voice '%s': %s", voice, e)

        stream_kwargs: dict[str, Any] = {}
        if voice_data is not None:
            stream_kwargs["voice"] = voice_data

        supported_kwargs = {
            "temperature", "top_k", "max_chars", "skip_normalize"
        }
        for k, v in gen_kwargs.items():
            if k in supported_kwargs:
                stream_kwargs[k] = v

        loop = asyncio.get_running_loop()

        # Run the synchronous streaming generator in a thread
        def _stream_generator() -> list[bytes]:
            chunks: list[bytes] = []
            for chunk_np in vieneu.infer_stream(text=text, **stream_kwargs):
                chunk_np = _ensure_mono(chunk_np)
                # Convert float32 [-1, 1] to int16 PCM
                pcm = np.clip(chunk_np * 32767, -32768, 32767).astype(np.int16)
                chunks.append(pcm.tobytes())
            return chunks

        chunks = await loop.run_in_executor(None, _stream_generator)
        for chunk in chunks:
            yield chunk

    def __enter__(self) -> "TTSService":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Sync close; async callers should use `await tts.close()`
        if self._vieneu_tts is not None:
            try:
                self._vieneu_tts.close()
            except Exception:
                pass
            self._vieneu_tts = None
        self._qwen_model = None
        self._voice_clone_cache.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_hash(audio_bytes: bytes) -> str:
    """Compute a simple hash of audio bytes for caching."""
    import hashlib

    return hashlib.sha256(audio_bytes).hexdigest()[:16]
