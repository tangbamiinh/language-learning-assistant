"""
Qwen3-ASR Speech-to-Text wrapper service.

Supports both local GPU inference (via `qwen-asr` package) and cloud API fallback
(DashScope). Provides a clean async API for transcribing Vietnamese (vi) and
Chinese (zh) audio, with auto-detection support.

Usage:
    # Local GPU inference (transformers backend)
    stt = STTService(backend="transformers")
    await stt.initialize()
    result = await stt.atranscribe("audio.wav")

    # Cloud API fallback (DashScope)
    stt = STTService(backend="dashscope")
    result = await stt.atranscribe("audio.wav")
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {"vi", "zh", "en", "yue"}
"""ISO 639-1 codes the service handles natively."""

LANGUAGE_DISPLAY_MAP: dict[str, str] = {
    "vi": "Vietnamese",
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
}
"""Maps ISO codes to Qwen3-ASR language display names."""


@dataclass
class TranscriptionResult:
    """Result of a single transcription."""

    text: str
    """The transcribed text."""

    language: str
    """Detected or forced language code (e.g. 'vi', 'zh', 'en')."""

    confidence: float = 0.0
    """Estimated confidence score (0.0-1.0). 0.0 when unknown."""

    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    """Raw result data from the underlying engine (for debugging)."""

    def __bool__(self) -> bool:
        return bool(self.text.strip())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class STTError(Exception):
    """Base exception for STT service errors."""


class STTInitializationError(STTError):
    """Raised when the STT model fails to initialize."""


class STTTranscriptionError(STTError):
    """Raised when transcription fails."""


class STTConfigurationError(STTError):
    """Raised when required configuration is missing."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class STTService:
    """
    Speech-to-Text service wrapping Qwen3-ASR-1.7B.

    Supports two backends:
    - ``"transformers"`` / ``"vllm"``: Local GPU inference via the ``qwen-asr`` package.
    - ``"dashscope"``: Cloud API fallback via Alibaba DashScope.

    The service lazily initializes the model on first call. Use
    :meth:`initialize` / :meth:`ainitialize` to warm-start explicitly.

    Args:
        backend: One of ``"transformers"``, ``"vllm"``, or ``"dashscope"``.
            When ``None`` (default) the service auto-selects: local transformers
            if a CUDA GPU is available, otherwise DashScope.
        model_path: HuggingFace model ID or local path for the ASR model.
        dashscope_api_key: DashScope API key. Falls back to
            ``DASHSCOPE_API_KEY`` env var.
        dashscope_model: DashScope model name (default ``qwen3-asr-flash``).
        max_batch_size: Maximum batch size for local inference.
        max_new_tokens: Maximum generated tokens per sample.
        dtype: Torch dtype for model weights.
        device_map: Device placement string for ``from_pretrained``.
        gpu_memory_utilization: GPU memory fraction for vLLM backend (0-1).
    """

    def __init__(
        self,
        backend: str | None = None,
        model_path: str | None = None,
        dashscope_api_key: str | None = None,
        dashscope_model: str = "qwen3-asr-flash",
        max_batch_size: int = 32,
        max_new_tokens: int = 256,
        dtype: str = "bfloat16",
        device_map: str = "cuda:0",
        gpu_memory_utilization: float = 0.7,
    ) -> None:
        # --- Resolve backend ------------------------------------------------
        self._resolved_backend = self._resolve_backend(backend)
        logger.info("STT backend: %s", self._resolved_backend)

        # --- DashScope config -----------------------------------------------
        self._dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._dashscope_model = dashscope_model
        self._dashscope_base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/api/v1",
        )

        # --- Local model config ---------------------------------------------
        self._model_path = model_path or os.getenv(
            "QWEN_ASR_MODEL_PATH", "Qwen/Qwen3-ASR-1.7B"
        )
        self._max_batch_size = max_batch_size
        self._max_new_tokens = max_new_tokens
        self._dtype = dtype
        self._device_map = device_map
        self._gpu_memory_utilization = gpu_memory_utilization

        # --- Runtime state --------------------------------------------------
        self._model: Any = None
        self._initialized = False

    # -----------------------------------------------------------------------
    # Public async API
    # -----------------------------------------------------------------------

    async def ainitialize(self) -> None:
        """Warm-initialize the underlying model (loads weights into memory).

        Call this once at startup to avoid cold-start latency on the first
        transcription request.
        """
        await asyncio.get_event_loop().run_in_executor(None, self._ensure_initialized)

    async def atranscribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file (async).

        Args:
            audio_path: Local file path or URL to the audio file.
            language: Optional ISO 639-1 language code to force.
                Pass ``None`` for automatic detection.

        Returns:
            :class:`TranscriptionResult` with ``text``, ``language``, and
            ``confidence``.

        Raises:
            STTTranscriptionError: If transcription fails.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.transcribe, audio_path, language
        )
        return result

    async def atranscribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe raw audio bytes (async).

        Args:
            audio_bytes: Raw PCM audio data (mono, float32).
            sample_rate: Sample rate of the audio data.
            language: Optional ISO 639-1 language code to force.

        Returns:
            :class:`TranscriptionResult`.

        Raises:
            STTTranscriptionError: If transcription fails.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.transcribe_bytes, audio_bytes, sample_rate, language
        )
        return result

    async def abatch_transcribe(
        self,
        audio_paths: list[str],
        languages: list[str | None] | None = None,
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files in a single batch (async).

        Args:
            audio_paths: List of local file paths or URLs.
            languages: Optional list of language codes (same length as
                ``audio_paths``). Pass ``None`` for auto-detection on all.

        Returns:
            List of :class:`TranscriptionResult`, one per input.

        Raises:
            STTTranscriptionError: If batch transcription fails.
        """
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self.batch_transcribe, audio_paths, languages
        )
        return results

    # -----------------------------------------------------------------------
    # Public sync API
    # -----------------------------------------------------------------------

    def initialize(self) -> None:
        """Warm-initialize the underlying model (sync version)."""
        self._ensure_initialized()

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe an audio file (sync).

        Args:
            audio_path: Local file path or URL to the audio file.
            language: Optional ISO 639-1 language code to force.

        Returns:
            :class:`TranscriptionResult`.
        """
        self._ensure_initialized()
        lang_display = self._resolve_language(language)
        logger.debug(
            "Transcribing %s (language=%s, backend=%s)",
            audio_path,
            language,
            self._resolved_backend,
        )

        try:
            if self._resolved_backend == "dashscope":
                return self._transcribe_dashscope(audio_path, lang_display)
            else:
                results = self._model.transcribe(
                    audio=audio_path,
                    language=lang_display,
                    return_time_stamps=False,
                )
                return self._parse_local_result(results[0], language)
        except STTError:
            raise
        except Exception as exc:
            raise STTTranscriptionError(
                f"Transcription failed for {audio_path}: {exc}"
            ) from exc

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        language: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe raw audio bytes (sync).

        Args:
            audio_bytes: Raw PCM audio data (mono, float32).
            sample_rate: Sample rate of the audio data.
            language: Optional ISO 639-1 language code to force.

        Returns:
            :class:`TranscriptionResult`.
        """
        self._ensure_initialized()
        lang_display = self._resolve_language(language)

        try:
            if self._resolved_backend == "dashscope":
                # Write bytes to a temporary WAV for DashScope
                return self._transcribe_bytes_dashscope(
                    audio_bytes, sample_rate, lang_display
                )
            else:
                # Convert bytes to (np.ndarray, sr) tuple accepted by qwen-asr
                wav = self._bytes_to_wav_array(audio_bytes, sample_rate)
                results = self._model.transcribe(
                    audio=(wav, sample_rate),
                    language=lang_display,
                    return_time_stamps=False,
                )
                return self._parse_local_result(results[0], language)
        except STTError:
            raise
        except Exception as exc:
            raise STTTranscriptionError(
                f"Transcription of audio bytes failed: {exc}"
            ) from exc

    def batch_transcribe(
        self,
        audio_paths: list[str],
        languages: list[str | None] | None = None,
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files in a single batch (sync).

        Args:
            audio_paths: List of local file paths or URLs.
            languages: Optional list of language codes.

        Returns:
            List of :class:`TranscriptionResult`.
        """
        if not audio_paths:
            return []

        self._ensure_initialized()

        lang_display_list: list[str | None]
        if languages is None:
            lang_display_list = [None] * len(audio_paths)
        else:
            lang_display_list = [
                self._resolve_language(lang) for lang in languages
            ]

        logger.debug(
            "Batch transcribing %d files (backend=%s)",
            len(audio_paths),
            self._resolved_backend,
        )

        try:
            if self._resolved_backend == "dashscope":
                # DashScope doesn't support true batching; serialize.
                results: list[TranscriptionResult] = []
                for path, lang in zip(audio_paths, languages or [None] * len(audio_paths)):
                    results.append(self.transcribe(path, lang))
                return results
            else:
                raw_results = self._model.transcribe(
                    audio=audio_paths,
                    language=lang_display_list,
                    return_time_stamps=False,
                )
                return [
                    self._parse_local_result(raw, lang)
                    for raw, lang in zip(
                        raw_results, languages or [None] * len(audio_paths)
                    )
                ]
        except STTError:
            raise
        except Exception as exc:
            raise STTTranscriptionError(
                f"Batch transcription failed ({len(audio_paths)} files): {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """Whether the underlying model has been loaded."""
        return self._initialized

    @property
    def backend(self) -> str:
        """The active backend name."""
        return self._resolved_backend

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _resolve_backend(self, requested: str | None) -> str:
        """Determine which backend to use."""
        if requested is None:
            if torch.cuda.is_available():
                return "transformers"
            return "dashscope"
        backend = requested.lower()
        if backend in ("transformers", "vllm"):
            return backend
        if backend == "dashscope":
            return "dashscope"
        raise STTConfigurationError(
            f"Unknown backend '{requested}'. "
            f"Choose from: transformers, vllm, dashscope"
        )

    def _resolve_language(self, language: str | None) -> str | None:
        """Convert ISO 639-1 code to Qwen3-ASR display name, or pass through."""
        if language is None:
            return None
        lang = language.lower()
        if lang in LANGUAGE_DISPLAY_MAP:
            return LANGUAGE_DISPLAY_MAP[lang]
        # Pass through in case it's already a display name
        return language

    def _ensure_initialized(self) -> None:
        """Load the model if not already loaded (thread-safe via simple flag)."""
        if self._initialized:
            return

        if self._resolved_backend == "dashscope":
            self._init_dashscope()
        else:
            if self._resolved_backend == "vllm":
                self._init_vllm()
            else:
                self._init_transformers()

        self._initialized = True
        logger.info("STT service initialized (backend=%s)", self._resolved_backend)

    # -----------------------------------------------------------------------
    # Local backend initialization
    # -----------------------------------------------------------------------

    def _init_transformers(self) -> None:
        """Initialize local model via transformers backend."""
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise STTInitializationError(
                "qwen-asr package is not installed. "
                "Run: pip install qwen-asr"
            ) from exc

        if not torch.cuda.is_available():
            logger.warning(
                "No CUDA GPU detected. Falling back to DashScope backend. "
                "Set DASHSCOPE_API_KEY or request backend='dashscope' explicitly."
            )
            self._resolved_backend = "dashscope"
            self._init_dashscope()
            return

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype, torch.bfloat16)

        logger.info(
            "Loading Qwen3-ASR model '%s' (transformers, dtype=%s)",
            self._model_path,
            self._dtype,
        )

        self._model = Qwen3ASRModel.from_pretrained(
            self._model_path,
            dtype=torch_dtype,
            device_map=self._device_map,
            max_inference_batch_size=self._max_batch_size,
            max_new_tokens=self._max_new_tokens,
        )

    def _init_vllm(self) -> None:
        """Initialize local model via vLLM backend."""
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise STTInitializationError(
                "qwen-asr[vllm] package is not installed. "
                "Run: pip install 'qwen-asr[vllm]'"
            ) from exc

        if not torch.cuda.is_available():
            logger.warning(
                "No CUDA GPU detected. Falling back to DashScope backend."
            )
            self._resolved_backend = "dashscope"
            self._init_dashscope()
            return

        logger.info(
            "Loading Qwen3-ASR model '%s' (vLLM, gpu_mem=%.1f)",
            self._model_path,
            self._gpu_memory_utilization,
        )

        self._model = Qwen3ASRModel.LLM(
            model=self._model_path,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_inference_batch_size=self._max_batch_size,
            max_new_tokens=self._max_new_tokens,
        )

    # -----------------------------------------------------------------------
    # DashScope backend initialization
    # -----------------------------------------------------------------------

    def _init_dashscope(self) -> None:
        """Validate DashScope configuration."""
        if not self._dashscope_api_key:
            raise STTConfigurationError(
                "DASHSCOPE_API_KEY environment variable is required for "
                "DashScope backend. Obtain a key at "
                "https://bailian.console.aliyun.com/"
            )
        # dashscope library is imported lazily below
        self._model = None  # DashScope uses stateless HTTP calls

    # -----------------------------------------------------------------------
    # DashScope transcription
    # -----------------------------------------------------------------------

    def _transcribe_dashscope(
        self, audio_path: str, language: str | None
    ) -> TranscriptionResult:
        """Transcribe via DashScope MultimodalConversation API."""
        import dashscope  # noqa: PLC0415

        dashscope.api_key = self._dashscope_api_key
        dashscope.base_http_api_url = self._dashscope_base_url

        # Determine audio input format
        audio_input = audio_path
        if Path(audio_path).exists():
            audio_input = f"file://{Path(audio_path).resolve()}"

        messages = [{"role": "user", "content": [{"audio": audio_input}]}]

        asr_options: dict[str, Any] = {"enable_itn": False}
        if language:
            asr_options["language"] = language

        response = dashscope.MultiModalConversation.call(
            api_key=self._dashscope_api_key,
            model=self._dashscope_model,
            messages=messages,
            result_format="message",
            asr_options=asr_options,
        )

        return self._parse_dashscope_response(response)

    def _transcribe_bytes_dashscope(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        language: str | None,
    ) -> TranscriptionResult:
        """Transcribe raw bytes via DashScope (writes temp WAV)."""
        # Convert raw PCM bytes to WAV and write to temp file
        wav_array = self._bytes_to_wav_array(audio_bytes, sample_rate)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, wav_array, sample_rate, subtype="FLOAT")
            tmp_path = tmp.name

        try:
            return self._transcribe_dashscope(tmp_path, language)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _parse_dashscope_response(self, response: Any) -> TranscriptionResult:
        """Extract transcription text from DashScope response."""
        raw_dict = response.__dict__ if hasattr(response, "__dict__") else {}

        # Navigate the response structure
        text = ""
        detected_lang = None

        try:
            # response.output.choices[0].message.content
            output = getattr(response, "output", None)
            if output is not None:
                choices = getattr(output, "choices", None) or getattr(output, "text", None)
                if choices and isinstance(choices, list) and choices:
                    choice = choices[0]
                    message = getattr(choice, "message", None)
                    if message:
                        content = getattr(message, "content", "")
                        if isinstance(content, list):
                            # Content may be a list of dicts with "text" key
                            for item in content:
                                if isinstance(item, dict) and "text" in item:
                                    text += item["text"]
                                elif isinstance(item, str):
                                    text += item
                        elif isinstance(content, str):
                            text = content
                elif isinstance(choices, str):
                    text = choices
        except Exception as parse_err:
            logger.warning("Failed to parse DashScope response: %s", parse_err)

        if not text:
            raise STTTranscriptionError(
                "Empty transcription result from DashScope API. "
                f"Response status: {getattr(response, 'status_code', 'unknown')}"
            )

        # Detect language from the text heuristically if not provided
        if not detected_lang:
            detected_lang = self._heuristic_language_detect(text)

        return TranscriptionResult(
            text=text.strip(),
            language=detected_lang,
            confidence=0.0,  # DashScope doesn't expose per-token confidence
            raw=raw_dict,
        )

    # -----------------------------------------------------------------------
    # Local result parsing
    # -----------------------------------------------------------------------

    def _parse_local_result(
        self, result: Any, requested_lang: str | None
    ) -> TranscriptionResult:
        """Parse a single result object from qwen-asr."""
        text = getattr(result, "text", "") or ""
        detected_lang = getattr(result, "language", "") or ""

        # Map display name back to ISO code
        iso_lang = self._display_name_to_iso(detected_lang)

        # Use requested language if detection was unclear
        if not iso_lang and requested_lang:
            iso_lang = requested_lang.lower()
        elif not iso_lang:
            iso_lang = self._heuristic_language_detect(text)

        return TranscriptionResult(
            text=text.strip(),
            language=iso_lang,
            confidence=0.0,  # qwen-asr doesn't expose confidence scores
            raw={"text": text, "language": detected_lang},
        )

    @staticmethod
    def _display_name_to_iso(display_name: str) -> str:
        """Map Qwen3-ASR language display name back to ISO 639-1 code."""
        reverse_map = {v: k for k, v in LANGUAGE_DISPLAY_MAP.items()}
        if not display_name:
            return ""
        # Try exact match first, then case-insensitive
        if display_name in reverse_map:
            return reverse_map[display_name]
        lower = display_name.lower()
        for key, val in reverse_map.items():
            if val.lower() == lower:
                return key
        # Pass through if it's already an ISO code
        if display_name.lower() in SUPPORTED_LANGUAGES:
            return display_name.lower()
        return display_name.lower()

    @staticmethod
    def _heuristic_language_detect(text: str) -> str:
        """Very simple language detection based on character ranges."""
        if not text:
            return ""
        # Check for Chinese characters (CJK Unified Ideographs)
        chinese_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        vietnamese_chars = sum(
            1 for c in text if "\u0102" <= c <= "\u017F" or c in "àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỷ"
        )

        total_chars = len(text.rstrip())
        if total_chars == 0:
            return ""

        if chinese_chars > total_chars * 0.3:
            return "zh"
        if vietnamese_chars > total_chars * 0.1:
            return "vi"
        return "en"  # Default fallback

    @staticmethod
    def _bytes_to_wav_array(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        """Convert raw PCM bytes to a numpy WAV array."""
        try:
            # Try to read as an existing audio format first
            with io.BytesIO(audio_bytes) as buf:
                wav, _ = sf.read(buf, dtype="float32", always_2d=False)
            return np.asarray(wav, dtype=np.float32)
        except Exception:
            # Assume raw PCM float32
            return np.frombuffer(audio_bytes, dtype=np.float32).copy()

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"STTService(backend={self._resolved_backend!r}, {status})"
