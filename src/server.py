"""Phase 1 Bridge Server for the Language Learning Assistant.

FastAPI server that connects the Next.js web app to:
- LLM: Qwen3.5-9B via OpenRouter (OpenAI-compatible API)
- STT: Qwen3-ASR-1.7B for Vietnamese/Chinese speech recognition
- TTS: Qwen3-TTS (Chinese) + VieNeu-TTS (Vietnamese) for voice synthesis

The server receives requests from the Next.js web app's API routes, manages
in-memory conversation history per session, and injects relevant skill context
(cognates, pronunciation, grammar) into LLM prompts.

Environment variables:
    OPENROUTER_API_KEY: Required. API key for OpenRouter.
    PYTHON_API_PORT: Optional. Port to run the server on. Default: 8000.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_VOICE, build_cognate_prompt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Project root – server.py lives in src/, so go up one level
PROJECT_ROOT = Path(__file__).parent.parent
COGNATES_PATH = PROJECT_ROOT / "src" / "cognates.json"

# Conversation history limits
MAX_MESSAGES_PER_SESSION = 20

# LLM model
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.5-9b")

# Vietnamese diacritic characters for language detection
VIETNAMESE_CHARS = set(
    "àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩị"
    "òóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỷ"
)

# Vietnamese keywords that indicate pronunciation questions
PRONUNCIATION_KEYWORDS = {
    "phát âm", "đọc", "thanh", "tone", "âm", "giọng", "pinyin",
    "biến điệu", "đánh vần", "cách đọc", "phát âm",
}

# Vietnamese keywords that indicate grammar questions
GRAMMAR_KEYWORDS = {
    "ngữ pháp", "cấu trúc", "cú pháp", "cách dùng", "lúc nào",
    "như thế nào", "cách nói", "ngữ pháp", "grammar",
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint."""

    message: str = Field(..., min_length=1, description="User message text")
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (vi, zh, en). Auto-detected if not provided.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier. Auto-generated if not provided.",
    )


class ChatResponse(BaseModel):
    """Response body for the /api/chat endpoint."""

    reply: str = Field(..., description="Assistant reply text")
    pinyin: str | None = Field(default=None, description="Pinyin of key Chinese words, if applicable")
    vietnamese: str | None = Field(default=None, description="Vietnamese translation, if applicable")
    chinese: str | None = Field(default=None, description="Chinese characters, if applicable")
    language: str = Field(..., description="Detected or requested language code")


class VoiceRequest(BaseModel):
    """Request body for the /api/voice endpoint."""

    audio_base64: str = Field(..., min_length=1, description="Base64-encoded audio data")
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code. Auto-detected if not provided.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session identifier. Auto-generated if not provided.",
    )


class VoiceResponse(BaseModel):
    """Response body for the /api/voice endpoint."""

    transcript: str = Field(..., description="STT transcription of the audio")
    reply: str = Field(..., description="Assistant reply text")
    audio_base64: str | None = Field(
        default=None,
        description="Base64-encoded TTS audio. Null if TTS is unavailable.",
    )
    language: str = Field(..., description="Detected or requested language code")


class CognateRequest(BaseModel):
    """Request body for the /api/cognates endpoint."""

    word: str = Field(..., min_length=1, description="Vietnamese word to look up")


class CognateResponse(BaseModel):
    """Response body for the /api/cognates endpoint."""

    cognate: dict[str, Any] | None = Field(
        default=None,
        description="Exact match in the cognates database, or null",
    )
    related: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Fuzzy/partial matches (up to 5)",
    )


class HealthResponse(BaseModel):
    """Response body for the /api/health endpoint."""

    status: str = Field(default="ok", description="Server status")
    stt_available: bool = Field(default=False, description="Whether STT service is available")
    tts_available: bool = Field(default=False, description="Whether TTS service is available")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# OpenRouter client (initialized lazily)
_client: AsyncOpenAI | None = None

# In-memory conversation history: session_id → list of chat messages
sessions: dict[str, list[dict[str, str]]] = {}

# Cognates loaded from JSON
_cognates: list[dict[str, Any]] = []

# STT / TTS services (lazy-loaded, may be None if GPU not available)
_stt_service: Any = None
_tts_service: Any = None
_stt_available: bool = False
_tts_available: bool = False


# ---------------------------------------------------------------------------
# Lazy initialization helpers
# ---------------------------------------------------------------------------


def _get_llm_client() -> AsyncOpenAI:
    """Get or create the OpenRouter LLM client.

    Returns:
        Configured AsyncOpenAI client pointing to OpenRouter.

    Raises:
        RuntimeError: If OPENROUTER_API_KEY is not set.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is required. "
            "Get your key at https://openrouter.ai/keys"
        )

    _client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    logger.info("OpenRouter client initialized")
    return _client


def load_cognates() -> list[dict[str, Any]]:
    """Load cognates from the JSON database.

    Returns:
        List of cognate dictionaries.
    """
    global _cognates
    if _cognates:
        return _cognates

    try:
        with open(COGNATES_PATH, encoding="utf-8") as f:
            _cognates = json.load(f)
        logger.info("Loaded %d cognates from %s", len(_cognates), COGNATES_PATH)
    except FileNotFoundError:
        logger.warning("Cognates file not found at %s. Running without cognate data.", COGNATES_PATH)
        _cognates = []
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse cognates JSON: %s", exc)
        _cognates = []

    return _cognates


def _detect_language(text: str) -> str:
    """Detect whether text is Vietnamese, Chinese, or English.

    Uses simple character-range heuristics.

    Args:
        text: Input text to analyze.

    Returns:
        ISO 639-1 language code: "vi", "zh", or "en".
    """
    if not text:
        return "en"

    # Check for Vietnamese diacritics
    vn_count = sum(1 for c in text if c in VIETNAMESE_CHARS)
    # Check for Chinese characters (CJK Unified Ideographs)
    zh_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    # Check for Latin alphabet (English/Vietnamese without accents)
    latin_count = sum(1 for c in text if c.isascii() and c.isalpha())

    total = len(text.strip())
    if total == 0:
        return "en"

    if vn_count > total * 0.05:
        return "vi"
    if zh_count > total * 0.2:
        return "zh"
    return "en"


def find_cognate_context(message: str) -> str:
    """Find relevant cognates for words in the user message.

    Searches the loaded cognates database for Vietnamese words that appear
    in the message and builds a context string for the LLM.

    Args:
        message: The user's message text.

    Returns:
        Formatted cognate context string, or empty string if no matches.
    """
    cognates = load_cognates()
    if not cognates:
        return ""

    message_lower = message.lower()
    matched: list[dict[str, Any]] = []

    for entry in cognates:
        vn = entry.get("vietnamese", "").lower()
        if vn and vn in message_lower:
            matched.append(entry)
            if len(matched) >= 5:
                break

    if not matched:
        return ""

    lines = ["\n\n### Relevant Hán Việt Cognates:\n"]
    for entry in matched:
        cn = entry.get("chinese_simplified", "")
        py = entry.get("pinyin", "")
        vn = entry.get("vietnamese", "")
        en = entry.get("english", "")
        lines.append(f"- {cn} ({py}) = {vn} — {en}")

    return "\n".join(lines)


def _detect_skill_context(message: str) -> str:
    """Detect which skill context to inject based on message content.

    Checks for pronunciation and grammar keywords in the message.

    Args:
        message: The user's message text.

    Returns:
        Skill context string to append to the system prompt.
    """
    message_lower = message.lower()
    contexts: list[str] = []

    # Check for pronunciation-related keywords
    if any(kw in message_lower for kw in PRONUNCIATION_KEYWORDS):
        contexts.append(
            "\n\n### Pronunciation Focus:\n"
            "The student is asking about pronunciation. Provide detailed pinyin "
            "with tone marks, explain tone contours compared to Vietnamese tones, "
            "and give minimal pairs if relevant. Be extra careful with tone 3 "
            "(dipping) and tone 4 (sharp falling)."
        )

    # Check for grammar-related keywords
    if any(kw in message_lower for kw in GRAMMAR_KEYWORDS):
        contexts.append(
            "\n\n### Grammar Focus:\n"
            "The student is asking about Chinese grammar. Explain the grammar point "
            "clearly in Vietnamese, compare to Vietnamese grammar where helpful, "
            "provide a formula, and give graduated examples (easy to challenging). "
            "Include practice exercises."
        )

    return "\n".join(contexts)


def build_system_prompt(mode: str = "text", extra_context: str = "") -> str:
    """Build the complete system prompt for the LLM.

    Combines the base system prompt with cognate context and skill context.

    Args:
        mode: "text" for chat mode, "voice" for voice mode.
        extra_context: Additional context from cognates and skill detection.

    Returns:
        Complete system prompt string.
    """
    base = SYSTEM_PROMPT_VOICE if mode == "voice" else SYSTEM_PROMPT
    if extra_context.strip():
        return base + extra_context
    return base


def _get_session(session_id: str) -> list[dict[str, str]]:
    """Get or create conversation history for a session.

    Args:
        session_id: The session identifier.

    Returns:
        List of chat messages for this session.
    """
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


def _add_message(session_id: str, role: str, content: str) -> None:
    """Add a message to the session history, trimming if needed.

    Args:
        session_id: The session identifier.
        role: "user" or "assistant".
        content: Message content.
    """
    history = _get_session(session_id)
    history.append({"role": role, "content": content})

    # Keep only the last MAX_MESSAGES_PER_SESSION messages
    if len(history) > MAX_MESSAGES_PER_SESSION:
        sessions[session_id] = history[-MAX_MESSAGES_PER_SESSION:]


async def _chat_with_llm(
    message: str,
    system_prompt: str,
    session_id: str,
    language: str,
) -> str:
    """Send a message to the LLM and return the assistant's reply.

    Manages conversation history and builds the appropriate prompt.

    Args:
        message: The user's message.
        system_prompt: The system prompt to use.
        session_id: The session identifier.
        language: The language code.

    Returns:
        The assistant's reply text.

    Raises:
        HTTPException: If the LLM call fails.
    """
    client = _get_llm_client()
    history = _get_session(session_id)

    # Build messages list
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Add conversation history (skip system-like entries, keep user/assistant)
    for msg in history:
        if msg["role"] in ("user", "assistant"):
            messages.append(msg)

    # Add current user message
    messages.append({"role": "user", "content": message})

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            extra_headers={
                "HTTP-Referer": "https://github.com/tangbamiinh/language-learning-assistant",
                "X-Title": "Language Learning Assistant",
            },
        )

        reply = response.choices[0].message.content or ""

        # Update history
        _add_message(session_id, "user", message)
        _add_message(session_id, "assistant", reply)

        return reply

    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to get response from LLM: {exc}",
        ) from exc


async def _ensure_stt() -> Any:
    """Lazy-load the STT service.

    Returns:
        Initialized STTService, or None if unavailable.
    """
    global _stt_service, _stt_available
    if _stt_available:
        return _stt_service

    try:
        from src.stt_service import STTService

        _stt_service = STTService()
        # Try to initialize – if it fails (no GPU, no API key), mark unavailable
        try:
            await _stt_service.ainitialize()
            _stt_available = True
            logger.info("STT service initialized successfully")
        except Exception as exc:
            logger.warning("STT initialization failed: %s. Text-only mode.", exc)
            _stt_available = False
            _stt_service = None
    except ImportError as exc:
        logger.warning("STT service not available: %s", exc)
        _stt_available = False

    return _stt_service


async def _ensure_tts() -> Any:
    """Lazy-load the TTS service.

    Returns:
        Initialized TTSService, or None if unavailable.
    """
    global _tts_service, _tts_available
    if _tts_available:
        return _tts_service

    try:
        from src.tts_service import TTSService

        _tts_service = TTSService()
        _tts_available = True
        logger.info("TTS service initialized successfully")
    except ImportError as exc:
        logger.warning("TTS service not available: %s", exc)
        _tts_available = False
        _tts_service = None

    return _tts_service


# ---------------------------------------------------------------------------
# FastAPI app lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    logger.info("Language Learning Assistant API starting up...")
    load_cognates()
    # Pre-check OpenRouter API key
    try:
        _get_llm_client()
    except RuntimeError as exc:
        logger.warning("OpenRouter not configured: %s", exc)

    yield

    logger.info("Shutting down API server. Active sessions: %d", len(sessions))
    sessions.clear()


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Language Learning Assistant API",
    description="Bridge server connecting Next.js web app to LLM, STT, and TTS services.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS – allow requests from Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Process a chat message through the LLM.

    Receives a user message, builds the system prompt with relevant context,
    manages conversation history, and returns the assistant's reply.

    Args:
        req: ChatRequest with message, optional language and session_id.

    Returns:
        ChatResponse with reply and structured data fields.

    Raises:
        HTTPException: If the message is empty or the LLM call fails.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Determine session ID
    session_id = req.session_id or str(uuid.uuid4())

    # Determine language
    language = req.language or _detect_language(req.message)

    # Build context
    cognate_context = find_cognate_context(req.message)
    skill_context = _detect_skill_context(req.message)
    extra_context = cognate_context + skill_context

    # Build system prompt
    system_prompt = build_system_prompt(mode="text", extra_context=extra_context)

    # Call LLM
    reply = await _chat_with_llm(
        message=req.message,
        system_prompt=system_prompt,
        session_id=session_id,
        language=language,
    )

    return ChatResponse(
        reply=reply,
        pinyin=None,
        vietnamese=None,
        chinese=None,
        language=language,
    )


@app.post("/api/voice", response_model=VoiceResponse)
async def voice(req: VoiceRequest) -> VoiceResponse:
    """Process a voice message: STT → LLM → TTS.

    Receives base64-encoded audio, transcribes it with STT, sends the
    transcript to the LLM, and synthesizes the reply with TTS.

    Gracefully degrades to text-only if STT/TTS models are not available
    (e.g., no GPU).

    Args:
        req: VoiceRequest with audio_base64, optional language and session_id.

    Returns:
        VoiceResponse with transcript, reply, and optional audio_base64.

    Raises:
        HTTPException: If audio is invalid or processing fails.
    """
    if not req.audio_base64.strip():
        raise HTTPException(status_code=400, detail="Audio data cannot be empty")

    # Determine session ID
    session_id = req.session_id or str(uuid.uuid4())

    # Decode audio
    try:
        audio_bytes = base64.b64decode(req.audio_base64)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Invalid base64 audio data: {exc}"
        ) from exc

    # Determine language
    language = req.language

    # Step 1: STT – transcribe the audio
    transcript = ""
    stt = await _ensure_stt()

    if stt is not None:
        try:
            result = await stt.atranscribe_bytes(
                audio_bytes=audio_bytes,
                sample_rate=16000,
                language=language,
            )
            transcript = result.text
            # Use detected language if not provided
            if not language and result.language:
                language = result.language
            logger.info("STT transcript: %s (lang=%s)", transcript[:80], language)
        except Exception as exc:
            logger.error("STT failed: %s", exc)
            raise HTTPException(
                status_code=500, detail=f"Speech-to-text failed: {exc}"
            ) from exc
    else:
        logger.warning("STT unavailable – voice endpoint requires STT service")
        raise HTTPException(
            status_code=503,
            detail="Speech-to-text service is not available. "
            "Ensure GPU is available or configure DashScope API.",
        )

    if not transcript.strip():
        raise HTTPException(
            status_code=400, detail="Could not transcribe audio. Please try again."
        )

    # Default language
    language = language or _detect_language(transcript)

    # Step 2: Build context and call LLM
    cognate_context = find_cognate_context(transcript)
    skill_context = _detect_skill_context(transcript)
    extra_context = cognate_context + skill_context

    system_prompt = build_system_prompt(mode="voice", extra_context=extra_context)

    reply = await _chat_with_llm(
        message=transcript,
        system_prompt=system_prompt,
        session_id=session_id,
        language=language,
    )

    # Step 3: TTS – synthesize reply to speech (optional)
    audio_base64: str | None = None
    tts = await _ensure_tts()

    if tts is not None and reply.strip():
        try:
            # Determine TTS language: use "zh" if reply contains Chinese, else "vi"
            has_chinese = any("\u4e00" <= c <= "\u9fff" for c in reply)
            tts_language = "zh" if has_chinese else language

            synthesis = await tts.synthesize(
                text=reply,
                language=tts_language,
            )
            audio_base64 = base64.b64encode(synthesis.audio_bytes).decode("ascii")
            logger.info(
                "TTS synthesized %s audio (%.1f seconds)",
                synthesis.backend,
                synthesis.duration_seconds,
            )
        except Exception as exc:
            logger.warning("TTS failed (returning text-only): %s", exc)
            audio_base64 = None
    else:
        logger.info("TTS unavailable or empty reply – returning text-only response")

    return VoiceResponse(
        transcript=transcript,
        reply=reply,
        audio_base64=audio_base64,
        language=language,
    )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint.

    Returns the server status and availability of STT/TTS services.

    Returns:
        HealthResponse with status and service availability flags.
    """
    # Check STT availability
    stt_ok = _stt_available
    if not _stt_available and _stt_service is None:
        # Lazy check
        try:
            stt = await _ensure_stt()
            stt_ok = stt is not None
        except Exception:
            stt_ok = False

    # Check TTS availability
    tts_ok = _tts_available
    if not _tts_available and _tts_service is None:
        try:
            tts = await _ensure_tts()
            tts_ok = tts is not None
        except Exception:
            tts_ok = False

    return HealthResponse(
        status="ok",
        stt_available=stt_ok,
        tts_available=tts_ok,
    )


@app.post("/api/cognates", response_model=CognateResponse)
async def lookup_cognate(req: CognateRequest) -> CognateResponse:
    """Look up a Vietnamese word in the cognates database.

    Searches for exact matches first, then returns fuzzy/partial matches.

    Args:
        req: CognateRequest with the word to look up.

    Returns:
        CognateResponse with exact match (if found) and related entries.
    """
    cognates = load_cognates()
    if not cognates:
        return CognateResponse(cognate=None, related=[])

    word_lower = req.word.lower().strip()
    exact_match: dict[str, Any] | None = None
    related: list[dict[str, Any]] = []

    for entry in cognates:
        vn = entry.get("vietnamese", "").lower()
        if not vn:
            continue

        # Exact match
        if vn == word_lower:
            exact_match = entry
            continue

        # Partial match: word appears in the Vietnamese entry or vice versa
        if word_lower in vn or vn in word_lower:
            if len(related) < 5:
                related.append(entry)

        # Also check if individual words match (for compound words)
        word_tokens = set(word_lower.replace(" ", "").split())
        vn_tokens = set(vn.replace(" ", "").split())
        if word_tokens & vn_tokens and entry not in related:
            if len(related) < 5:
                related.append(entry)

    return CognateResponse(cognate=exact_match, related=related)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PYTHON_API_PORT", "8000"))
    logger.info("Starting Language Learning Assistant API on port %d", port)

    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info",
    )
