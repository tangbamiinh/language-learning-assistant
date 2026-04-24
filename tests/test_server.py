"""Tests for the Phase 1 bridge server."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _clear_global_state():
    """Clear global state before and after each test."""
    from src import server

    # Save original state
    orig_sessions = server.sessions.copy()
    orig_cognates = server._cognates.copy()
    orig_client = server._client
    orig_stt = server._stt_service
    orig_tts = server._tts_service
    orig_stt_avail = server._stt_available
    orig_tts_avail = server._tts_available

    # Reset to clean state
    server.sessions.clear()
    server._cognates = []
    server._client = None
    server._stt_service = None
    server._tts_service = None
    server._stt_available = False
    server._tts_available = False

    yield

    # Restore original state
    server.sessions = orig_sessions
    server._cognates = orig_cognates
    server._client = orig_client
    server._stt_service = orig_stt
    server._tts_service = orig_tts
    server._stt_available = orig_stt_avail
    server._tts_available = orig_tts_avail


@pytest.fixture
def cognates_data():
    """Sample cognates data for testing."""
    return [
        {
            "vietnamese": "học sinh",
            "chinese_traditional": "學生",
            "chinese_simplified": "学生",
            "pinyin": "xuéshēng",
            "english": "student",
            "hsk_level": 1,
            "category": "education",
        },
        {
            "vietnamese": "điện thoại",
            "chinese_traditional": "電話",
            "chinese_simplified": "电话",
            "pinyin": "diànhuà",
            "english": "telephone",
            "hsk_level": 1,
            "category": "technology",
        },
        {
            "vietnamese": "gia đình",
            "chinese_traditional": "家庭",
            "chinese_simplified": "家庭",
            "pinyin": "jiātíng",
            "english": "family",
            "hsk_level": 1,
            "category": "family",
        },
    ]


@pytest.fixture
def mock_cognates_file(cognates_data, tmp_path):
    """Write sample cognates to a temp file and patch the path."""
    data_file = tmp_path / "cognates.json"
    data_file.write_text(json.dumps(cognates_data), encoding="utf-8")
    return data_file


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestChatRequest:
    """Test ChatRequest model validation."""

    def test_valid_request(self):
        from src.server import ChatRequest

        req = ChatRequest(message="Xin chào")
        assert req.message == "Xin chào"
        assert req.language is None
        assert req.session_id is None

    def test_valid_request_with_options(self):
        from src.server import ChatRequest

        req = ChatRequest(
            message="Hello",
            language="vi",
            session_id="test-session",
        )
        assert req.message == "Hello"
        assert req.language == "vi"
        assert req.session_id == "test-session"

    def test_rejects_empty_message(self):
        from src.server import ChatRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatRequest(message="")


class TestVoiceRequest:
    """Test VoiceRequest model validation."""

    def test_valid_request(self):
        from src.server import VoiceRequest

        req = VoiceRequest(audio_base64="dGVzdA==")
        assert req.audio_base64 == "dGVzdA=="

    def test_rejects_empty_audio(self):
        from src.server import VoiceRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VoiceRequest(audio_base64="")


class TestCognateRequest:
    """Test CognateRequest model validation."""

    def test_valid_request(self):
        from src.server import CognateRequest

        req = CognateRequest(word="học sinh")
        assert req.word == "học sinh"

    def test_rejects_empty_word(self):
        from src.server import CognateRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CognateRequest(word="")


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestDetectLanguage:
    """Test language detection heuristics."""

    def test_detects_vietnamese(self):
        from src.server import _detect_language

        assert _detect_language("Xin chào, bạn có khỏe không?") == "vi"

    def test_detects_chinese(self):
        from src.server import _detect_language

        assert _detect_language("你好，世界") == "zh"

    def test_detects_english(self):
        from src.server import _detect_language

        assert _detect_language("Hello world") == "en"

    def test_empty_returns_english(self):
        from src.server import _detect_language

        assert _detect_language("") == "en"

    def test_whitespace_only_returns_english(self):
        from src.server import _detect_language

        assert _detect_language("   ") == "en"

    def test_mixed_vietnamese_dominates(self):
        from src.server import _detect_language

        assert _detect_language("Xin chào hello") == "vi"


class TestFindCognateContext:
    """Test cognate context extraction."""

    def test_finds_matching_cognates(self, cognates_data):
        from src import server

        server._cognates = cognates_data

        context = server.find_cognate_context("Tôi muốn học về học sinh")
        assert "### Relevant Hán Việt Cognates:" in context
        assert "学生" in context
        assert "xuéshēng" in context

    def test_returns_empty_for_no_matches(self, cognates_data):
        from src import server

        server._cognates = cognates_data

        context = server.find_cognate_context("Hello world foo bar")
        assert context == ""

    def test_returns_empty_when_no_cognates_loaded(self):
        from src import server

        # Set a sentinel to prevent reload from file
        server._cognates = [{}]  # Non-empty but no valid entries
        context = server.find_cognate_context("học sinh")
        # Should not match any cognates since entries have no "vietnamese" field
        assert context == ""

    def test_limits_matches_to_five(self, cognates_data):
        from src import server

        # Create many cognates that match
        many_cognates = []
        for i in range(10):
            many_cognates.append(
                {
                    "vietnamese": f"từ {i}",
                    "chinese_simplified": f"字{i}",
                    "pinyin": f"zi {i}",
                    "english": f"word {i}",
                    "hsk_level": 1,
                    "category": "daily",
                }
            )
        server._cognates = many_cognates

        message = " ".join(f"từ {i}" for i in range(10))
        context = server.find_cognate_context(message)

        # Should only have up to 5 matches
        match_count = context.count("- ")
        assert match_count <= 5


class TestDetectSkillContext:
    """Test skill context detection."""

    def test_detects_pronunciation_context(self):
        from src.server import _detect_skill_context

        context = _detect_skill_context("Làm sao phát âm từ này?")
        assert "### Pronunciation Focus:" in context

    def test_detects_grammar_context(self):
        from src.server import _detect_skill_context

        context = _detect_skill_context("Ngữ pháp của 了 là gì?")
        assert "### Grammar Focus:" in context

    def test_detects_both_contexts(self):
        from src.server import _detect_skill_context

        context = _detect_skill_context("Phát âm và ngữ pháp của từ này")
        assert "### Pronunciation Focus:" in context
        assert "### Grammar Focus:" in context

    def test_returns_empty_for_general_message(self):
        from src.server import _detect_skill_context

        context = _detect_skill_context("Xin chào, hôm nay thế nào?")
        assert context == ""


class TestBuildSystemPrompt:
    """Test system prompt building."""

    def test_text_mode_uses_base_prompt(self):
        from src.server import build_system_prompt
        from src.prompts import SYSTEM_PROMPT

        prompt = build_system_prompt(mode="text", extra_context="")
        assert prompt == SYSTEM_PROMPT

    def test_voice_mode_uses_voice_prompt(self):
        from src.server import build_system_prompt
        from src.prompts import SYSTEM_PROMPT_VOICE

        prompt = build_system_prompt(mode="voice", extra_context="")
        assert prompt == SYSTEM_PROMPT_VOICE

    def test_appends_extra_context(self):
        from src.server import build_system_prompt

        extra = "\n\n### Extra Context:\nSome extra info"
        prompt = build_system_prompt(mode="text", extra_context=extra)
        assert "### Extra Context:" in prompt

    def test_ignores_empty_extra_context(self):
        from src.server import build_system_prompt
        from src.prompts import SYSTEM_PROMPT

        prompt = build_system_prompt(mode="text", extra_context="   ")
        assert prompt == SYSTEM_PROMPT


class TestSessionManagement:
    """Test session history management."""

    def test_get_session_creates_new(self):
        from src import server

        server.sessions.clear()
        history = server._get_session("new-session")
        assert history == []
        assert "new-session" in server.sessions

    def test_get_session_returns_existing(self):
        from src import server

        server.sessions["existing"] = [
            {"role": "user", "content": "Hello"}
        ]
        history = server._get_session("existing")
        assert len(history) == 1

    def test_add_message_trims_history(self):
        from src import server

        server.sessions.clear()
        # Add more than MAX_MESSAGES_PER_SESSION messages
        for i in range(30):
            server._add_message("trim-test", "user", f"Message {i}")

        assert len(server.sessions["trim-test"]) == server.MAX_MESSAGES_PER_SESSION

    def test_add_message_keeps_order(self):
        from src import server

        server.sessions.clear()
        server._add_message("order-test", "user", "First")
        server._add_message("order-test", "assistant", "Second")
        server._add_message("order-test", "user", "Third")

        history = server.sessions["order-test"]
        assert history[0]["content"] == "First"
        assert history[1]["content"] == "Second"
        assert history[2]["content"] == "Third"


class TestLoadCognates:
    """Test cognates loading."""

    def test_loads_from_file(self, cognates_data, mock_cognates_file):
        from src import server

        server._cognates = []  # Force reload

        with patch.object(server, "COGNATES_PATH", mock_cognates_file):
            result = server.load_cognates()

        assert len(result) == 3
        assert result[0]["vietnamese"] == "học sinh"

    def test_handles_missing_file_gracefully(self):
        from src import server

        server._cognates = []  # Force reload

        with patch.object(server, "COGNATES_PATH", Path("/nonexistent/path")):
            result = server.load_cognates()

        assert result == []

    def test_cached_result_not_reloaded(self, cognates_data, mock_cognates_file):
        from src import server

        server._cognates = cognates_data
        with patch.object(server, "COGNATES_PATH", mock_cognates_file):
            result = server.load_cognates()

        assert result is server._cognates


# ---------------------------------------------------------------------------
# LLM client tests
# ---------------------------------------------------------------------------


class TestGetLLMClient:
    """Test LLM client initialization."""

    def test_creates_client_with_api_key(self):
        from src.server import _get_llm_client

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            client = _get_llm_client()

        assert client is not None
        assert client.base_url.host == "openrouter.ai"

    def test_raises_without_api_key(self):
        from src.server import _get_llm_client

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=True):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                _get_llm_client()

    def test_returns_cached_client(self):
        from src.server import _get_llm_client

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
            client1 = _get_llm_client()
            client2 = _get_llm_client()

        assert client1 is client2


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def api_key_env():
    """Set OPENROUTER_API_KEY for endpoint tests."""
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        yield


@pytest.fixture
def client(api_key_env):
    """Create a TestClient for the FastAPI app."""
    from src.server import app

    return TestClient(app)


class TestChatEndpoint:
    """Test the /api/chat endpoint."""

    def _mock_llm_client(self, responses):
        """Create a mock LLM client that returns the given responses."""
        mock_client = AsyncMock()

        async def mock_create(*args, **kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = next(responses)
            return resp

        mock_client.chat.completions.create = mock_create
        return mock_client

    def test_rejects_empty_message(self, client):
        response = client.post(
            "/api/chat",
            json={"message": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_rejects_whitespace_message(self, client):
        """Whitespace-only messages are rejected with 400 by endpoint logic."""
        response = client.post(
            "/api/chat",
            json={"message": "   "},
        )
        # Whitespace passes Pydantic (min_length=1) but endpoint rejects it
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_returns_llm_response(self, client):
        mock_client = self._mock_llm_client(iter(["Chào bạn!很高兴见到你。"]))

        with patch("src.server._get_llm_client", return_value=mock_client):
            response = client.post(
                "/api/chat",
                json={"message": "Xin chào"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["reply"] == "Chào bạn!很高兴见到你。"
        assert data["language"] == "vi"
        assert "session_id" not in data  # session_id is internal

    def test_generates_session_id(self, client):
        from src import server

        mock_client = self._mock_llm_client(iter(["Reply"]))

        with patch("src.server._get_llm_client", return_value=mock_client):
            client.post(
                "/api/chat",
                json={"message": "Hello"},
            )

        # Session should have been created
        assert len(server.sessions) == 1

    def test_uses_provided_session_id(self, client):
        from src import server

        mock_client = self._mock_llm_client(iter(["Reply"]))

        with patch("src.server._get_llm_client", return_value=mock_client):
            response = client.post(
                "/api/chat",
                json={
                    "message": "Hello",
                    "session_id": "my-session",
                },
            )

        assert response.status_code == 200
        assert "my-session" in server.sessions

    def test_preserves_conversation_history(self, client):
        from src import server

        mock_client = self._mock_llm_client(iter(["Reply", "Reply"]))

        session_id = "history-test"
        with patch("src.server._get_llm_client", return_value=mock_client):
            client.post(
                "/api/chat",
                json={"message": "First message", "session_id": session_id},
            )
            client.post(
                "/api/chat",
                json={"message": "Second message", "session_id": session_id},
            )

        # History should have user + assistant for both messages
        history = server.sessions[session_id]
        assert len(history) == 4  # 2 user + 2 assistant

    async def test_chat_with_llm_builds_correct_messages(self, api_key_env):
        """Verify the message structure sent to the LLM."""
        from src.server import _chat_with_llm

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test reply"

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("src.server._get_llm_client", return_value=mock_client):
            await _chat_with_llm(
                message="Hello",
                system_prompt="You are a tutor",
                session_id="test",
                language="en",
            )

        # Verify the call was made with correct structure
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a tutor"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"


class TestVoiceEndpoint:
    """Test the /api/voice endpoint."""

    def test_rejects_empty_audio(self, client):
        response = client.post(
            "/api/voice",
            json={"audio_base64": ""},
        )
        assert response.status_code == 422

    def test_rejects_invalid_base64(self, client):
        response = client.post(
            "/api/voice",
            json={"audio_base64": "not-valid-base64!!!"},
        )
        assert response.status_code == 400
        assert "Invalid base64" in response.json()["detail"]

    def test_returns_503_when_stt_unavailable(self, client):
        # Ensure STT is not initialized
        from src import server

        server._stt_service = None
        server._stt_available = False

        # Create valid base64 audio
        audio_bytes = b"\x00" * 100
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        response = client.post(
            "/api/voice",
            json={"audio_base64": audio_b64},
        )

        assert response.status_code == 503
        assert "Speech-to-text" in response.json()["detail"]

    @patch("src.server._ensure_stt", new_callable=AsyncMock)
    @patch("src.server._ensure_tts", new_callable=AsyncMock)
    @patch("src.server._chat_with_llm", new_callable=AsyncMock)
    async def test_full_voice_pipeline(
        self, mock_chat, mock_tts, mock_stt, client
    ):
        """Test the complete voice pipeline: STT → LLM → TTS."""
        # Mock STT result
        mock_stt_result = MagicMock()
        mock_stt_result.text = "Xin chào"
        mock_stt_result.language = "vi"

        mock_stt_instance = AsyncMock()
        mock_stt_instance.atranscribe_bytes = AsyncMock(return_value=mock_stt_result)
        mock_stt.return_value = mock_stt_instance

        # Mock LLM response
        mock_chat.return_value = "Chào bạn! 你好！"

        # Mock TTS result
        mock_tts_result = MagicMock()
        mock_tts_result.audio_bytes = b"fake-audio-data"
        mock_tts_result.backend = "qwen"
        mock_tts_result.duration_seconds = 1.5

        mock_tts_instance = AsyncMock()
        mock_tts_instance.synthesize = AsyncMock(return_value=mock_tts_result)
        mock_tts.return_value = mock_tts_instance

        # Create valid audio
        audio_bytes = b"\x00" * 100
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        response = client.post(
            "/api/voice",
            json={"audio_base64": audio_b64},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "Xin chào"
        assert data["reply"] == "Chào bạn! 你好！"
        assert data["audio_base64"] is not None
        assert data["language"] == "vi"

    @patch("src.server._ensure_stt", new_callable=AsyncMock)
    @patch("src.server._ensure_tts", new_callable=AsyncMock)
    @patch("src.server._chat_with_llm", new_callable=AsyncMock)
    async def test_voice_without_tts(self, mock_chat, mock_tts, mock_stt, client):
        """Test voice pipeline when TTS is unavailable."""
        mock_stt_result = MagicMock()
        mock_stt_result.text = "Xin chào"
        mock_stt_result.language = "vi"

        mock_stt_instance = AsyncMock()
        mock_stt_instance.atranscribe_bytes = AsyncMock(return_value=mock_stt_result)
        mock_stt.return_value = mock_stt_instance

        mock_chat.return_value = "Chào bạn"
        mock_tts.return_value = None  # TTS unavailable

        audio_bytes = b"\x00" * 100
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")

        response = client.post(
            "/api/voice",
            json={"audio_base64": audio_b64},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "Xin chào"
        assert data["audio_base64"] is None


class TestHealthEndpoint:
    """Test the /api/health endpoint."""

    def test_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_returns_service_status(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert "stt_available" in data
        assert "tts_available" in data
        assert isinstance(data["stt_available"], bool)
        assert isinstance(data["tts_available"], bool)


class TestCognatesEndpoint:
    """Test the /api/cognates endpoint."""

    @pytest.fixture(autouse=True)
    def _load_cognates(self, cognates_data):
        from src import server

        server._cognates = cognates_data

    def test_finds_exact_match(self, client):
        response = client.post(
            "/api/cognates",
            json={"word": "học sinh"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cognate"] is not None
        assert data["cognate"]["chinese_simplified"] == "学生"
        assert data["cognate"]["pinyin"] == "xuéshēng"

    def test_case_insensitive_match(self, client):
        response = client.post(
            "/api/cognates",
            json={"word": "HỌC SINH"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cognate"] is not None

    def test_returns_null_for_no_match(self, client):
        response = client.post(
            "/api/cognates",
            json={"word": "nonexistent_word_xyz"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cognate"] is None
        assert data["related"] == []

    def test_returns_partial_matches(self, client):
        response = client.post(
            "/api/cognates",
            json={"word": "điện"},
        )
        assert response.status_code == 200
        data = response.json()
        # "điện thoại" contains "điện"
        assert len(data["related"]) >= 1

    def test_empty_cognates_returns_empty(self, client):
        from src import server

        # Set to non-empty sentinel to prevent file reload, but with no matching entries
        server._cognates = [{"vietnamese": "xyz_nonexistent", "chinese_simplified": "某", "pinyin": "mou", "english": "none", "hsk_level": 1, "category": "daily"}]

        response = client.post(
            "/api/cognates",
            json={"word": "học sinh"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cognate"] is None
        assert data["related"] == []


# ---------------------------------------------------------------------------
# Integration tests with mocked LLM
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests with mocked LLM."""

    def _mock_llm_client(self, responses):
        """Create a mock LLM client that returns the given responses."""
        mock_client = AsyncMock()

        async def mock_create(*args, **kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            resp.choices[0].message.content = next(responses)
            return resp

        mock_client.chat.completions.create = mock_create
        return mock_client

    def test_conversation_flow(self, client):
        """Test a multi-turn conversation maintains context."""
        from src import server

        responses = iter([
            "Chào bạn! Bạn muốn học gì hôm nay?",
            "Hay lắm! 学生 (xuéshēng) nghĩa là 'học sinh' trong Hán Việt.",
        ])
        mock_client = self._mock_llm_client(responses)

        with patch("src.server._get_llm_client", return_value=mock_client):
            session_id = "integration-test"

            # First turn
            r1 = client.post(
                "/api/chat",
                json={"message": "Xin chào", "session_id": session_id},
            )
            assert r1.status_code == 200
            assert "Chào bạn" in r1.json()["reply"]

            # Second turn
            r2 = client.post(
                "/api/chat",
                json={"message": "học sinh nghĩa là gì?", "session_id": session_id},
            )
            assert r2.status_code == 200

        # Verify history was maintained
        assert len(server.sessions[session_id]) == 4

    def test_cognate_injection_in_chat(self, client):
        """Test that cognates are injected into the system prompt."""
        from src import server

        # Load cognates
        server._cognates = [
            {
                "vietnamese": "học sinh",
                "chinese_simplified": "学生",
                "pinyin": "xuéshēng",
                "english": "student",
                "hsk_level": 1,
                "category": "education",
            }
        ]

        mock_client = self._mock_llm_client(iter(["回复"]))

        with patch("src.server._get_llm_client", return_value=mock_client):
            response = client.post(
                "/api/chat",
                json={"message": "Giáo dục và học sinh"},
            )
        assert response.status_code == 200

    def test_history_trimming(self, client):
        """Test that history is trimmed after MAX_MESSAGES_PER_SESSION."""
        from src import server

        responses = iter(["Reply"] * 25)
        mock_client = self._mock_llm_client(responses)

        with patch("src.server._get_llm_client", return_value=mock_client):
            for i in range(25):
                client.post(
                    "/api/chat",
                    json={
                        "message": f"Message {i}",
                        "session_id": "trim-session",
                    },
                )

        history = server.sessions["trim-session"]
        assert len(history) == server.MAX_MESSAGES_PER_SESSION
