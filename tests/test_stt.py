"""Tests for the STT service."""

from unittest.mock import MagicMock, patch
import pytest
from src.stt_service import STTService, TranscriptionResult, STTError


class TestSTTService:
    """Test STT service initialization and basic operations."""

    @patch("torch.cuda.is_available")
    def test_init_no_gpu_uses_dashscope(self, mock_cuda):
        """When no GPU, STT should default to dashscope backend."""
        mock_cuda.return_value = False
        service = STTService()
        assert service._backend == "dashscope"

    @patch("torch.cuda.is_available")
    def test_init_with_gpu_uses_transformers(self, mock_cuda):
        """When GPU available, STT should default to transformers backend."""
        mock_cuda.return_value = True
        with patch("src.stt_service.Qwen3ASRModel") as mock_model:
            mock_model.from_pretrained = MagicMock()
            service = STTService()
            assert service._backend == "transformers"

    def test_language_mapping_vi(self):
        """Vietnamese language code should map to 'Vietnamese'."""
        # Qwen3-ASR expects display names, not ISO codes
        assert STTService._language_display("vi") == "Vietnamese"

    def test_language_mapping_zh(self):
        """Chinese language code should map to 'Chinese'."""
        assert STTService._language_display("zh") == "Chinese"

    def test_language_mapping_en(self):
        """English language code should map to 'English'."""
        assert STTService._language_display("en") == "English"

    def test_language_mapping_none(self):
        """None should return None for auto-detection."""
        assert STTService._language_display(None) is None

    def test_language_mapping_unknown(self):
        """Unknown language should raise STTConfigurationError."""
        with pytest.raises(STTError):
            STTService._language_display("xx")


class TestTranscriptionResult:
    """Test TranscriptionResult dataclass."""

    def test_create_result(self):
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            confidence=0.95,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.confidence == 0.95

    def test_result_str(self):
        result = TranscriptionResult(
            text="Xin chào",
            language="vi",
            confidence=0.88,
        )
        assert "Xin chào" in str(result)
