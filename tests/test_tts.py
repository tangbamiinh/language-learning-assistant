"""Tests for the TTS service."""

from unittest.mock import MagicMock, patch
import pytest
from src.tts_service import TTSService, SynthesisResult, Backend


class TestTTSService:
    """Test TTS service initialization and language routing."""

    @patch("torch.cuda.is_available")
    def test_init_with_gpu(self, mock_cuda):
        """Service should initialize with GPU mode when available."""
        mock_cuda.return_value = True
        with (
            patch("src.tts_service.Qwen3TTSModel"),
            patch("src.tts_service.Vieneu"),
        ):
            service = TTSService()
            assert service._gpu_available is True

    @patch("torch.cuda.is_available")
    def test_init_without_gpu(self, mock_cuda):
        """Service should fall back to CPU mode when no GPU."""
        mock_cuda.return_value = False
        with (
            patch("src.tts_service.Qwen3TTSModel"),
            patch("src.tts_service.Vieneu"),
        ):
            service = TTSService()
            assert service._gpu_available is False

    def test_select_backend_chinese(self):
        """Chinese language should use Qwen3-TTS backend."""
        assert TTSService._select_backend("zh") == Backend.QWEN

    def test_select_backend_vietnamese(self):
        """Vietnamese language should use VieNeu-TTS backend."""
        assert TTSService._select_backend("vi") == Backend.VIENEU

    def test_select_backend_english(self):
        """English should default to Qwen3-TTS (configurable)."""
        assert TTSService._select_backend("en") in (
            Backend.QWEN,
            Backend.VIENEU,
        )

    def test_select_backend_unknown(self):
        """Unknown language should raise error."""
        with pytest.raises(Exception):
            TTSService._select_backend("xx")


class TestSynthesisResult:
    """Test SynthesisResult dataclass."""

    def test_create_result(self):
        result = SynthesisResult(
            audio=b"fake_wav_data",
            language="zh",
            duration_sec=2.5,
        )
        assert result.audio == b"fake_wav_data"
        assert result.language == "zh"
        assert result.duration_sec == 2.5

    def test_result_to_base64(self):
        import base64
        result = SynthesisResult(
            audio=b"test_audio",
            language="vi",
            duration_sec=1.0,
        )
        b64 = result.to_base64()
        assert isinstance(b64, str)
        assert base64.b64decode(b64) == b"test_audio"
