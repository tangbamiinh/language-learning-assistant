"""Tests for the prompts module."""

import pytest
from src.prompts import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_VOICE,
    build_cognate_prompt,
    build_scenario_prompt,
    build_grammar_prompt,
    build_tone_drill_prompt,
)


class TestSystemPrompts:
    """Test that system prompts are properly configured."""

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_hanviet(self):
        assert "Hán Việt" in SYSTEM_PROMPT or "hanviet" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_chinese(self):
        assert "Chinese" in SYSTEM_PROMPT or "chinese" in SYSTEM_PROMPT.lower()

    def test_system_prompt_mentions_vietnamese(self):
        assert "Vietnamese" in SYSTEM_PROMPT or "vietnamese" in SYSTEM_PROMPT.lower()

    def test_voice_prompt_shorter(self):
        """Voice prompt should be shorter than text prompt for concise responses."""
        assert len(SYSTEM_PROMPT_VOICE) < len(SYSTEM_PROMPT)


class TestCognatePrompt:
    """Test cognate prompt builder."""

    def test_build_cognate_prompt_returns_string(self):
        prompt = build_cognate_prompt("quốc gia")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_build_cognate_prompt_includes_word(self):
        prompt = build_cognate_prompt("giáo dục")
        assert "giáo dục" in prompt


class TestScenarioPrompt:
    """Test scenario prompt builder."""

    def test_build_scenario_prompt_minimal(self):
        scenario = {"title": "Restaurant"}
        prompt = build_scenario_prompt(scenario)
        assert isinstance(prompt, str)
        assert "Restaurant" in prompt

    def test_build_scenario_prompt_full(self):
        scenario = {
            "title": "Restaurant",
            "setting": "A Chinese restaurant in Hanoi",
            "student_role": "customer",
            "tutor_role": "waiter",
            "goal": "Order food and pay the bill",
            "key_vocabulary": ["菜单", "点餐", "买单"],
            "hsk_level": 2,
        }
        prompt = build_scenario_prompt(scenario)
        assert "customer" in prompt
        assert "waiter" in prompt


class TestGrammarPrompt:
    """Test grammar prompt builder."""

    def test_build_grammar_prompt(self):
        prompt = build_grammar_prompt("了 le - completed action")
        assert isinstance(prompt, str)
        assert "了" in prompt or "le" in prompt.lower()


class TestToneDrillPrompt:
    """Test tone drill prompt builder."""

    def test_build_tone_drill_single_tone(self):
        prompt = build_tone_drill_prompt([1])
        assert isinstance(prompt, str)

    def test_build_tone_drill_multiple_tones(self):
        prompt = build_tone_drill_prompt([1, 2, 3, 4])
        assert isinstance(prompt, str)

    def test_build_tone_drill_all_tones(self):
        prompt = build_tone_drill_prompt([1, 2, 3, 4, 5])
        assert isinstance(prompt, str)
