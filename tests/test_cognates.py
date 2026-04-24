"""Tests for the cognates database."""

import json
import pytest
from pathlib import Path

COGNATES_PATH = Path(__file__).parent.parent / "src" / "cognates.json"


@pytest.fixture
def cognates():
    with open(COGNATES_PATH) as f:
        return json.load(f)


class TestCognatesDatabase:
    """Test that the cognates database is well-formed."""

    def test_file_exists(self):
        assert COGNATES_PATH.exists()

    def test_valid_json(self, cognates):
        assert isinstance(cognates, list)
        assert len(cognates) > 0

    def test_minimum_entries(self, cognates):
        assert len(cognates) >= 100

    def test_required_fields(self, cognates):
        required = {
            "vietnamese", "chinese_traditional", "chinese_simplified",
            "pinyin", "english", "hsk_level", "category",
        }
        for i, entry in enumerate(cognates):
            for field in required:
                assert field in entry, f"Entry {i} missing field '{field}'"

    def test_no_empty_vietnamese(self, cognates):
        for i, entry in enumerate(cognates):
            assert entry["vietnamese"].strip(), f"Entry {i} has empty vietnamese"

    def test_no_empty_chinese(self, cognates):
        for i, entry in enumerate(cognates):
            assert entry["chinese_traditional"].strip(), f"Entry {i} has empty chinese_traditional"
            assert entry["chinese_simplified"].strip(), f"Entry {i} has empty chinese_simplified"

    def test_no_empty_pinyin(self, cognates):
        for i, entry in enumerate(cognates):
            assert entry["pinyin"].strip(), f"Entry {i} has empty pinyin"

    def test_valid_hsk_levels(self, cognates):
        for i, entry in enumerate(cognates):
            level = entry["hsk_level"]
            assert level is None or (isinstance(level, int) and 1 <= level <= 6), \
                f"Entry {i} has invalid HSK level: {level}"

    def test_valid_categories(self, cognates):
        valid = {
            "daily", "education", "business", "nature", "technology",
            "society", "health", "travel", "family", "food",
        }
        for i, entry in enumerate(cognates):
            assert entry["category"] in valid, \
                f"Entry {i} has invalid category: {entry['category']}"

    def test_no_duplicate_vietnamese(self, cognates):
        seen = set()
        for i, entry in enumerate(cognates):
            assert entry["vietnamese"] not in seen, \
                f"Entry {i} has duplicate vietnamese: {entry['vietnamese']}"
            seen.add(entry["vietnamese"])

    def test_has_hsk_distribution(self, cognates):
        """Should have entries across multiple HSK levels."""
        levels = {e["hsk_level"] for e in cognates if e["hsk_level"] is not None}
        assert len(levels) >= 3, "Should cover at least 3 HSK levels"

    def test_has_category_distribution(self, cognates):
        """Should have entries across multiple categories."""
        categories = {e["category"] for e in cognates}
        assert len(categories) >= 5, "Should cover at least 5 categories"
