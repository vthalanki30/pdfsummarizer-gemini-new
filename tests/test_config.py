"""Configuration parsing and validation."""

from __future__ import annotations

import pytest

from src.config import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MODEL,
    Settings,
    load_env_files,
    load_settings,
)
from src.errors import ConfigError

VALID_ENV = {
    "GEMINI_API_KEY": "test-key-value-1234567890",
    "GEMINI_MODEL": "gemini-3.1-flash-lite",
}


def test_defaults_applied():
    settings = load_settings(VALID_ENV)
    assert settings.model == DEFAULT_MODEL
    assert settings.chunk_tokens == DEFAULT_CHUNK_TOKENS
    assert settings.chunk_overlap_tokens == DEFAULT_CHUNK_OVERLAP_TOKENS
    assert settings.max_pages == DEFAULT_MAX_PAGES
    assert settings.embedding_model == "gemini-embedding-001"


def test_values_from_mapping():
    settings = load_settings(
        {
            **VALID_ENV,
            "CHUNK_TOKENS": "1200",
            "CHUNK_OVERLAP_TOKENS": "80",
            "MAX_PAGES": "10",
            "MAX_CONCURRENCY": "2",
            "TEMPERATURE": "0.7",
        }
    )
    assert settings.chunk_tokens == 1200
    assert settings.chunk_overlap_tokens == 80
    assert settings.max_pages == 10
    assert settings.max_concurrency == 2
    assert settings.temperature == 0.7


def test_missing_key_raises_actionable_error():
    with pytest.raises(ConfigError) as excinfo:
        load_settings({})
    assert "GEMINI_API_KEY" in str(excinfo.value)
    assert ".env" in str(excinfo.value)


def test_blank_key_is_missing():
    with pytest.raises(ConfigError):
        load_settings({"GEMINI_API_KEY": "   "})


def test_non_integer_raises():
    with pytest.raises(ConfigError) as excinfo:
        load_settings({**VALID_ENV, "MAX_PAGES": "many"})
    assert "MAX_PAGES" in str(excinfo.value)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ConfigError) as excinfo:
        load_settings({**VALID_ENV, "CHUNK_TOKENS": "500", "CHUNK_OVERLAP_TOKENS": "500"})
    assert "CHUNK_OVERLAP_TOKENS" in str(excinfo.value)


def test_temperature_bounds():
    with pytest.raises(ConfigError):
        load_settings({**VALID_ENV, "TEMPERATURE": "3.5"})


def test_secret_is_masked_everywhere():
    settings = load_settings(VALID_ENV)
    assert "test-key-value-1234567890" not in repr(settings)
    assert "test-key-value-1234567890" not in str(settings.public_dict())
    assert settings.masked_key.startswith("test")
    assert settings.masked_key.endswith("7890")


def test_short_key_is_fully_masked():
    settings = Settings(api_key="short")
    assert settings.masked_key == "****"


def test_load_env_files_ignores_missing_paths(tmp_path):
    assert load_env_files((tmp_path / "nope.env",)) == []


def test_load_env_files_reads_real_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("PDFSUM_TEST_FLAG=hello\n", encoding="utf-8")
    monkeypatch.delenv("PDFSUM_TEST_FLAG", raising=False)
    loaded = load_env_files((env_file,))
    assert loaded == [env_file]
    import os

    assert os.environ["PDFSUM_TEST_FLAG"] == "hello"
