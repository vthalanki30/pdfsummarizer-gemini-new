"""Configuration loading and validation.

Secrets come from the process environment first, then from ``.env`` files
(repo root, then the parent directory). Nothing here ever mutates ``os.environ``
for values that already exist, and nothing writes secrets to disk.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from .errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILES = (REPO_ROOT / ".env", REPO_ROOT.parent / ".env")

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_MAX_PAGES = 120
DEFAULT_CHUNK_TOKENS = 3000
DEFAULT_CHUNK_OVERLAP_TOKENS = 200
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.2
DEFAULT_STUFF_MAX_TOKENS = 18000


def _env_str(env: Mapping[str, str], name: str, default: str = "") -> str:
    return (env.get(name) or default).strip()


def _env_int(env: Mapping[str, str], name: str, default: int, *, minimum: int = 0) -> int:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Validated application settings."""

    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    max_pages: int = DEFAULT_MAX_PAGES
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_retries: int = DEFAULT_MAX_RETRIES
    temperature: float = DEFAULT_TEMPERATURE
    stuff_max_tokens: int = DEFAULT_STUFF_MAX_TOKENS

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ConfigError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
                "(see https://aistudio.google.com/app/apikey)."
            )
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ConfigError(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_TOKENS "
                f"(got {self.chunk_overlap_tokens} >= {self.chunk_tokens})."
            )
        if self.max_concurrency < 1:
            raise ConfigError("MAX_CONCURRENCY must be at least 1.")
        if self.max_retries < 0:
            raise ConfigError("MAX_RETRIES must be 0 or more.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigError("TEMPERATURE must be between 0 and 2.")

    @property
    def masked_key(self) -> str:
        """API key safe to render in a UI or log line."""
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

    def public_dict(self) -> dict[str, object]:
        """Settings as a dict with the secret masked and no key material."""
        return {
            "model": self.model,
            "embedding_model": self.embedding_model,
            "max_pages": self.max_pages,
            "chunk_tokens": self.chunk_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "max_concurrency": self.max_concurrency,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "stuff_max_tokens": self.stuff_max_tokens,
            "api_key": self.masked_key,
        }


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build ``Settings`` from a mapping (defaults to the process environment)."""
    source: Mapping[str, str] = os.environ if env is None else env
    return Settings(
        api_key=_env_str(source, "GEMINI_API_KEY"),
        model=_env_str(source, "GEMINI_MODEL", DEFAULT_MODEL),
        embedding_model=_env_str(source, "GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        max_pages=_env_int(source, "MAX_PAGES", DEFAULT_MAX_PAGES, minimum=1),
        chunk_tokens=_env_int(source, "CHUNK_TOKENS", DEFAULT_CHUNK_TOKENS, minimum=100),
        chunk_overlap_tokens=_env_int(
            source, "CHUNK_OVERLAP_TOKENS", DEFAULT_CHUNK_OVERLAP_TOKENS, minimum=0
        ),
        max_concurrency=_env_int(source, "MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY, minimum=1),
        max_retries=_env_int(source, "MAX_RETRIES", DEFAULT_MAX_RETRIES, minimum=0),
        temperature=_env_float(source, "TEMPERATURE", DEFAULT_TEMPERATURE),
        stuff_max_tokens=_env_int(
            source, "STUFF_MAX_TOKENS", DEFAULT_STUFF_MAX_TOKENS, minimum=1000
        ),
    )


def load_env_files(paths: tuple[Path, ...] = ENV_FILES) -> list[Path]:
    """Load ``.env`` files without overriding existing environment variables."""
    loaded: list[Path] = []
    for path in paths:
        if path.is_file():
            load_dotenv(path, override=False)
            loaded.append(path)
    return loaded


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings for application code (environment first, then .env)."""
    load_env_files()
    return load_settings()


def reset_settings_cache() -> None:
    """Clear the cache (tests, and after the user edits configuration)."""
    get_settings.cache_clear()
