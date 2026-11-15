"""Model construction and message handling.

Two details this module exists to hide:

1. LangChain v1 returns ``content`` as a list of typed blocks, not a string.
   ``str(message.content)`` silently produces ``[{'type': 'text', ...}]``.
2. Token usage lives in ``usage_metadata`` and is absent for some providers,
   so accounting has to tolerate zeroes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .config import Settings
from .schemas import Usage


def message_text(message: BaseMessage | str) -> str:
    """Best-effort plain text from a chat message or ``content`` value."""
    if isinstance(message, str):
        content: Any = message
    elif hasattr(message, "content"):
        content = message.content
    else:
        # tolerate being handed a raw content value (list of blocks, dict, ...)
        content = message
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return str(content.get("text", "")).strip()
    if isinstance(content, Iterable):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type in (None, "text", "text_delta"):
                    parts.append(str(block.get("text", "")))
                elif block_type == "thinking":
                    continue
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "".join(parts).strip()
    return str(content).strip()


class UsageMeter:
    """Accumulates token usage and call counts across a pipeline run."""

    def __init__(self) -> None:
        self.usage = Usage()

    def add(self, message: BaseMessage | None) -> None:
        metadata = getattr(message, "usage_metadata", None) or {}
        input_tokens = int(metadata.get("input_tokens", 0) or 0)
        output_tokens = int(metadata.get("output_tokens", 0) or 0)
        total = int(metadata.get("total_tokens", 0) or 0) or input_tokens + output_tokens
        self.usage = self.usage.merge(
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total,
                calls=1,
            )
        )

    def with_elapsed(self, seconds: float) -> Usage:
        return self.usage.model_copy(update={"elapsed_seconds": round(seconds, 2)})


def build_chat_model(
    settings: Settings,
    *,
    json_mode: bool = False,
    max_output_tokens: int = 2048,
    temperature: float | None = None,
) -> BaseChatModel:
    """Construct the Gemini chat model.

    Imported lazily so unit tests never need the provider SDK or a network.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "google_api_key": settings.api_key,
        "temperature": settings.temperature if temperature is None else temperature,
        "max_output_tokens": max_output_tokens,
        "max_retries": settings.max_retries,
    }
    if json_mode:
        kwargs["response_mime_type"] = "application/json"
    return ChatGoogleGenerativeAI(**kwargs)
