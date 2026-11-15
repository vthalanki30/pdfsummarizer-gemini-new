"""Deterministic chat model for tests: no network, no API key, no randomness."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field

MAP_MARKER = "bullets of substantive notes"
FINALIZE_MARKER = "Convert the notes above"


class ScriptedChatModel(BaseChatModel):
    """Answers based on the prompt, recording every prompt it received."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responder: Callable[[list[BaseMessage]], str]
    calls: list[str] = Field(default_factory=list)
    input_tokens: int = 10
    output_tokens: int = 20

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append("\n".join(str(message.content) for message in messages))
        text = self.responder(messages)
        message = AIMessage(
            content=text,
            usage_metadata={
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def is_map_prompt(prompt: str) -> bool:
    return MAP_MARKER in prompt


def is_finalize_prompt(prompt: str) -> bool:
    return FINALIZE_MARKER in prompt
