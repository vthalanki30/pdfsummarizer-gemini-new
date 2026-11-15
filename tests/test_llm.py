"""Message text extraction and usage accounting (LangChain v1 content blocks)."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from src.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, Settings
from src.llm import UsageMeter, build_chat_model, message_text


class TestMessageText:
    def test_plain_string_message(self):
        assert message_text(HumanMessage(content="hello")) == "hello"

    def test_string_content(self):
        assert message_text("  hello  ") == "hello"

    def test_content_block_list(self):
        message = AIMessage(
            content=[
                {"type": "text", "text": "first "},
                {"type": "text", "text": "second"},
            ]
        )
        assert message_text(message) == "first second"

    def test_thinking_blocks_are_skipped(self):
        message = AIMessage(
            content=[
                {"type": "thinking", "thinking": "internal reasoning"},
                {"type": "text", "text": "answer"},
            ]
        )
        assert message_text(message) == "answer"

    def test_empty_content(self):
        assert message_text(AIMessage(content=[])) == ""
        assert message_text(AIMessage(content="")) == ""

    def test_dict_content(self):
        assert message_text({"text": "value"}) == "value"

    def test_objects_with_text_attribute(self):
        class Block:
            text = "block text"

        assert message_text([Block()]) == "block text"


class TestUsageMeter:
    def test_accumulates_across_calls(self):
        meter = UsageMeter()
        meter.add(
            AIMessage(
                content="a",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )
        )
        meter.add(
            AIMessage(
                content="b",
                usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            )
        )
        assert meter.usage.input_tokens == 13
        assert meter.usage.output_tokens == 6
        assert meter.usage.total_tokens == 19
        assert meter.usage.calls == 2

    def test_missing_usage_metadata_still_counts_the_call(self):
        meter = UsageMeter()
        meter.add(AIMessage(content="no usage"))
        assert meter.usage.calls == 1
        assert meter.usage.total_tokens == 0

    def test_elapsed_is_attached(self):
        meter = UsageMeter()
        usage = meter.with_elapsed(1.23456)
        assert usage.elapsed_seconds == 1.23


def test_build_chat_model_json_mode():
    settings = Settings(api_key="dummy-key", model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE)
    plain = build_chat_model(settings, max_output_tokens=64)
    as_json = build_chat_model(settings, json_mode=True, max_output_tokens=64)

    assert plain.model == DEFAULT_MODEL
    assert plain.max_output_tokens == 64
    assert getattr(plain, "response_mime_type", None) in (None, "")
    assert as_json.response_mime_type == "application/json"
    assert as_json.max_retries == settings.max_retries
