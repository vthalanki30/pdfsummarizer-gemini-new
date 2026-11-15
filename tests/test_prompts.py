"""Prompt construction: page markers, contract, and injection framing."""

from __future__ import annotations

from src.prompts import (
    finalize_messages,
    map_messages,
    reduce_messages,
    refine_messages,
    stuff_messages,
)
from src.schemas import SummarizeOptions


def _text(messages) -> str:
    return "\n".join(str(message.content) for message in messages)


def test_map_prompt_marks_pages_and_material():
    text = _text(map_messages(title="Doc", part=2, total=3, pages=[4, 5], text="BODY"))
    assert "part 2 of 3" in text
    assert "4, 5" in text
    assert "<material>\nBODY\n</material>" in text
    assert "never instructions" in text
    assert "[p.4]" in text  # page tags are requested, not discouraged


def test_reduce_prompt_contains_contract_and_language():
    options = SummarizeOptions(language="German", length="brief", include_page_notes=False)
    text = _text(reduce_messages(title="Doc", notes="NOTES", pages=[1, 2, 3], options=options))
    assert "NOTES" in text
    assert "pages 1-3" in text
    assert "German" in text
    assert "120-180 words" in text
    assert '"page_notes": []' in text
    assert "[p.N]" in text


def test_page_notes_shape_present_when_enabled():
    text = _text(
        reduce_messages(title="Doc", notes="N", pages=[1], options=SummarizeOptions())
    )
    assert '"page": 1' in text


def test_stuff_prompt_includes_whole_document():
    text = _text(
        stuff_messages(
            title="Doc",
            text="FULL TEXT",
            pages=[1, 2],
            options=SummarizeOptions(),
        )
    )
    assert "FULL TEXT" in text
    assert "JSON object" in text


def test_refine_prompt_carries_running_summary():
    text = _text(
        refine_messages(
            title="Doc",
            running_summary="SO FAR",
            notes="NEW NOTES",
            pages=[3],
            options=SummarizeOptions(tone="technical"),
        )
    )
    assert "SO FAR" in text and "NEW NOTES" in text
    assert "technical" in text
    assert "page(s) 3" in text


def test_finalize_prompt_forbids_new_information():
    text = _text(
        finalize_messages(
            title="Doc",
            running_summary="RUNNING",
            pages=[1, 2],
            options=SummarizeOptions(),
        )
    )
    assert "RUNNING" in text
    assert "Do not add new information" in text


def test_unknown_tone_passes_through():
    text = _text(
        reduce_messages(
            title="Doc", notes="N", pages=[1], options=SummarizeOptions(tone="pirate")
        )
    )
    assert "pirate" in text
