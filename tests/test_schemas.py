"""Summary schema coercion, rendering, and formatting."""

from __future__ import annotations

import json

import pytest

from src.schemas import (
    PageNote,
    SectionSummary,
    SummarizeOptions,
    SummaryPayload,
    SummaryResult,
    Usage,
    format_pages,
)


class TestFormatPages:
    @pytest.mark.parametrize(
        ("pages", "expected"),
        [
            ([], ""),
            ([3], "p.3"),
            ([1, 2, 3], "pp.1-3"),
            ([1, 2, 5], "pp.1-2, 5"),
            ([7, 3, 4], "pp.3-4, 7"),
            ([0, -2], ""),
        ],
    )
    def test_ranges(self, pages, expected):
        assert format_pages(pages) == expected


def test_pages_coerced_from_strings():
    section = SectionSummary(title="A", summary="S", pages="pp. 3-4")
    assert section.pages == [3, 4]


def test_pages_coerced_from_mixed_list():
    section = SectionSummary(title="A", summary="S", pages=["p.2", 5, "7-8"])
    assert section.pages == [2, 5, 7, 8]


def test_page_note_coerces_page_string():
    assert PageNote(page="p.4", note="x").page == 4
    assert PageNote(page=4, note="x").page == 4


def test_string_lists_are_split():
    payload = SummaryPayload(
        key_points="- one\n- two\n• three",
        action_items="single action",
    )
    assert payload.key_points == ["one", "two", "three"]
    assert payload.action_items == ["single action"]


def test_none_lists_become_empty():
    payload = SummaryPayload(sections=None, page_notes=None, keywords=None)
    assert payload.sections == [] and payload.page_notes == [] and payload.keywords == []


def test_page_notes_as_prefixed_strings_are_parsed():
    payload = SummaryPayload(
        page_notes=[
            "Page 1 — Introduces the report",
            "**p.3**: Revenue grew 12%",
            "pp. 4-5 - Quarterly context",
        ]
    )
    assert [(note.page, note.note) for note in payload.page_notes] == [
        (1, "Introduces the report"),
        (3, "Revenue grew 12%"),
        (4, "Quarterly context"),
    ]


def test_page_notes_without_a_page_reference_keep_page_zero():
    payload = SummaryPayload(page_notes=["A general remark"])
    assert len(payload.page_notes) == 1
    assert payload.page_notes[0].page == 0
    assert payload.page_notes[0].note == "A general remark"


def test_page_notes_as_page_to_note_mapping():
    payload = SummaryPayload(page_notes={"2": "Second page", "4": "Fourth page"})
    assert [(note.page, note.note) for note in payload.page_notes] == [
        (2, "Second page"),
        (4, "Fourth page"),
    ]


def test_page_notes_as_single_dict():
    payload = SummaryPayload(page_notes={"page": "p.7", "note": "Seventh"})
    assert len(payload.page_notes) == 1
    assert payload.page_notes[0].page == 7


def test_page_notes_as_multiline_string():
    payload = SummaryPayload(page_notes="Page 1 — first\nPage 2 — second")
    assert [note.page for note in payload.page_notes] == [1, 2]


def test_unknown_keys_ignored():
    payload = SummaryPayload(overview="x", surprise="ignored")
    assert not hasattr(payload, "surprise")


def test_usage_merge_sums_everything():
    merged = Usage(input_tokens=10, output_tokens=5, total_tokens=15, calls=1).merge(
        Usage(input_tokens=2, output_tokens=3, total_tokens=5, calls=1)
    )
    assert merged.input_tokens == 12
    assert merged.output_tokens == 8
    assert merged.total_tokens == 20
    assert merged.calls == 2


def _result(**overrides) -> SummaryResult:
    base = dict(
        source="report.pdf",
        title="Quarterly Report",
        strategy="map_reduce",
        model="gemini-3.1-flash-lite",
        page_count=3,
        overview="Revenue grew while costs stayed flat.",
        key_points=["Revenue up 12%", "Costs flat"],
        sections=[SectionSummary(title="Results", summary="Revenue rose.", pages=[1, 2])],
        action_items=["Approve phase two"],
        keywords=["revenue", "costs"],
        page_notes=[PageNote(page=1, note="Introduces the report")],
        usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150, calls=3, elapsed_seconds=2.5),
        warnings=["A warning"],
    )
    base.update(overrides)
    return SummaryResult(**base)


def test_markdown_contains_every_section():
    markdown = _result().to_markdown()
    assert markdown.startswith("# Quarterly Report")
    assert "## Overview" in markdown
    assert "Revenue up 12%" in markdown
    assert "### Results (pp.1-2)" in markdown
    assert "- [ ] Approve phase two" in markdown
    assert "**p.1**" in markdown
    assert "revenue, costs" in markdown
    assert "3 model call(s)" in markdown
    assert "A warning" in markdown


def test_markdown_omits_empty_sections():
    markdown = _result(action_items=[], keywords=[], page_notes=[], warnings=[]).to_markdown()
    assert "## Action items" not in markdown
    assert "## Keywords" not in markdown
    assert "## Page notes" not in markdown
    assert "## Warnings" not in markdown


def test_markdown_falls_back_to_source_as_title():
    assert _result(title="").to_markdown().startswith("# report.pdf")


def test_json_roundtrips():
    original = _result()
    restored = SummaryResult.model_validate(json.loads(original.to_json()))
    assert restored == original


def test_default_options():
    options = SummarizeOptions()
    assert options.strategy == "auto"
    assert options.length == "standard"
    assert options.include_page_notes is True
