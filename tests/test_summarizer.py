"""Summarisation pipelines, exercised against a deterministic fake model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.chunking import chunk_document, pack_chunks
from src.config import Settings
from src.errors import EmptyDocumentError, SummarizationError
from src.pdf_loader import PdfDocument, load_pdf
from src.schemas import SummarizeOptions
from src.summarizer import Summarizer
from tests.fakes import ScriptedChatModel, is_map_prompt

PAYLOAD = {
    "title": "Quarterly Review",
    "overview": "Revenue grew 12% while costs stayed flat, so the board approved phase two.",
    "key_points": ["Revenue grew 12%", "Costs stayed flat"],
    "sections": [{"title": "Results", "summary": "Revenue rose across regions.", "pages": [1, 2]}],
    "action_items": ["Approve phase two"],
    "keywords": ["Revenue", "Costs"],
    "page_notes": [{"page": 1, "note": "Introduces the report"}],
}
NOTES = "- revenue grew 12%\n- costs stayed flat"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        api_key="test-key",
        chunk_tokens=600,
        chunk_overlap_tokens=60,
        max_concurrency=2,
        max_retries=0,
        temperature=0.0,
        stuff_max_tokens=10_000,
    )


def make_model(responder) -> ScriptedChatModel:
    return ScriptedChatModel(responder=responder)


def standard_responder(messages):
    prompt = str(messages[-1].content)
    return NOTES if is_map_prompt(prompt) else json.dumps(PAYLOAD)


def summarizer_for(settings: Settings, model: ScriptedChatModel) -> Summarizer:
    return Summarizer(settings, llm=model, json_llm=model)


class TestStrategies:
    def test_stuff_is_a_single_call(self, settings, long_pdf: Path):
        model = make_model(standard_responder)
        result = summarizer_for(settings, model).summarize(
            load_pdf(long_pdf), SummarizeOptions(strategy="stuff")
        )
        assert len(model.calls) == 1
        assert result.strategy == "stuff"
        assert result.overview.startswith("Revenue grew")

    def test_auto_picks_stuff_for_short_documents(self, settings, simple_pdf: Path):
        model = make_model(standard_responder)
        result = summarizer_for(settings, model).summarize(load_pdf(simple_pdf))
        assert result.strategy == "stuff"
        assert len(model.calls) == 1

    def test_auto_picks_map_reduce_for_long_documents(self, settings, long_pdf: Path):
        tight = Settings(
            api_key=settings.api_key,
            chunk_tokens=600,
            chunk_overlap_tokens=60,
            max_concurrency=2,
            stuff_max_tokens=1000,
        )
        model = make_model(standard_responder)
        result = summarizer_for(tight, model).summarize(load_pdf(long_pdf))
        assert result.strategy == "map_reduce"

    def test_explicit_strategy_overrides_auto(self, settings, simple_pdf: Path):
        model = make_model(standard_responder)
        result = summarizer_for(settings, model).summarize(
            load_pdf(simple_pdf), SummarizeOptions(strategy="map_reduce")
        )
        assert result.strategy == "map_reduce"

    def test_map_reduce_calls_once_per_packed_group(self, settings, long_pdf: Path):
        document = load_pdf(long_pdf)
        groups = pack_chunks(
            chunk_document(document, chunk_tokens=600, overlap_tokens=60), budget_tokens=600
        )
        model = make_model(standard_responder)
        result = summarizer_for(settings, model).summarize(
            document, SummarizeOptions(strategy="map_reduce")
        )
        assert len(groups) > 1
        assert len(model.calls) == len(groups) + 1
        assert result.strategy == "map_reduce"

    def test_map_prompts_carry_page_markers(self, settings, long_pdf: Path):
        model = make_model(standard_responder)
        summarizer_for(settings, model).summarize(
            load_pdf(long_pdf), SummarizeOptions(strategy="map_reduce")
        )
        map_calls = [call for call in model.calls if is_map_prompt(call)]
        assert map_calls
        assert any("[Page 1]" in call for call in map_calls)

    def test_reduce_prompt_labels_note_blocks_with_page_ranges(self, settings, long_pdf: Path):
        document = load_pdf(long_pdf)
        chunks = chunk_document(document, chunk_tokens=600, overlap_tokens=60)
        groups = pack_chunks(chunks, budget_tokens=600)
        model = make_model(standard_responder)
        summarizer_for(settings, model).summarize(
            document, SummarizeOptions(strategy="map_reduce")
        )
        reduce_prompt = next(call for call in model.calls if not is_map_prompt(call))
        assert f"Notes 1 (page {groups[0].pages[0]})" in reduce_prompt
        assert f"Notes {len(groups)}" in reduce_prompt

    def test_refine_makes_sequential_calls(self, settings, long_pdf: Path):
        document = load_pdf(long_pdf)
        groups = pack_chunks(
            chunk_document(document, chunk_tokens=600, overlap_tokens=60), budget_tokens=600
        )
        model = make_model(standard_responder)
        result = summarizer_for(settings, model).summarize(
            document, SummarizeOptions(strategy="refine")
        )
        assert result.strategy == "refine"
        # notes per group, one refinement per group, one final JSON conversion
        assert len(model.calls) == len(groups) * 2 + 1


class TestJsonHandling:
    def test_invalid_json_is_repaired_once(self, settings, simple_pdf: Path):
        state = {"json_calls": 0}

        def responder(messages):
            prompt = str(messages[-1].content)
            if is_map_prompt(prompt):
                return NOTES
            state["json_calls"] += 1
            return "Sorry, I cannot do that." if state["json_calls"] == 1 else json.dumps(PAYLOAD)

        model = make_model(responder)
        result = summarizer_for(settings, model).summarize(load_pdf(simple_pdf))
        assert state["json_calls"] == 2
        assert result.title == "Quarterly Review"

    def test_json_in_code_fences_is_accepted(self, settings, simple_pdf: Path):
        fenced = "```json\n" + json.dumps(PAYLOAD) + "\n```"
        model = make_model(lambda messages: fenced)
        result = summarizer_for(settings, model).summarize(load_pdf(simple_pdf))
        assert result.overview

    def test_double_failure_raises(self, settings, simple_pdf: Path):
        model = make_model(lambda messages: "still not json")
        with pytest.raises(SummarizationError, match="valid JSON"):
            summarizer_for(settings, model).summarize(load_pdf(simple_pdf))

    def test_empty_overview_raises(self, settings, simple_pdf: Path):
        payload = {**PAYLOAD, "overview": "", "sections": []}
        model = make_model(lambda messages: json.dumps(payload))
        with pytest.raises(SummarizationError, match="empty overview"):
            summarizer_for(settings, model).summarize(load_pdf(simple_pdf))

    def test_overview_falls_back_to_first_section(self, settings, simple_pdf: Path):
        payload = {**PAYLOAD, "overview": ""}
        model = make_model(lambda messages: json.dumps(payload))
        result = summarizer_for(settings, model).summarize(load_pdf(simple_pdf))
        assert result.overview == "Revenue rose across regions."


class TestAssembly:
    def test_pages_are_clamped_deduplicated_and_sorted(self, settings, long_pdf: Path):
        payload = {
            **PAYLOAD,
            "sections": [{"title": "S", "summary": "x", "pages": [99, 2, 0, "p.2", 3]}],
            "page_notes": [
                {"page": 0, "note": "invalid"},
                {"page": 3, "note": "third"},
                {"page": 3, "note": "duplicate"},
                {"page": 99, "note": "invalid"},
                {"page": 1, "note": "first"},
            ],
        }
        model = make_model(lambda messages: json.dumps(payload))
        result = summarizer_for(settings, model).summarize(
            load_pdf(long_pdf), SummarizeOptions(include_page_notes=True, strategy="stuff")
        )
        assert result.sections[0].pages == [2, 3]
        assert [note.page for note in result.page_notes] == [1, 3]
        assert result.page_notes[0].note == "first"

    def test_page_notes_can_be_disabled(self, settings, simple_pdf: Path):
        model = make_model(lambda messages: json.dumps(PAYLOAD))
        result = summarizer_for(settings, model).summarize(
            load_pdf(simple_pdf), SummarizeOptions(include_page_notes=False)
        )
        assert result.page_notes == []

    def test_keywords_lowercased_and_deduplicated(self, settings, simple_pdf: Path):
        payload = {**PAYLOAD, "keywords": ["Revenue", "revenue", " COSTS ", ""]}
        model = make_model(lambda messages: json.dumps(payload))
        result = summarizer_for(settings, model).summarize(load_pdf(simple_pdf))
        assert result.keywords == ["revenue", "costs"]

    def test_lists_are_capped(self, settings, simple_pdf: Path):
        payload = {
            **PAYLOAD,
            "key_points": [f"point {index}" for index in range(30)],
            "action_items": [f"action {index}" for index in range(30)],
            "keywords": [f"kw{index}" for index in range(30)],
        }
        model = make_model(lambda messages: json.dumps(payload))
        result = summarizer_for(settings, model).summarize(load_pdf(simple_pdf))
        assert len(result.key_points) == 12
        assert len(result.action_items) == 10
        assert len(result.keywords) == 12

    def test_title_falls_back_to_document_title(self, settings, simple_pdf: Path):
        model = make_model(lambda messages: json.dumps({**PAYLOAD, "title": ""}))
        result = summarizer_for(settings, model).summarize(load_pdf(simple_pdf))
        assert result.title

    def test_usage_is_accumulated_and_timed(self, settings, simple_pdf: Path):
        model = make_model(lambda messages: json.dumps(PAYLOAD))
        result = summarizer_for(settings, model).summarize(load_pdf(simple_pdf))
        assert result.usage.calls == 1
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.usage.elapsed_seconds >= 0

    def test_document_warnings_flow_through(self, settings, simple_pdf: Path):
        document = load_pdf(simple_pdf)
        document.warnings.append("synthetic warning")
        model = make_model(lambda messages: json.dumps(PAYLOAD))
        result = summarizer_for(settings, model).summarize(document)
        assert "synthetic warning" in result.warnings


class TestFailures:
    def test_model_exception_is_wrapped(self, settings, long_pdf: Path):
        def responder(messages):
            raise RuntimeError("quota exceeded")

        model = make_model(responder)
        with pytest.raises(SummarizationError, match="quota exceeded"):
            summarizer_for(settings, model).summarize(
                load_pdf(long_pdf), SummarizeOptions(strategy="map_reduce")
            )

    def test_all_empty_notes_raise(self, settings, long_pdf: Path):
        def responder(messages):
            prompt = str(messages[-1].content)
            return "" if is_map_prompt(prompt) else json.dumps(PAYLOAD)

        model = make_model(responder)
        with pytest.raises(SummarizationError, match="no notes"):
            summarizer_for(settings, model).summarize(
                load_pdf(long_pdf), SummarizeOptions(strategy="map_reduce")
            )

    def test_scanned_pdf_is_rejected(self, settings, blank_pdf: Path):
        model = make_model(standard_responder)
        with pytest.raises(EmptyDocumentError):
            summarizer_for(settings, model).summarize(load_pdf(blank_pdf))

    def test_page_less_document_is_rejected(self, settings):
        model = make_model(standard_responder)
        with pytest.raises(EmptyDocumentError):
            summarizer_for(settings, model).summarize(PdfDocument(name="empty.pdf", pages=[]))


def test_progress_reports_every_group(settings, long_pdf: Path):
    document = load_pdf(long_pdf)
    groups = pack_chunks(
        chunk_document(document, chunk_tokens=600, overlap_tokens=60), budget_tokens=600
    )
    events: list[tuple[str, int, int]] = []
    model = make_model(standard_responder)
    summarizer_for(settings, model).summarize(
        document,
        SummarizeOptions(strategy="map_reduce"),
        progress=lambda stage, done, total: events.append((stage, done, total)),
    )
    assert events
    assert all(stage == "map" for stage, _, _ in events)
    assert events[-1][1] == events[-1][2] == len(groups)


def test_injected_chunks_skip_rechunking(settings, long_pdf: Path):
    document = load_pdf(long_pdf)
    chunks = chunk_document(document, chunk_tokens=600, overlap_tokens=60)
    model = make_model(standard_responder)
    summarizer_for(settings, model).summarize(
        document, SummarizeOptions(strategy="map_reduce"), chunks=chunks
    )
    assert any(is_map_prompt(call) for call in model.calls)
