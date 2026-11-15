"""Summarisation pipelines: stuff, map-reduce, refine.

Implemented directly on LCEL primitives plus ``BaseChatModel.batch`` rather than
LangChain's legacy ``load_summarize_chain`` — that helper moved out of
``langchain-core`` in v1, and doing it here keeps retries, concurrency, token
accounting and progress reporting under our control.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import ValidationError

from .chunking import ChunkGroup, chunk_document, pack_chunks
from .config import Settings, get_settings
from .errors import EmptyDocumentError, SummarizationError
from .llm import UsageMeter, build_chat_model, message_text
from .pdf_loader import PdfDocument
from .prompts import (
    finalize_messages,
    map_messages,
    reduce_messages,
    refine_messages,
    stuff_messages,
)
from .schemas import (
    PageNote,
    SectionSummary,
    SummarizeOptions,
    SummaryPayload,
    SummaryResult,
)

ProgressCallback = Callable[[str, int, int], None]

MAX_KEY_POINTS = 12
MAX_ACTION_ITEMS = 10
MAX_KEYWORDS = 12
MAX_SECTIONS = 10
NOTES_MAX_TOKENS = 2048
JSON_MAX_TOKENS = 8192
MAP_BULLETS = 7

REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON. Return only the JSON object described "
    "above — no code fences, no commentary, no trailing text."
)


class Summarizer:
    """Turns an extracted PDF into a :class:`SummaryResult`."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: BaseChatModel | None = None,
        json_llm: BaseChatModel | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self._json_llm = json_llm or llm

    # ------------------------------------------------------------------ models
    @property
    def text_model(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = build_chat_model(self.settings, max_output_tokens=NOTES_MAX_TOKENS)
        return self._llm

    @property
    def json_model(self) -> BaseChatModel:
        if self._json_llm is None:
            self._json_llm = build_chat_model(
                self.settings, json_mode=True, max_output_tokens=JSON_MAX_TOKENS
            )
        return self._json_llm

    # --------------------------------------------------------------- public API
    def summarize(
        self,
        document: PdfDocument,
        options: SummarizeOptions | None = None,
        *,
        chunks: list[Document] | None = None,
        progress: ProgressCallback | None = None,
    ) -> SummaryResult:
        options = options or SummarizeOptions()
        started = time.perf_counter()

        if not document.pages:
            raise EmptyDocumentError(f"'{document.name}' has no pages to summarise.")

        chunks = chunks if chunks is not None else chunk_document(
            document,
            chunk_tokens=self.settings.chunk_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        if not chunks:
            raise EmptyDocumentError(
                f"'{document.name}' produced no text chunks, so there is nothing to summarise."
            )

        warnings = list(document.warnings)
        meter = UsageMeter()
        estimated_tokens = sum(
            int(chunk.metadata.get("tokens", 0)) or len(chunk.page_content) // 4 for chunk in chunks
        )
        strategy = self._resolve_strategy(options.strategy, estimated_tokens)

        try:
            if strategy == "stuff":
                payload = self._run_json(
                    stuff_messages(
                        title=document.title,
                        text=document.page_marked_text,
                        pages=[page.number for page in document.pages],
                        options=options,
                    ),
                    meter,
                )
            else:
                groups = pack_chunks(chunks, budget_tokens=self.settings.chunk_tokens)
                notes = self._run_map(groups, document, options, meter, progress)
                if strategy == "refine":
                    running = self._run_refine(
                        notes, groups, document, options, meter, progress
                    )
                    payload = self._run_json(
                        finalize_messages(
                            title=document.title,
                            running_summary=running,
                            pages=[page.number for page in document.pages],
                            options=options,
                        ),
                        meter,
                    )
                else:
                    payload = self._run_json(
                        reduce_messages(
                            title=document.title,
                            notes=self._combine_notes(notes, groups),
                            pages=[page.number for page in document.pages],
                            options=options,
                        ),
                        meter,
                    )
        except SummarizationError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalise SDK failures for the UI
            raise SummarizationError(f"Summarisation failed: {exc}") from exc

        result = self._build_result(
            payload,
            document=document,
            strategy=strategy,
            warnings=warnings,
            include_page_notes=options.include_page_notes,
        )
        elapsed = time.perf_counter() - started
        return result.model_copy(update={"usage": meter.with_elapsed(elapsed)})

    # ------------------------------------------------------------------ stages
    def _resolve_strategy(self, requested: str, estimated_tokens: int) -> str:
        if requested != "auto":
            return requested
        if estimated_tokens <= self.settings.stuff_max_tokens:
            return "stuff"
        return "map_reduce"

    def _run_map(
        self,
        groups: list[ChunkGroup],
        document: PdfDocument,
        options: SummarizeOptions,
        meter: UsageMeter,
        progress: ProgressCallback | None,
    ) -> list[str]:
        prompts = [
            map_messages(
                title=document.title,
                part=index,
                total=len(groups),
                pages=group.pages,
                text=group.text,
                bullets=MAP_BULLETS,
            )
            for index, group in enumerate(groups, start=1)
        ]

        notes: list[str] = []
        batch_size = max(1, self.settings.max_concurrency)
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            try:
                messages = self.text_model.batch(
                    batch, config={"max_concurrency": self.settings.max_concurrency}
                )
            except Exception as exc:  # noqa: BLE001
                raise SummarizationError(f"Reading part of the document failed: {exc}") from exc
            for message in messages:
                meter.add(message)
                notes.append(message_text(message))
            if progress:
                progress("map", min(start + batch_size, len(prompts)), len(prompts))

        cleaned = [
            note if note else f"(no notes returned for pages {', '.join(map(str, group.pages))})"
            for note, group in zip(notes, groups, strict=False)
        ]
        if all(not note for note in notes):
            raise SummarizationError(
                "The model returned no notes for any part of the document. "
                "Try a smaller CHUNK_TOKENS value or a better text extract."
            )
        return cleaned

    def _run_refine(
        self,
        notes: list[str],
        groups: list[ChunkGroup],
        document: PdfDocument,
        options: SummarizeOptions,
        meter: UsageMeter,
        progress: ProgressCallback | None,
    ) -> str:
        running = ""
        for index, (note, group) in enumerate(zip(notes, groups, strict=False), start=1):
            messages = refine_messages(
                title=document.title,
                running_summary=running,
                notes=note,
                pages=group.pages,
                options=options,
            )
            running = self._invoke(self.text_model, messages, meter)
            if progress:
                progress("refine", index, len(notes))
        return running

    def _combine_notes(self, notes: list[str], groups: list[ChunkGroup]) -> str:
        blocks: list[str] = []
        for index, (note, group) in enumerate(zip(notes, groups, strict=False), start=1):
            if not group.pages:
                header = f"Notes {index}:"
            elif len(group.pages) == 1:
                header = f"Notes {index} (page {group.pages[0]}):"
            else:
                header = f"Notes {index} (pages {min(group.pages)}-{max(group.pages)}):"
            blocks.append(f"{header}\n{note}")
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------- calls
    def _invoke(
        self,
        model: BaseChatModel,
        messages: list[BaseMessage],
        meter: UsageMeter,
    ) -> str:
        try:
            message = model.invoke(messages)
        except Exception as exc:  # noqa: BLE001
            raise SummarizationError(f"Model call failed: {exc}") from exc
        meter.add(message)
        text = message_text(message)
        if not text:
            raise SummarizationError("The model returned an empty response.")
        return text

    def _run_json(self, messages: list[BaseMessage], meter: UsageMeter) -> SummaryPayload:
        raw = self._invoke(self.json_model, messages, meter)
        parsed = self._parse_json(raw)
        if parsed is None:
            repair_messages = [
                *messages,
                AIMessage(content=raw[:6000]),
                HumanMessage(content=REPAIR_INSTRUCTION),
            ]
            raw = self._invoke(self.json_model, repair_messages, meter)
            parsed = self._parse_json(raw)
        if parsed is None:
            raise SummarizationError(
                "The model did not return valid JSON after two attempts. "
                "Try again, or switch to another model in GEMINI_MODEL."
            )
        try:
            return SummaryPayload.model_validate(parsed)
        except ValidationError as exc:
            first = exc.errors()[0] if exc.errors() else {}
            raise SummarizationError(
                f"The model's JSON did not match the expected shape: "
                f"{first.get('loc')} {first.get('msg')}"
            ) from exc

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    # ---------------------------------------------------------------- assembly
    def _build_result(
        self,
        payload: SummaryPayload,
        *,
        document: PdfDocument,
        strategy: str,
        warnings: list[str],
        include_page_notes: bool,
    ) -> SummaryResult:
        page_count = document.page_count

        def valid_pages(pages: list[int]) -> list[int]:
            return sorted({page for page in pages if 1 <= page <= page_count})

        sections = [
            SectionSummary(
                title=section.title.strip() or f"Section {index}",
                summary=section.summary.strip(),
                pages=valid_pages(section.pages),
            )
            for index, section in enumerate(payload.sections[:MAX_SECTIONS], start=1)
            if section.summary.strip()
        ]

        overview = payload.overview.strip()
        if not overview and sections:
            overview = sections[0].summary
        if not overview:
            raise SummarizationError(
                "The model returned an empty overview. The document may be too short "
                "or the extract too noisy to summarise."
            )

        seen_keywords: set[str] = set()
        keywords: list[str] = []
        for keyword in payload.keywords:
            normalised = keyword.strip().lower().strip(".,;:")
            if normalised and normalised not in seen_keywords:
                seen_keywords.add(normalised)
                keywords.append(normalised)

        seen_notes: dict[int, PageNote] = {}
        for note in payload.page_notes if include_page_notes else []:
            text = note.note.strip()
            if text and 1 <= note.page <= page_count and note.page not in seen_notes:
                seen_notes[note.page] = PageNote(page=note.page, note=text)

        return SummaryResult(
            source=document.name,
            title=(payload.title.strip() or document.title)[:200],
            strategy=strategy,
            model=self.settings.model,
            page_count=page_count,
            overview=overview,
            key_points=[point.strip() for point in payload.key_points[:MAX_KEY_POINTS] if point.strip()],
            sections=sections,
            action_items=[
                item.strip() for item in payload.action_items[:MAX_ACTION_ITEMS] if item.strip()
            ],
            keywords=keywords[:MAX_KEYWORDS],
            page_notes=[seen_notes[page] for page in sorted(seen_notes)],
            warnings=warnings,
        )
