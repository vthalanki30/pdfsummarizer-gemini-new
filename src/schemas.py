"""Structured summary output.

Models are permissive on input (LLMs return "pp. 3-4" for a page list more often
than you would like) and strict on output, so the UI can render without
defensive checks everywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_DIGITS_RE = re.compile(r"\d+")

# "Page 3 — ...", "p.3: ...", "pp. 3-4 - ..." at the start of a page note.
_PAGE_PREFIX_RE = re.compile(
    r"""^\s*
    (?:page|p{1,2})\.?\s*(\d+)
    (?:\s*(?:-|–|—|to)\s*\d+)?
    (?:\s*of\s*\d+)?
    \s*[:.\-–—)\]]*\s*
    (.*)$""",
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)

SummaryStrategy = Literal["auto", "stuff", "map_reduce", "refine"]
SummaryLength = Literal["brief", "standard", "detailed"]


@dataclass(frozen=True)
class SummarizeOptions:
    """User-facing knobs for one summarisation run."""

    strategy: SummaryStrategy = "auto"
    length: SummaryLength = "standard"
    tone: str = "executive"
    language: str = "English"
    include_page_notes: bool = True
    max_sections: int = 8


def _coerce_page_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [int(value)]
    if isinstance(value, str):
        return [int(match) for match in _DIGITS_RE.findall(value)]
    pages: list[int] = []
    for item in value:
        pages.extend(_coerce_page_list(item))
    return pages


def _coerce_page_note(value: Any) -> Any:
    """Turn a free-text note into a ``{page, note}`` pair.

    Asked to leave ``page_notes`` empty, models often fill it anyway — as
    ``["Page 1 covers the team", ...]``. Dropping those silently loses content;
    parsing the leading page reference keeps it and stays honest about the
    cases where no page can be identified (page 0 is filtered out later).
    """
    if not isinstance(value, str):
        return value
    text = value.strip().lstrip("-*•–— \t").replace("**", "")
    match = _PAGE_PREFIX_RE.match(text)
    if not match:
        return {"page": 0, "note": text}
    note = match.group(2).strip(" \t-–—:.")
    return {"page": int(match.group(1)), "note": note or text}


class Usage(BaseModel):
    """Token accounting across every model call in one summarisation."""

    model_config = ConfigDict(extra="ignore")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    elapsed_seconds: float = 0.0

    def merge(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            calls=self.calls + other.calls,
            elapsed_seconds=round(self.elapsed_seconds + other.elapsed_seconds, 2),
        )


class SectionSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str
    summary: str
    pages: list[int] = Field(default_factory=list)

    _pages = field_validator("pages", mode="before")(_coerce_page_list)


class PageNote(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page: int
    note: str

    @field_validator("page", mode="before")
    @classmethod
    def _page_int(cls, value: Any) -> int:
        if isinstance(value, int):
            return value
        digits = _DIGITS_RE.findall(str(value))
        return int(digits[0]) if digits else 0


class SummaryPayload(BaseModel):
    """Exactly what the model is asked to return."""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    overview: str = ""
    key_points: list[str] = Field(default_factory=list)
    sections: list[SectionSummary] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    page_notes: list[PageNote] = Field(default_factory=list)

    @field_validator("key_points", "action_items", "keywords", mode="before")
    @classmethod
    def _string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [line.strip(" -•\t") for line in value.splitlines() if line.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("page_notes", mode="before")
    @classmethod
    def _coerce_page_notes(cls, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, dict):
            if {"page", "note"} & set(value):
                value = [value]
            else:
                value = [{"page": page, "note": note} for page, note in value.items()]
        elif isinstance(value, str):
            value = [line for line in value.splitlines() if line.strip()]
        if not isinstance(value, list):
            return []
        return [_coerce_page_note(item) for item in value]

    @field_validator("sections", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return [] if value is None else value


def format_pages(pages: list[int]) -> str:
    """``[3]`` -> ``p.3``; ``[1,2,3,7]`` -> ``pp.1-3, 7``."""
    if not pages:
        return ""
    ordered = sorted({page for page in pages if page > 0})
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for page in ordered[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(f"{start}-{previous}" if start != previous else str(start))
        start = previous = page
    ranges.append(f"{start}-{previous}" if start != previous else str(start))
    joined = ", ".join(ranges)
    return f"p.{joined}" if len(ordered) == 1 else f"pp.{joined}"


class SummaryResult(BaseModel):
    """Final artefact: what the CLI writes and the UI renders."""

    model_config = ConfigDict(extra="ignore")

    source: str = ""
    title: str = ""
    strategy: str = ""
    model: str = ""
    page_count: int = 0
    overview: str = ""
    key_points: list[str] = Field(default_factory=list)
    sections: list[SectionSummary] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    page_notes: list[PageNote] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    warnings: list[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines: list[str] = [f"# {self.title or self.source}", ""]
        meta = [f"**Source:** {self.source}"]
        if self.page_count:
            meta.append(f"**Pages:** {self.page_count}")
        if self.strategy:
            meta.append(f"**Strategy:** `{self.strategy}`")
        if self.model:
            meta.append(f"**Model:** `{self.model}`")
        lines.append("  \n".join(meta))
        lines.append("")

        if self.overview:
            lines += ["## Overview", "", self.overview.strip(), ""]

        if self.key_points:
            lines += ["## Key points", ""]
            lines += [f"- {point}" for point in self.key_points]
            lines.append("")

        if self.sections:
            lines += ["## Sections", ""]
            for section in self.sections:
                label = format_pages(section.pages)
                heading = f"### {section.title}" + (f" ({label})" if label else "")
                lines += [heading, "", section.summary.strip(), ""]

        if self.action_items:
            lines += ["## Action items", ""]
            lines += [f"- [ ] {item}" for item in self.action_items]
            lines.append("")

        if self.page_notes:
            lines += ["## Page notes", ""]
            lines += [f"- **p.{note.page}** — {note.note}" for note in self.page_notes]
            lines.append("")

        if self.keywords:
            lines += ["## Keywords", "", ", ".join(self.keywords), ""]

        if self.warnings:
            lines += ["## Warnings", ""]
            lines += [f"- {warning}" for warning in self.warnings]
            lines.append("")

        usage = self.usage
        if usage.calls:
            lines += [
                "---",
                "",
                f"*{usage.calls} model call(s) · {usage.total_tokens:,} tokens "
                f"({usage.input_tokens:,} in / {usage.output_tokens:,} out) · "
                f"{usage.elapsed_seconds:.1f}s*",
                "",
            ]
        return "\n".join(lines).strip() + "\n"

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, ensure_ascii=False)
