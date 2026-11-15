"""Shared test fixtures: real PDFs generated with reportlab, no network needed."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

LINE_LIMIT = 95


def pytest_collection_modifyitems(config, items):
    """Skip `live` tests unless the user asked for them with ``-m live``."""
    marker_expression = getattr(config.option, "markexpr", "") or ""
    if "live" in marker_expression:
        return
    skip = pytest.mark.skip(reason="live test: run with `pytest -m live`")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


def write_pdf(path: Path, pages: list[str]) -> Path:
    """Write a PDF where each entry in ``pages`` becomes one page of text."""
    pdf = canvas.Canvas(str(path), pagesize=LETTER)
    for text in pages:
        y = 750
        for raw_line in text.splitlines() or [""]:
            for line in textwrap.wrap(raw_line, LINE_LIMIT) or [""]:
                pdf.drawString(72, y, line)
                y -= 14
                if y < 72:
                    break
        pdf.showPage()
    pdf.save()
    return path


def encrypt_pdf(source: Path, target: Path, password: str = "s3cret") -> Path:
    reader = PdfReader(str(source))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(password)
    with target.open("wb") as handle:
        writer.write(handle)
    return target


@pytest.fixture
def simple_pdf(tmp_path: Path) -> Path:
    pages = [
        "ALPHA report introduction. This document explains the quarterly results, "
        "the operating context, and the decisions taken by the leadership team.",
        "BETA section. Revenue grew across all regions, with the strongest gains in "
        "the northern territory. Costs stayed flat year over year despite expansion.",
        "GAMMA conclusion. The board approved the next phase of investment and asked "
        "management to report back in ninety days with updated projections.",
    ]
    return write_pdf(tmp_path / "simple.pdf", pages)


@pytest.fixture
def long_pdf(tmp_path: Path) -> Path:
    paragraph = (
        "Section {n}. This paragraph exists purely to give the chunker enough text to "
        "split on. It repeats a handful of ordinary words so that sentence boundaries, "
        "commas, and whitespace offer realistic split points for the recursive splitter. "
    )
    pages = [paragraph.format(n=i) * 20 for i in range(1, 6)]
    return write_pdf(tmp_path / "long.pdf", pages)


@pytest.fixture
def blank_pdf(tmp_path: Path) -> Path:
    return write_pdf(tmp_path / "scanned.pdf", ["", "", ""])


@pytest.fixture
def header_pdf(tmp_path: Path) -> Path:
    header = "ACME CONFIDENTIAL - INTERNAL USE ONLY"
    pages = []
    for number in range(1, 6):
        pages.append(
            f"{header}\n"
            f"Body {number}: the project status is on track and the budget is unchanged. "
            "Risks were reviewed and no new blockers were identified by the team.\n"
            f"Page {number} of 5"
        )
    return write_pdf(tmp_path / "headers.pdf", pages)
