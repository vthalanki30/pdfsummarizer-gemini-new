"""Live tests: these call the real Gemini API.

Run explicitly with ``pytest -m live``. They are skipped in normal runs and in CI
because they cost tokens and depend on network and quota.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import get_settings
from src.pdf_loader import load_pdf
from src.schemas import SummarizeOptions
from src.summarizer import Summarizer

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def settings():
    return get_settings()


def test_live_stuff_summary(simple_pdf: Path, settings):
    result = Summarizer(settings).summarize(
        load_pdf(simple_pdf),
        SummarizeOptions(strategy="stuff", length="brief"),
    )
    assert result.overview
    assert result.title
    assert result.usage.calls >= 1
    assert result.usage.total_tokens > 0
    assert result.page_count == 3


def test_live_map_reduce_summary(long_pdf: Path, settings):
    result = Summarizer(settings).summarize(
        load_pdf(long_pdf),
        SummarizeOptions(strategy="map_reduce", length="brief", include_page_notes=False),
    )
    assert result.strategy == "map_reduce"
    assert result.overview
    assert result.usage.calls >= 2
    assert all(1 <= page <= 5 for section in result.sections for page in section.pages)


def test_live_refine_summary(simple_pdf: Path, settings):
    result = Summarizer(settings).summarize(
        load_pdf(simple_pdf),
        SummarizeOptions(strategy="refine", length="brief", include_page_notes=False),
    )
    assert result.strategy == "refine"
    assert result.overview
