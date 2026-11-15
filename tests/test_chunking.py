"""Page-aware chunking guarantees."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunking import ChunkStats, build_splitter, chunk_document, group_by_page, pack_chunks
from src.errors import ConfigError, EmptyDocumentError
from src.pdf_loader import load_pdf
from src.tokens import estimate_tokens


@pytest.fixture
def document(long_pdf: Path):
    return load_pdf(long_pdf)


def test_chunks_have_expected_metadata(document):
    chunks = chunk_document(document, chunk_tokens=250, overlap_tokens=30)
    assert chunks
    first = chunks[0]
    for key in ("source", "page", "page_label", "chunk_index", "chunk_of_page", "chars", "tokens"):
        assert key in first.metadata
    assert first.metadata["source"] == "long.pdf"
    assert first.metadata["page_label"] == "p.1"


def test_chunk_indexes_are_sequential(document):
    chunks = chunk_document(document, chunk_tokens=250, overlap_tokens=30)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_chunks_never_span_pages(document):
    chunks = chunk_document(document, chunk_tokens=250, overlap_tokens=30)
    normalised_pages = {
        page.number: " ".join(page.text.split()) for page in document.pages
    }
    for chunk in chunks:
        page_number = chunk.metadata["page"]
        assert " ".join(chunk.page_content.split()) in normalised_pages[page_number]


def test_all_text_pages_are_covered(document):
    chunks = chunk_document(document, chunk_tokens=250, overlap_tokens=30)
    covered = {chunk.metadata["page"] for chunk in chunks}
    assert covered == {page.number for page in document.pages if page.chars}


def test_token_budget_respected(document):
    chunks = chunk_document(document, chunk_tokens=250, overlap_tokens=30)
    assert all(chunk.metadata["tokens"] <= 250 * 1.2 for chunk in chunks)


def test_smaller_budget_produces_more_chunks(document):
    coarse = chunk_document(document, chunk_tokens=900, overlap_tokens=50)
    fine = chunk_document(document, chunk_tokens=200, overlap_tokens=30)
    assert len(fine) > len(coarse)


def test_group_by_page_is_ordered(document):
    chunks = chunk_document(document, chunk_tokens=250, overlap_tokens=30)
    grouped = group_by_page(chunks)
    assert sorted(grouped) == [1, 2, 3, 4, 5]
    for page_chunks in grouped.values():
        indexes = [chunk.metadata["chunk_index"] for chunk in page_chunks]
        assert indexes == sorted(indexes)


def test_chunk_stats(document):
    chunks = chunk_document(document, chunk_tokens=250, overlap_tokens=30)
    stats = ChunkStats.from_chunks(chunks)
    assert stats.count == len(chunks)
    assert stats.total_tokens == sum(chunk.metadata["tokens"] for chunk in chunks)
    assert stats.min_tokens <= stats.max_tokens
    assert stats.pages_covered == 5


def test_empty_stats():
    stats = ChunkStats.from_chunks([])
    assert stats == ChunkStats(0, 0, 0, 0, 0.0, 0)


def test_blank_pdf_raises(blank_pdf: Path):
    with pytest.raises(EmptyDocumentError):
        chunk_document(load_pdf(blank_pdf))


def test_overlap_must_be_smaller_than_chunk(document):
    with pytest.raises(ConfigError):
        chunk_document(document, chunk_tokens=300, overlap_tokens=300)


def test_negative_overlap_rejected(document):
    with pytest.raises(ConfigError):
        chunk_document(document, chunk_tokens=300, overlap_tokens=-5)


def test_tiny_chunk_budget_rejected(document):
    with pytest.raises(ConfigError):
        chunk_document(document, chunk_tokens=50, overlap_tokens=5)


def test_splitter_uses_token_length(document):
    splitter = build_splitter(400, 40)
    assert splitter._length_function is estimate_tokens
    assert splitter._chunk_size == 400


class TestPacking:
    def _chunks(self, document, chunk_tokens=400):
        return chunk_document(document, chunk_tokens=chunk_tokens, overlap_tokens=40)

    def test_packing_preserves_chunk_order(self, document):
        chunks = self._chunks(document)
        groups = pack_chunks(chunks, budget_tokens=1200)
        flattened = [index for group in groups for index in group.chunk_indexes]
        assert flattened == [chunk.metadata["chunk_index"] for chunk in chunks]

    def test_packing_covers_all_pages(self, document):
        chunks = self._chunks(document)
        groups = pack_chunks(chunks, budget_tokens=1200)
        pages = {page for group in groups for page in group.pages}
        assert pages == {chunk.metadata["page"] for chunk in chunks}

    def test_packing_keeps_total_tokens(self, document):
        chunks = self._chunks(document)
        groups = pack_chunks(chunks, budget_tokens=1200)
        assert sum(group.tokens for group in groups) == sum(
            chunk.metadata["tokens"] for chunk in chunks
        )

    def test_packing_respects_budget_except_for_oversized_chunks(self, document):
        chunks = self._chunks(document)
        groups = pack_chunks(chunks, budget_tokens=1200)
        for group in groups:
            if len(group.chunk_indexes) > 1:
                assert group.tokens <= 1200

    def test_packing_reduces_request_count(self, document):
        chunks = self._chunks(document)
        groups = pack_chunks(chunks, budget_tokens=1200)
        assert len(groups) < len(chunks)

    def test_page_markers_rendered(self, document):
        chunks = self._chunks(document)
        groups = pack_chunks(chunks, budget_tokens=1200)
        for group in groups:
            for page in group.pages:
                assert f"[Page {page}]" in group.text

    def test_page_markers_can_be_disabled(self, document):
        groups = pack_chunks(self._chunks(document), budget_tokens=1200, include_page_markers=False)
        assert all("[Page" not in group.text for group in groups)

    def test_empty_input(self):
        assert pack_chunks([], budget_tokens=1000) == []

    def test_tiny_budget_rejected(self, document):
        with pytest.raises(ConfigError):
            pack_chunks(self._chunks(document), budget_tokens=10)
