"""PDF extraction, normalisation, and quality signals."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.errors import EmptyDocumentError, EncryptedPdfError, PdfError, PdfTooLargeError
from src.pdf_loader import PageText, load_pdf, normalize_text, require_text
from tests.conftest import encrypt_pdf, write_pdf


def test_extracts_each_page(simple_pdf: Path):
    document = load_pdf(simple_pdf)
    assert document.page_count == 3
    assert document.name == "simple.pdf"
    assert document.path == simple_pdf
    assert "ALPHA" in document.pages[0].text
    assert "GAMMA" in document.pages[2].text


def test_page_numbers_are_one_based(simple_pdf: Path):
    document = load_pdf(simple_pdf)
    assert [page.number for page in document.pages] == [1, 2, 3]


def test_text_property_joins_pages(simple_pdf: Path):
    document = load_pdf(simple_pdf)
    assert "ALPHA" in document.text and "GAMMA" in document.text
    assert document.total_chars == sum(page.chars for page in document.pages)


def test_page_marked_text_labels_every_page(simple_pdf: Path):
    document = load_pdf(simple_pdf)
    marked = document.page_marked_text
    assert marked.index("[Page 1]") < marked.index("ALPHA")
    assert marked.index("[Page 2]") < marked.index("BETA")
    assert marked.index("[Page 3]") < marked.index("GAMMA")


def test_loads_from_bytes(simple_pdf: Path):
    document = load_pdf(simple_pdf.read_bytes(), name="upload.pdf")
    assert document.name == "upload.pdf"
    assert document.page_count == 3
    assert document.path is None


def test_loads_from_stream(simple_pdf: Path):
    with simple_pdf.open("rb") as handle:
        document = load_pdf(handle, name="stream.pdf")
    assert document.page_count == 3
    assert document.name == "stream.pdf"


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(PdfError, match="not found"):
        load_pdf(tmp_path / "ghost.pdf")


def test_garbage_bytes_raise(tmp_path: Path):
    with pytest.raises(PdfError):
        load_pdf(b"not a pdf at all")


def test_page_limit_is_enforced(simple_pdf: Path):
    with pytest.raises(PdfTooLargeError) as excinfo:
        load_pdf(simple_pdf, max_pages=2)
    assert "3 pages" in str(excinfo.value)


def test_page_limit_allows_exact_size(simple_pdf: Path):
    assert load_pdf(simple_pdf, max_pages=3).page_count == 3


def test_encrypted_pdf_is_reported(simple_pdf: Path, tmp_path: Path):
    locked = encrypt_pdf(simple_pdf, tmp_path / "locked.pdf")
    with pytest.raises(EncryptedPdfError, match="password protected"):
        load_pdf(locked)


def test_blank_pdf_flagged_as_scanned(blank_pdf: Path):
    document = load_pdf(blank_pdf)
    assert document.page_count == 3
    assert document.total_chars == 0
    assert document.is_probably_scanned
    assert document.textless_ratio == 1.0
    assert any("scan" in warning.lower() for warning in document.warnings)


def test_require_text_raises_on_empty(blank_pdf: Path):
    with pytest.raises(EmptyDocumentError, match="no extractable text"):
        require_text(load_pdf(blank_pdf))


def test_require_text_passes_when_text_present(simple_pdf: Path):
    document = load_pdf(simple_pdf)
    assert require_text(document) is document


def test_healthy_pdf_not_flagged(simple_pdf: Path):
    document = load_pdf(simple_pdf)
    assert not document.is_probably_scanned
    assert document.empty_pages == []


def test_running_header_and_page_numbers_removed(header_pdf: Path):
    document = load_pdf(header_pdf)
    assert "ACME CONFIDENTIAL - INTERNAL USE ONLY" in document.stripped_lines
    assert "Body 1" in document.pages[0].text
    assert "ACME CONFIDENTIAL" not in document.pages[0].text
    assert any("header/footer" in warning for warning in document.warnings)


def test_boilerplate_can_be_kept(header_pdf: Path):
    document = load_pdf(header_pdf, strip_boilerplate=False)
    assert "ACME CONFIDENTIAL" in document.pages[0].text
    assert document.stripped_lines == []


def test_metadata_and_title(simple_pdf: Path):
    document = load_pdf(simple_pdf)
    assert isinstance(document.metadata, dict)
    assert document.title


def test_summary_line_reports_counts(simple_pdf: Path):
    line = load_pdf(simple_pdf).summary_line()
    assert "simple.pdf" in line
    assert "3 pages" in line


def test_page_lookup(simple_pdf: Path):
    document = load_pdf(simple_pdf)
    assert document.page(2) is not None
    assert "BETA" in document.page(2).text
    assert document.page(99) is None


def test_single_page_document(tmp_path: Path):
    document = load_pdf(write_pdf(tmp_path / "one.pdf", ["Solo page with enough text to count."]))
    assert document.page_count == 1
    assert not document.is_probably_scanned


class TestNormalizeText:
    def test_empty(self):
        assert normalize_text("") == ""

    def test_ligatures_and_smart_quotes(self):
        assert normalize_text("ﬁle ’quoted’ “text”") == "file 'quoted' \"text\""

    def test_non_breaking_space_and_soft_hyphen(self):
        assert normalize_text("a\xa0b\u00adc") == "a bc"  # NBSP -> space, soft hyphen dropped

    def test_zero_width_characters_removed(self):
        assert normalize_text("ab\u200bcd\ufeff") == "abcd"

    def test_collapses_blank_line_runs(self):
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_strips_trailing_whitespace(self):
        assert normalize_text("a   \nb\t\n") == "a\nb"

    def test_windows_line_endings(self):
        assert normalize_text("a\r\nb") == "a\nb"


def test_page_text_properties():
    page = PageText(number=1, text="hello")
    assert page.chars == 5
    assert page.est_tokens >= 1
    assert not page.is_empty
    assert PageText(number=2, text="").is_empty
