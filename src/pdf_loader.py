"""PDF text extraction with quality signals.

The loader's job is to be honest about what it got: which pages carried text,
whether the document looks like a scan, and how much boilerplate was removed.
Downstream code can then warn the user instead of summarising an empty string.
"""

from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .errors import EmptyDocumentError, EncryptedPdfError, PdfError, PdfTooLargeError
from .tokens import estimate_tokens

PdfSource = str | Path | bytes | BinaryIO

# Lines that are only a page number in one of its common disguises.
_PAGE_NUMBER_RE = re.compile(
    r"""^\s*(?:
        page\s*\d+(?:\s*(?:of|/)\s*\d+)?      # "Page 3", "Page 3 of 10"
        |\d+\s*(?:of|/)\s*\d+                  # "3 of 10"
        |[-–—\s]*\d+[-–—\s]*                   # "- 3 -", "3"
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_SPACES_RE = re.compile(r"[ \t]+$", re.MULTILINE)

RUNNING_LINE_MIN_PAGES = 3
RUNNING_LINE_RATIO = 0.6
RUNNING_LINE_MAX_CHARS = 120
SCANNED_MIN_CHARS_PER_PAGE = 40


def normalize_text(raw: str) -> str:
    """Clean extracted text without destroying its structure."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    for char in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00ad"):
        text = text.replace(char, "")
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = _TRAILING_SPACES_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class PageText:
    """One page's extracted text (1-based page number)."""

    number: int
    text: str

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def is_empty(self) -> bool:
        return not self.text

    @property
    def est_tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class PdfDocument:
    """Extracted document plus quality signals."""

    name: str
    pages: list[PageText]
    metadata: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    stripped_lines: list[str] = field(default_factory=list)
    path: Path | None = None

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_chars(self) -> int:
        return sum(page.chars for page in self.pages)

    @property
    def est_tokens(self) -> int:
        return sum(page.est_tokens for page in self.pages)

    @property
    def empty_pages(self) -> list[int]:
        return [page.number for page in self.pages if page.is_empty]

    @property
    def textless_ratio(self) -> float:
        if not self.pages:
            return 1.0
        return len(self.empty_pages) / self.page_count

    @property
    def is_probably_scanned(self) -> bool:
        """True when too little text was recovered to summarise meaningfully.

        A single short page is a sparse document, not necessarily a scan, so the
        ratio test only applies to multi-page files. A genuinely text-free file
        is always flagged.
        """
        if not self.pages:
            return True
        if self.total_chars == 0:
            return True
        thin_pages = sum(
            1 for page in self.pages if page.chars < SCANNED_MIN_CHARS_PER_PAGE
        )
        return self.page_count >= 2 and thin_pages / self.page_count >= 0.8

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def title(self) -> str:
        return self.metadata.get("title") or self.name

    def page(self, number: int) -> PageText | None:
        for page in self.pages:
            if page.number == number:
                return page
        return None

    def summary_line(self) -> str:
        return (
            f"{self.name}: {self.page_count} pages, {self.total_chars:,} chars, "
            f"~{self.est_tokens:,} tokens"
            + (f", {len(self.empty_pages)} empty page(s)" if self.empty_pages else "")
        )


def _open_reader(source: PdfSource) -> tuple[PdfReader, Path | None, str]:
    path: Path | None = None
    name = "document.pdf"

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise PdfError(f"PDF not found: {path}")
        name = path.name
        try:
            return PdfReader(str(path)), path, name
        except (PdfReadError, OSError) as exc:
            raise PdfError(f"Could not read '{name}': {exc}") from exc

    if isinstance(source, bytes):
        try:
            return PdfReader(io.BytesIO(source)), None, name
        except (PdfReadError, OSError) as exc:
            raise PdfError(f"Could not read PDF bytes: {exc}") from exc

    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("latin-1", errors="ignore")
        return _open_reader(data)

    raise PdfError(f"Unsupported PDF source type: {type(source).__name__}")


def _plain_metadata(reader: PdfReader) -> dict[str, str]:
    meta: dict[str, str] = {}
    info = reader.metadata
    if not info:
        return meta
    for key in ("title", "author", "subject", "creator", "producer"):
        value = getattr(info, key, None)
        if value:
            meta[key] = normalize_text(str(value))[:300]
    return meta


def _detect_running_lines(pages: list[PageText]) -> set[str]:
    """Find short lines repeated verbatim across most pages (headers/footers)."""
    if len(pages) < RUNNING_LINE_MIN_PAGES:
        return set()

    counts: dict[str, int] = {}
    for page in pages:
        lines = {line.strip() for line in page.text.splitlines() if line.strip()}
        for line in lines:
            if len(line) <= RUNNING_LINE_MAX_CHARS:
                counts[line] = counts.get(line, 0) + 1

    threshold = max(RUNNING_LINE_MIN_PAGES, int(len(pages) * RUNNING_LINE_RATIO + 0.5))
    return {line for line, count in counts.items() if count >= threshold}


def _strip_boilerplate(text: str, running: set[str], removed: set[str]) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if stripped in running or _PAGE_NUMBER_RE.match(stripped):
            if stripped:
                removed.add(stripped)
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    return _MULTI_BLANK_RE.sub("\n\n", cleaned).strip()


def load_pdf(
    source: PdfSource,
    *,
    name: str | None = None,
    max_pages: int | None = None,
    strip_boilerplate: bool = True,
) -> PdfDocument:
    """Extract text from a PDF path, bytes, or binary stream.

    Raises:
        PdfError: unreadable file.
        EncryptedPdfError: password protected.
        PdfTooLargeError: more pages than ``max_pages``.
    """
    reader, path, inferred_name = _open_reader(source)
    display_name = name or inferred_name

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - pypdf raises varied types here
            raise EncryptedPdfError(f"'{display_name}' is encrypted and could not be opened.") from exc
        if not unlocked:
            raise EncryptedPdfError(
                f"'{display_name}' is password protected. Remove the password and try again."
            )

    try:
        page_count = len(reader.pages)
    except PdfReadError as exc:
        raise PdfError(f"Could not count pages in '{display_name}': {exc}") from exc

    if page_count == 0:
        raise PdfError(f"'{display_name}' contains no pages.")

    if max_pages is not None and page_count > max_pages:
        raise PdfTooLargeError(
            f"'{display_name}' has {page_count} pages, above the limit of {max_pages}. "
            "Raise MAX_PAGES or split the document."
        )

    warnings: list[str] = []
    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - a single bad page must not kill the run
            warnings.append(f"Page {index} could not be read: {exc}")
            raw = ""
        pages.append(PageText(number=index, text=normalize_text(raw)))

    removed: set[str] = set()
    if strip_boilerplate:
        running = _detect_running_lines(pages)
        if running:
            pages = [
                PageText(number=p.number, text=_strip_boilerplate(p.text, running, removed))
                for p in pages
            ]

    document = PdfDocument(
        name=display_name,
        pages=pages,
        metadata=_plain_metadata(reader),
        warnings=warnings,
        stripped_lines=sorted(removed),
        path=path,
    )

    if document.is_probably_scanned:
        document.warnings.append(
            "Little or no text layer detected. This PDF is probably a scan or image export; "
            "text extraction (and therefore summarisation) will be poor. OCR the file first."
        )
    if document.stripped_lines:
        preview = ", ".join(repr(line) for line in document.stripped_lines[:3])
        document.warnings.append(
            f"Removed {len(document.stripped_lines)} repeated header/footer line(s): {preview}"
        )
    return document


def require_text(document: PdfDocument) -> PdfDocument:
    """Raise ``EmptyDocumentError`` when there is nothing worth summarising."""
    if document.total_chars == 0:
        raise EmptyDocumentError(
            f"'{document.name}' has no extractable text. If it is a scan, OCR it first."
        )
    return document
