"""Page-aware chunking.

Chunks never cross page boundaries: a summary sentence must be traceable to a
page, and RAG citations are worthless if a chunk spans two pages. Overlap is
therefore applied *within* a page only.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DEFAULT_CHUNK_OVERLAP_TOKENS, DEFAULT_CHUNK_TOKENS
from .errors import ConfigError, EmptyDocumentError
from .pdf_loader import PdfDocument
from .tokens import estimate_tokens

SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
MIN_CHUNK_CHARS = 20


def build_splitter(
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> RecursiveCharacterTextSplitter:
    """A splitter whose notion of length is estimated Gemini tokens."""
    if chunk_tokens < 100:
        raise ConfigError(f"chunk_tokens must be >= 100, got {chunk_tokens}")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ConfigError(
            f"overlap_tokens must be >= 0 and < chunk_tokens ({chunk_tokens}), got {overlap_tokens}"
        )
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens,
        chunk_overlap=overlap_tokens,
        length_function=estimate_tokens,
        separators=SEPARATORS,
        keep_separator=True,
    )


def chunk_document(
    document: PdfDocument,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    min_chars: int = MIN_CHUNK_CHARS,
) -> list[Document]:
    """Split an extracted PDF into LangChain documents carrying page metadata."""
    splitter = build_splitter(chunk_tokens, overlap_tokens)
    chunks: list[Document] = []

    for page in document.pages:
        if page.chars < min_chars:
            continue
        pieces = [piece for piece in splitter.split_text(page.text) if piece.strip()]
        for index_in_page, piece in enumerate(pieces):
            text = piece.strip()
            if len(text) < min_chars:
                continue
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": document.name,
                        "page": page.number,
                        "page_label": f"p.{page.number}",
                        "chunk_index": len(chunks),
                        "chunk_of_page": index_in_page,
                        "chunks_on_page": len(pieces),
                        "chars": len(text),
                        "tokens": estimate_tokens(text),
                    },
                )
            )

    if not chunks:
        raise EmptyDocumentError(
            f"'{document.name}' produced no usable text chunks. "
            "It may be a scanned PDF with no text layer."
        )
    return chunks


@dataclass(frozen=True)
class ChunkStats:
    count: int
    total_tokens: int
    min_tokens: int
    max_tokens: int
    avg_tokens: float
    pages_covered: int

    @classmethod
    def from_chunks(cls, chunks: list[Document]) -> ChunkStats:
        if not chunks:
            return cls(0, 0, 0, 0, 0.0, 0)
        tokens = [int(chunk.metadata.get("tokens", estimate_tokens(chunk.page_content))) for chunk in chunks]
        return cls(
            count=len(chunks),
            total_tokens=sum(tokens),
            min_tokens=min(tokens),
            max_tokens=max(tokens),
            avg_tokens=round(sum(tokens) / len(tokens), 1),
            pages_covered=len({chunk.metadata.get("page") for chunk in chunks}),
        )


def group_by_page(chunks: list[Document]) -> dict[int, list[Document]]:
    """Chunks keyed by page number, preserving order."""
    grouped: dict[int, list[Document]] = {}
    for chunk in chunks:
        page = int(chunk.metadata.get("page", 0))
        grouped.setdefault(page, []).append(chunk)
    return grouped


@dataclass(frozen=True)
class ChunkGroup:
    """Adjacent chunks packed into one model request.

    ``pages`` is what makes this safe: a group may span pages, but it always
    records which pages it covers, so nothing loses its provenance.
    """

    text: str
    pages: list[int]
    tokens: int
    chunk_indexes: list[int]


def pack_chunks(
    chunks: list[Document],
    *,
    budget_tokens: int = DEFAULT_CHUNK_TOKENS,
    include_page_markers: bool = True,
) -> list[ChunkGroup]:
    """Pack consecutive chunks into budget-sized groups for the map step.

    Page-pure chunking is right for citations but produces one request per page
    on typical documents. Packing keeps the map step proportional to token count
    instead of page count, while page markers preserve attribution.
    """
    if budget_tokens < 100:
        raise ConfigError(f"budget_tokens must be >= 100, got {budget_tokens}")
    if not chunks:
        return []

    groups: list[ChunkGroup] = []
    current: list[Document] = []
    current_tokens = 0

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        parts: list[str] = []
        pages: list[int] = []
        last_page: int | None = None
        for chunk in current:
            page = int(chunk.metadata.get("page", 0))
            if page not in pages:
                pages.append(page)
            if include_page_markers and page != last_page:
                parts.append(f"[Page {page}]")
                last_page = page
            parts.append(chunk.page_content)
        groups.append(
            ChunkGroup(
                text="\n\n".join(parts),
                pages=pages,
                tokens=current_tokens,
                chunk_indexes=[int(c.metadata.get("chunk_index", -1)) for c in current],
            )
        )
        current = []
        current_tokens = 0

    for chunk in chunks:
        chunk_tokens = int(chunk.metadata.get("tokens", estimate_tokens(chunk.page_content)))
        if current and current_tokens + chunk_tokens > budget_tokens:
            flush()
        current.append(chunk)
        current_tokens += chunk_tokens
    flush()
    return groups
