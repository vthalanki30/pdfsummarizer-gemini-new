"""Command line entry points.

    python scripts/cli.py inspect <pdf> [--json]

Phase 2 adds ``summarize``; the shared plumbing lives here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):  # allow `python scripts/cli.py ...`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import ChunkStats, chunk_document, group_by_page  # noqa: E402
from src.config import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    DEFAULT_CHUNK_TOKENS,
    DEFAULT_MAX_PAGES,
    get_settings,
    load_env_files,
)
from src.errors import AppError, EmptyDocumentError  # noqa: E402
from src.pdf_loader import load_pdf  # noqa: E402
from src.schemas import SummarizeOptions  # noqa: E402
from src.summarizer import Summarizer  # noqa: E402


def _tuning() -> tuple[int, int, int]:
    """Read chunk/page tuning from the environment without requiring an API key."""
    load_env_files()
    chunk_tokens = int(os.environ.get("CHUNK_TOKENS") or DEFAULT_CHUNK_TOKENS)
    overlap = int(os.environ.get("CHUNK_OVERLAP_TOKENS") or DEFAULT_CHUNK_OVERLAP_TOKENS)
    max_pages = int(os.environ.get("MAX_PAGES") or DEFAULT_MAX_PAGES)
    return chunk_tokens, overlap, max_pages


def inspect_command(args: argparse.Namespace) -> int:
    chunk_tokens, overlap, max_pages = _tuning()
    chunk_tokens = args.chunk_tokens or chunk_tokens
    overlap = args.overlap_tokens if args.overlap_tokens is not None else overlap

    document = load_pdf(
        args.pdf,
        max_pages=None if args.no_page_limit else max_pages,
        strip_boilerplate=not args.keep_boilerplate,
    )
    try:
        chunks = chunk_document(document, chunk_tokens=chunk_tokens, overlap_tokens=overlap)
        chunk_error = None
    except EmptyDocumentError as exc:
        # Inspecting a text-free PDF should still report *why* it is unusable.
        chunks = []
        chunk_error = str(exc)
    stats = ChunkStats.from_chunks(chunks)
    per_page = group_by_page(chunks)
    warnings = list(document.warnings) + ([chunk_error] if chunk_error else [])

    if args.json:
        payload = {
            "name": document.name,
            "pages": document.page_count,
            "metadata": document.metadata,
            "chars": document.total_chars,
            "est_tokens": document.est_tokens,
            "empty_pages": document.empty_pages,
            "is_probably_scanned": document.is_probably_scanned,
            "warnings": warnings,
            "stripped_lines": document.stripped_lines,
            "chunking": {
                "chunk_tokens": chunk_tokens,
                "overlap_tokens": overlap,
                "count": stats.count,
                "min_tokens": stats.min_tokens,
                "max_tokens": stats.max_tokens,
                "avg_tokens": stats.avg_tokens,
                "pages_covered": stats.pages_covered,
            },
            "pages_detail": [
                {
                    "page": page.number,
                    "chars": page.chars,
                    "est_tokens": page.est_tokens,
                    "chunks": len(per_page.get(page.number, [])),
                    "empty": page.is_empty,
                }
                for page in document.pages
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print()
    print(f"PDF          : {document.name}")
    if document.path:
        print(f"path         : {document.path}")
    if title := document.metadata.get("title"):
        print(f"title        : {title}")
    if author := document.metadata.get("author"):
        print(f"author       : {author}")
    print(f"pages        : {document.page_count}")
    print(f"characters   : {document.total_chars:,}")
    print(f"est. tokens  : {document.est_tokens:,}")
    print(f"empty pages  : {', '.join(map(str, document.empty_pages)) or 'none'}")
    print(f"scanned?     : {'yes - text layer is too thin' if document.is_probably_scanned else 'no'}")
    print(
        f"chunks       : {stats.count} "
        f"(min {stats.min_tokens} / avg {stats.avg_tokens} / max {stats.max_tokens} tokens, "
        f"budget {chunk_tokens}+{overlap})"
    )

    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(f"  ! {warning}")

    print("\n  page    chars   tokens   chunks")
    print("  " + "-" * 31)
    for page in document.pages:
        print(
            f"  {page.number:>4}  {page.chars:>7,}  {page.est_tokens:>6,}  "
            f"{len(per_page.get(page.number, [])):>6}"
        )
    print()
    return 0


def summarize_command(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.model:
        settings = replace(settings, model=args.model)

    document = load_pdf(args.pdf, max_pages=settings.max_pages)
    options = SummarizeOptions(
        strategy=args.strategy,
        length=args.length,
        tone=args.tone,
        language=args.language,
        include_page_notes=not args.no_page_notes,
    )

    progress = make_progress_printer(quiet=args.quiet)
    result = Summarizer(settings).summarize(document, options, progress=progress)

    markdown = result.to_markdown()
    if args.md:
        Path(args.md).write_text(markdown, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(result.to_json(), encoding="utf-8")
    if not args.no_print:
        print(markdown)

    usage = result.usage
    print(
        f"\n[{result.strategy}] {usage.calls} call(s), "
        f"{usage.total_tokens:,} tokens, {usage.elapsed_seconds:.1f}s",
        file=sys.stderr,
    )
    if args.md:
        print(f"markdown -> {args.md}", file=sys.stderr)
    if args.json:
        print(f"json     -> {args.json}", file=sys.stderr)
    return 0


def make_progress_printer(quiet: bool):
    def progress(stage: str, done: int, total: int) -> None:
        if quiet:
            return
        labels = {"map": "reading", "refine": "refining", "reduce": "summarising"}
        print(f"\r{labels.get(stage, stage)}: {done}/{total}", end="", file=sys.stderr, flush=True)
        if done == total:
            print(file=sys.stderr)

    return progress


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-summarizer",
        description="Inspect and summarise PDFs with LangChain + Gemini.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="show extraction and chunking stats (no API calls)")
    inspect.add_argument("pdf", help="path to a PDF file")
    inspect.add_argument("--json", action="store_true", help="machine-readable output")
    inspect.add_argument("--chunk-tokens", type=int, default=0, help="override CHUNK_TOKENS")
    inspect.add_argument("--overlap-tokens", type=int, default=None, help="override CHUNK_OVERLAP_TOKENS")
    inspect.add_argument("--keep-boilerplate", action="store_true", help="keep repeated headers/footers")
    inspect.add_argument("--no-page-limit", action="store_true", help="ignore MAX_PAGES")
    inspect.set_defaults(func=inspect_command)

    summarize = subparsers.add_parser("summarize", help="summarise a PDF (calls the Gemini API)")
    summarize.add_argument("pdf", help="path to a PDF file")
    summarize.add_argument(
        "--strategy",
        choices=("auto", "stuff", "map_reduce", "refine"),
        default="auto",
        help="summarisation strategy (auto picks stuff for short documents)",
    )
    summarize.add_argument(
        "--length", choices=("brief", "standard", "detailed"), default="standard"
    )
    summarize.add_argument(
        "--tone", choices=("executive", "technical", "casual"), default="executive"
    )
    summarize.add_argument("--language", default="English", help="output language")
    summarize.add_argument("--model", default="", help="override GEMINI_MODEL")
    summarize.add_argument("--no-page-notes", action="store_true", help="skip per-page notes")
    summarize.add_argument("--md", default="", help="also write markdown to this path")
    summarize.add_argument("--json", default="", help="also write JSON to this path")
    summarize.add_argument("--no-print", action="store_true", help="do not print to stdout")
    summarize.add_argument("--quiet", action="store_true", help="no progress output")
    summarize.set_defaults(func=summarize_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, ValueError):  # redirected streams
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AppError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
