"""Streamlit front end for the PDF summariser.

Run locally with::

    streamlit run app.py

The app is deliberately thin: extraction, chunking and summarisation all live in
``src`` so the CLI and the UI cannot drift apart. Everything here is presentation
plus a small amount of session state.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.chunking import ChunkStats, chunk_document
from src.config import Settings, load_env_files, load_settings
from src.errors import AppError
from src.pdf_loader import PdfDocument, load_pdf
from src.schemas import SummarizeOptions, SummaryResult, format_pages
from src.summarizer import Summarizer

SECRET_KEYS = ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EMBEDDING_MODEL")
SUMMARY_STATE = "summary"
STAGE_LABELS = {"map": "Reading", "refine": "Refining", "reduce": "Summarising"}

LENGTH_OPTIONS = ("brief", "standard", "detailed")
STRATEGY_OPTIONS = ("auto", "stuff", "map_reduce", "refine")
TONE_OPTIONS = ("executive", "technical", "casual")

STRATEGY_HELP = (
    "auto picks `stuff` when the document fits one request and `map_reduce` otherwise. "
    "`refine` is slower but keeps continuity across very long documents."
)


# --------------------------------------------------------------------------- #
# Settings resolution
# --------------------------------------------------------------------------- #
def merged_env(
    base: dict[str, str],
    secrets: dict[str, str] | None = None,
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Layer configuration: process env < Streamlit secrets < UI overrides."""
    merged = {key: value for key, value in base.items() if value is not None}
    for layer in (secrets or {}, overrides or {}):
        merged.update({key: str(value) for key, value in layer.items() if value not in (None, "")})
    return merged


def _streamlit_secrets() -> dict[str, str]:
    """Read the few keys we care about from ``st.secrets``, if a file exists."""
    found: dict[str, str] = {}
    try:
        secrets = st.secrets
        for key in SECRET_KEYS:
            try:
                value = secrets.get(key)
            except Exception:  # noqa: BLE001 - some backends raise on .get
                value = None
            if value:
                found[key] = str(value)
    except Exception:  # noqa: BLE001 - no secrets.toml configured is normal
        return {}
    return found


def resolve_settings(overrides: dict[str, str] | None = None) -> Settings:
    """Build ``Settings`` from .env, the environment, Streamlit secrets and the UI."""
    load_env_files()
    return load_settings(merged_env(dict(os.environ), _streamlit_secrets(), overrides))


# --------------------------------------------------------------------------- #
# Cached document work
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, max_entries=8)
def prepare_document(
    data: bytes,
    name: str,
    max_pages: int,
    chunk_tokens: int,
    overlap_tokens: int,
) -> tuple[PdfDocument, list, ChunkStats]:
    """Extract and chunk an uploaded PDF. Cached on the file bytes and settings."""
    document = load_pdf(data, name=name, max_pages=max_pages)
    chunks = chunk_document(document, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
    return document, chunks, ChunkStats.from_chunks(chunks)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_document_summary(document: PdfDocument, stats: ChunkStats) -> None:
    columns = st.columns(4)
    columns[0].metric("Pages", f"{document.page_count:,}")
    columns[1].metric("Characters", f"{document.total_chars:,}")
    columns[2].metric("Est. tokens", f"{document.est_tokens:,}")
    columns[3].metric("Chunks", f"{stats.count:,}")

    for warning in document.warnings:
        st.warning(warning)


def render_page_preview(document: PdfDocument) -> None:
    with st.expander("Extracted text", expanded=False):
        page_number = st.selectbox(
            "Page",
            options=[page.number for page in document.pages],
            format_func=lambda number: (
                f"p.{number}"
                + (" (empty)" if document.page(number) and document.page(number).is_empty else "")
            ),
            key="page_preview_select",
        )
        page = document.page(page_number)
        if page is None or page.is_empty:
            st.caption("No text on this page — it may be an image.")
        else:
            st.text(page.text)


def render_result(result: SummaryResult, *, download_stem: str) -> None:
    st.subheader(result.title or result.source)

    meta = [f"**Pages:** {result.page_count}"]
    if result.strategy:
        meta.append(f"**Strategy:** `{result.strategy}`")
    if result.model:
        meta.append(f"**Model:** `{result.model}`")
    st.caption("  \n".join(meta))

    if result.overview:
        st.markdown(result.overview)

    if result.key_points:
        st.markdown("#### Key points")
        for point in result.key_points:
            st.markdown(f"- {point}")

    if result.sections:
        st.markdown("#### Sections")
        for section in result.sections:
            label = format_pages(section.pages)
            heading = f"**{section.title}**" + (f" — `{label}`" if label else "")
            st.markdown(heading)
            st.markdown(section.summary)

    if result.action_items:
        st.markdown("#### Action items")
        for item in result.action_items:
            st.markdown(f"- [ ] {item}")

    if result.page_notes:
        st.markdown("#### Page notes")
        st.dataframe(
            [
                {"page": f"p.{note.page}", "note": note.note}
                for note in result.page_notes
            ],
            use_container_width=True,
            hide_index=True,
        )

    if result.keywords:
        st.markdown("#### Keywords")
        st.markdown(" · ".join(f"`{keyword}`" for keyword in result.keywords))

    for warning in result.warnings:
        st.warning(warning)

    usage = result.usage
    if usage.calls:
        st.caption(
            f"{usage.calls} model call(s) · {usage.total_tokens:,} tokens "
            f"({usage.input_tokens:,} in / {usage.output_tokens:,} out) · "
            f"{usage.elapsed_seconds:.1f}s"
        )

    markdown_column, json_column = st.columns(2)
    markdown_column.download_button(
        "Download Markdown",
        data=result.to_markdown(),
        file_name=f"{download_stem}.md",
        mime="text/markdown",
        use_container_width=True,
        key="download-md",
    )
    json_column.download_button(
        "Download JSON",
        data=result.to_json(),
        file_name=f"{download_stem}.json",
        mime="application/json",
        use_container_width=True,
        key="download-json",
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar_controls(settings: Settings) -> tuple[dict[str, str], SummarizeOptions]:
    st.sidebar.header("Summary options")

    strategy = st.sidebar.selectbox(
        "Strategy", STRATEGY_OPTIONS, index=0, help=STRATEGY_HELP
    )
    length = st.sidebar.radio("Length", LENGTH_OPTIONS, index=1, horizontal=True)
    tone = st.sidebar.selectbox("Tone", TONE_OPTIONS, index=0)
    language = st.sidebar.text_input("Output language", "English")
    include_page_notes = st.sidebar.checkbox("Per-page notes", value=True)

    with st.sidebar.expander("Advanced"):
        chunk_tokens = st.slider(
            "Chunk budget (tokens)", 500, 6000, settings.chunk_tokens, step=250
        )
        overlap = st.slider(
            "Chunk overlap (tokens)",
            0,
            600,
            min(settings.chunk_overlap_tokens, 600),
            step=25,
            help="Overlap is applied within a page; chunks never span pages.",
        )
        max_pages = st.number_input(
            "Max pages", min_value=1, max_value=2000, value=settings.max_pages, step=1
        )
        concurrency = st.slider(
            "Parallel requests", 1, 8, settings.max_concurrency, step=1
        )
        if overlap >= chunk_tokens:
            overlap = max(0, chunk_tokens - 100)
            st.caption(f"Overlap clamped to {overlap} tokens so it stays below the budget.")

    st.sidebar.caption(
        f"Model `{settings.model}` · key `{settings.masked_key}`"
    )

    overrides = {
        "CHUNK_TOKENS": str(chunk_tokens),
        "CHUNK_OVERLAP_TOKENS": str(overlap),
        "MAX_PAGES": str(max_pages),
        "MAX_CONCURRENCY": str(concurrency),
    }
    options = SummarizeOptions(
        strategy=strategy,
        length=length,
        tone=tone,
        language=language.strip() or "English",
        include_page_notes=include_page_notes,
    )
    return overrides, options


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="PDF Summarizer · LangChain + Gemini",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 PDF Summarizer")
    st.caption(
        "Upload a PDF for an executive summary, key points, section breakdown and "
        "page-level notes. Only the extracted text is sent to the Gemini API."
    )

    try:
        settings = resolve_settings()
    except AppError as exc:
        st.error(str(exc))
        st.info(
            "Local runs: copy `.env.example` to `.env` and set `GEMINI_API_KEY`. "
            "Streamlit Cloud: add the key in **App settings → Secrets**."
        )
        st.stop()
        return

    overrides, options = sidebar_controls(settings)

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded is None:
        st.info("No file selected yet. Long documents are summarised in parallel batches.")
        return

    data = uploaded.getvalue()
    signature = f"{uploaded.name}:{len(data)}"
    stored = st.session_state.get(SUMMARY_STATE)
    if stored and stored.get("signature") != signature:
        st.session_state.pop(SUMMARY_STATE, None)

    with st.spinner("Extracting text…"):
        try:
            document, chunks, stats = prepare_document(
                data,
                uploaded.name,
                int(overrides["MAX_PAGES"]),
                int(overrides["CHUNK_TOKENS"]),
                int(overrides["CHUNK_OVERLAP_TOKENS"]),
            )
        except AppError as exc:
            st.error(str(exc))
            return

    render_document_summary(document, stats)
    render_page_preview(document)

    left, right = st.columns([1, 3])
    run = left.button("Summarise", type="primary", use_container_width=True)
    right.caption(
        f"{stats.count} chunk(s) of ~{stats.avg_tokens} tokens · "
        f"strategy `{options.strategy}` · length `{options.length}`"
    )

    if run:
        try:
            run_settings = resolve_settings(overrides)
        except AppError as exc:
            st.error(str(exc))
            return

        progress = st.progress(0.0, text="Starting…")

        def report(stage: str, done: int, total: int) -> None:
            label = STAGE_LABELS.get(stage, stage.title())
            progress.progress(
                min(done / total, 1.0) if total else 1.0,
                text=f"{label}: {done}/{total}",
            )

        try:
            result = Summarizer(run_settings).summarize(
                document, options, chunks=chunks, progress=report
            )
        except AppError as exc:
            progress.empty()
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface anything unexpected
            progress.empty()
            st.error(f"Unexpected error: {exc}")
            return

        progress.empty()
        st.session_state[SUMMARY_STATE] = {"signature": signature, "result": result}

    stored = st.session_state.get(SUMMARY_STATE)
    if stored and stored.get("signature") == signature:
        st.divider()
        render_result(stored["result"], download_stem=Path(uploaded.name).stem)
        st.caption("Document Q&A lands in Phase 4.")


if __name__ == "__main__":
    main()
