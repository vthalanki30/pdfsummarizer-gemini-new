# PDF Summarizer (LangChain + Gemini)

End-to-end PDF summarisation and document Q&A built with **LangChain**, **Google Gemini**, and **Streamlit**.

Upload a PDF and get an executive summary, key points, section breakdown, and page-by-page notes —
then ask questions about the document with page-level citations.

## Status

✅ **Phase 2 — summarisation engine** complete (CLI only; UI lands in Phase 3)

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Validate Gemini access + GitHub auth, repo bootstrap | ✅ complete |
| 1 | PDF ingestion, page-aware chunking, token accounting | ✅ complete |
| 2 | Summarisation engine (stuff / map-reduce / refine) | ✅ complete |
| 3 | Streamlit UI | 🚧 in progress |
| 4 | RAG Q&A with page citations | ⬜ planned |
| 5 | Hardening, CI, Streamlit Cloud deploy | ⬜ planned |

## Try it now

Inspect any PDF (no API calls, no key required):

```bash
python scripts/cli.py inspect path/to/document.pdf
python scripts/cli.py inspect path/to/document.pdf --json
```

```
PDF          : 15246372705.4Q1.pdf
pages        : 13
characters   : 21,466
est. tokens  : 5,372
empty pages  : none
scanned?     : no
chunks       : 13 (min 146 / avg 413.2 / max 652 tokens, budget 3000+200)
```

The loader reports what it actually recovered: pages with no text, documents that look
like scans (which need OCR before summarising), and repeated headers/footers it stripped.

Summarise one (needs `GEMINI_API_KEY`):

```bash
python scripts/cli.py summarize path/to/document.pdf
python scripts/cli.py summarize path/to/document.pdf --length detailed --md out.md --json out.json
python scripts/cli.py summarize path/to/document.pdf --strategy map_reduce --no-page-notes
```

```
# Ancillary Sections of Food and Beverage Service Operations

**Source:** 15246372705.4Q1.pdf
**Pages:** 13
**Strategy:** `map_reduce`
**Model:** `gemini-3.1-flash-lite`

## Overview
...
## Sections
### Ancillary Support Units (pp.1-6)
...
## Page notes
- **p.1** — Introduces the project team and defines the role of ancillary departments.
...

---
*5 model call(s) · 9,083 tokens (7,412 in / 1,671 out) · 75.3s*
```

Options: `--strategy auto|stuff|map_reduce|refine`, `--length brief|standard|detailed`,
`--tone executive|technical|casual`, `--language`, `--model`, `--no-page-notes`,
`--md FILE`, `--json FILE`, `--quiet`.

## Project layout

```
app.py                  # Streamlit UI (Phase 3)
src/config.py           # validated settings from env / .env
src/pdf_loader.py       # extraction, normalisation, quality signals
src/chunking.py         # page-aware chunking, packing, statistics
src/tokens.py           # token estimates, exact counts via countTokens
src/llm.py              # Gemini model construction, usage accounting
src/prompts.py          # prompt templates + the JSON contract
src/schemas.py          # summary payload/result models, page coercion
src/summarizer.py       # stuff / map-reduce / refine pipelines
src/errors.py           # user-presentable error types
scripts/validate_env.py # Phase 0 gate: keys, API, model, embeddings
scripts/cli.py          # inspect + summarize
tests/                  # pytest suite, real PDFs generated with reportlab
```


## Requirements

- Python 3.11+
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

## Setup

```bash
# 1. Create a virtual environment
py -3.11 -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure secrets
copy .env.example .env            # Windows
# cp .env.example .env            # macOS / Linux
#    then edit .env and set GEMINI_API_KEY

# 4. Validate the environment (Phase 0 gate)
python scripts/validate_env.py
```

`scripts/validate_env.py` checks the key is present and well-formed, the API answers,
`GEMINI_MODEL` actually exists for your key, inference works, and the embedding model
needed for Q&A is reachable. It exits non-zero on any failure, so it is CI-friendly.
Use `--offline` for offline presence/format checks and `--json` for machine-readable output.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GEMINI_API_KEY` | *(required)* | Gemini API key |
| `GEMINI_MODEL` | `gemini-3.1-flash-lite` | Chat model for summarisation / Q&A |
| `GEMINI_EMBEDDING_MODEL` | `gemini-embedding-001` | Embeddings for Q&A retrieval |
| `MAX_PAGES` | `120` | Page guardrail |
| `CHUNK_TOKENS` | `3000` | Target tokens per chunk |
| `CHUNK_OVERLAP_TOKENS` | `200` | Overlap between chunks (within a page) |
| `MAX_CONCURRENCY` | `4` | Parallel map-step requests |
| `MAX_RETRIES` | `3` | SDK retries per model call |
| `TEMPERATURE` | `0.2` | Sampling temperature |
| `STUFF_MAX_TOKENS` | `18000` | `--strategy auto` uses `stuff` below this |

Token counts are estimates (≈4 characters per token) rather than a foreign tokenizer's
count — exact counts are fetched from Gemini's `countTokens` endpoint only where accuracy
changes behaviour.

Secrets live in `.env`, which is git-ignored. Never commit it.

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest                      # 151 unit tests, no network needed
python -m pytest -m live              # opt-in end-to-end tests that hit the real API
python -m ruff check .
```

Tests generate their own PDFs with `reportlab`, so the suite runs offline and needs no fixtures
from disk.

## How chunking works

Text is split per page, never across pages — a claim in a summary must point at one page, and a
retrieval hit is useless if its citation straddles two. Overlap is applied *within* a page, so
context still flows between neighbouring chunks of the same page.

Because of that, a 300-page PDF yields 300 page-chunks even at a 3,000-token budget. The
summarisation map step therefore packs adjacent page-chunks into budget-sized groups (keeping the
list of pages each group covers) instead of sending one request per page.

## How summarisation works

Three strategies, all on LangChain runnables and `BaseChatModel.batch` rather than the legacy
`load_summarize_chain` helper:

| Strategy | Calls | Use it when |
| --- | --- | --- |
| `stuff` | 1 | The document fits one request (`auto` picks it under `STUFF_MAX_TOKENS`) |
| `map_reduce` | groups + 1 | Long documents; parallel and cheapest per page |
| `refine` | groups × 2 + 1 | Long documents where continuity matters more than cost |

`--strategy auto` estimates total tokens and picks `stuff` for short PDFs, `map_reduce` otherwise.

Page attribution is enforced end to end: chunks carry page numbers, the stuff prompt marks every
page with `[Page N]`, map notes are asked to prefix each bullet with `[p.N]`, and the reduce prompt
receives notes labelled with their page ranges. The result model then clamps any page the model
cites to the document's real page range and drops the rest, so a fabricated `p.99` never reaches
the output.

The final response is constrained to a JSON object (`response_mime_type="application/json"`),
validated with pydantic, and retried once with a repair instruction if the model returns malformed
JSON. Coercion is deliberately lenient on input — `"pp. 3-4"`, `[3, 4]`, and a list of
`"Page 1 — ..."` strings all parse — and strict on output.

## License

MIT — see [LICENSE](LICENSE).
