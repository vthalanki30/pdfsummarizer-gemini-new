# PDF Summarizer (LangChain + Gemini)

End-to-end PDF summarisation and document Q&A built with **LangChain**, **Google Gemini**, and **Streamlit**.

Upload a PDF and get an executive summary, key points, section breakdown, and page-by-page notes —
then ask questions about the document with page-level citations.

## Status

🚧 **Phase 0 — environment & repository bootstrap** (in progress)

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Validate Gemini access + GitHub auth, repo bootstrap | 🚧 in progress |
| 1 | PDF ingestion, page-aware chunking, token accounting | ⬜ planned |
| 2 | Summarisation engine (stuff / map-reduce / refine) | ⬜ planned |
| 3 | Streamlit UI | ⬜ planned |
| 4 | RAG Q&A with page citations | ⬜ planned |
| 5 | Hardening, CI, Streamlit Cloud deploy | ⬜ planned |

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
| `MAX_CONCURRENCY` | `4` | Parallel map-step requests |

Secrets live in `.env`, which is git-ignored. Never commit it.

## License

MIT — see [LICENSE](LICENSE).
