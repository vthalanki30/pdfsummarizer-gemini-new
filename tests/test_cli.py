"""CLI smoke tests (no API calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.cli import main
from src.config import Settings
from src.errors import ConfigError
from src.schemas import SummaryResult, Usage


def test_inspect_human_output(simple_pdf: Path, capsys):
    assert main(["inspect", str(simple_pdf)]) == 0
    out = capsys.readouterr().out
    assert "simple.pdf" in out
    assert "pages        : 3" in out
    assert "chunks" in out


def test_inspect_json_output(simple_pdf: Path, capsys):
    assert main(["inspect", str(simple_pdf), "--json", "--no-page-limit"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pages"] == 3
    assert payload["chunking"]["count"] >= 3
    assert len(payload["pages_detail"]) == 3
    assert payload["is_probably_scanned"] is False


def test_inspect_reports_scanned_pdf(blank_pdf: Path, capsys):
    code = main(["inspect", str(blank_pdf), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["is_probably_scanned"] is True
    assert payload["chunking"]["count"] == 0
    assert any("scan" in warning.lower() or "text" in warning.lower() for warning in payload["warnings"])
    # inspection itself succeeded: a text-free PDF is a finding, not a crash
    assert code == 0


def test_inspect_missing_file_exits_two(tmp_path: Path, capsys):
    assert main(["inspect", str(tmp_path / "missing.pdf")]) == 2
    assert "error:" in capsys.readouterr().err


def test_inspect_respects_chunk_override(simple_pdf: Path, capsys):
    main(["inspect", str(simple_pdf), "--json", "--chunk-tokens", "150", "--overlap-tokens", "20"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["chunking"]["chunk_tokens"] == 150
    assert payload["chunking"]["overlap_tokens"] == 20


class FakeSummarizer:
    """Records the options it was given and returns a canned result."""

    last_options = None
    last_document = None

    def __init__(self, settings):
        self.settings = settings

    def summarize(self, document, options=None, progress=None, chunks=None):
        FakeSummarizer.last_options = options
        FakeSummarizer.last_document = document
        if progress:
            progress("map", 1, 1)
        return SummaryResult(
            source=document.name,
            title="Fake Summary",
            strategy="stuff",
            model=self.settings.model,
            page_count=document.page_count,
            overview="A canned overview.",
            key_points=["A canned point"],
            usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15, calls=1),
        )


@pytest.fixture
def cli_with_fakes(monkeypatch):
    from scripts import cli

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(api_key="test-key"))
    monkeypatch.setattr(cli, "Summarizer", FakeSummarizer)
    return cli


def test_summarize_prints_markdown(cli_with_fakes, simple_pdf: Path, capsys):
    assert cli_with_fakes.main(["summarize", str(simple_pdf), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "# Fake Summary" in out
    assert "A canned overview." in out


def test_summarize_writes_files(cli_with_fakes, simple_pdf: Path, tmp_path: Path, capsys):
    md_path = tmp_path / "out.md"
    json_path = tmp_path / "out.json"
    code = cli_with_fakes.main(
        [
            "summarize",
            str(simple_pdf),
            "--quiet",
            "--no-print",
            "--md",
            str(md_path),
            "--json",
            str(json_path),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    assert "A canned overview." in md_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["title"] == "Fake Summary"


def test_summarize_passes_options(cli_with_fakes, simple_pdf: Path, capsys):
    cli_with_fakes.main(
        [
            "summarize",
            str(simple_pdf),
            "--quiet",
            "--no-print",
            "--strategy",
            "refine",
            "--length",
            "detailed",
            "--tone",
            "technical",
            "--language",
            "French",
            "--no-page-notes",
        ]
    )
    options = FakeSummarizer.last_options
    assert options.strategy == "refine"
    assert options.length == "detailed"
    assert options.tone == "technical"
    assert options.language == "French"
    assert options.include_page_notes is False


def test_summarize_model_override(cli_with_fakes, simple_pdf: Path, capsys):
    cli_with_fakes.main(["summarize", str(simple_pdf), "--quiet", "--no-print", "--model", "x-test"])
    assert FakeSummarizer.last_document is not None
    assert cli_with_fakes.get_settings().model != ""


def test_summarize_missing_key_exits_two(monkeypatch, simple_pdf: Path, capsys):
    from scripts import cli

    def no_settings():
        raise ConfigError("GEMINI_API_KEY is not set.")

    monkeypatch.setattr(cli, "get_settings", no_settings)
    assert cli.main(["summarize", str(simple_pdf)]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_summarize_missing_file_exits_two(cli_with_fakes, tmp_path: Path, capsys):
    assert cli_with_fakes.main(["summarize", str(tmp_path / "ghost.pdf")]) == 2
    assert "error:" in capsys.readouterr().err
