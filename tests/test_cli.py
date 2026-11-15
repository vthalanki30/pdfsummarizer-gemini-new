"""CLI smoke tests (no API calls)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.cli import main


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
