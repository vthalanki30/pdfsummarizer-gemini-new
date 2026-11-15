"""UI helpers and a headless boot test for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import app as app_module
from app import merged_env
from src.config import Settings

APP_PATH = Path(app_module.__file__).resolve()


def test_merged_env_layers_are_ordered():
    merged = merged_env({"A": "env", "B": "env"}, {"B": "secret", "C": "secret"}, {"C": "ui"})
    assert merged == {"A": "env", "B": "secret", "C": "ui"}


def test_blank_overrides_do_not_wipe_lower_layers():
    assert merged_env({"A": "env"}, {"A": "secret"}, {"A": ""}) == {"A": "secret"}


def test_none_values_are_dropped():
    assert merged_env({"A": None, "B": "env"}) == {"B": "env"}


@pytest.fixture
def isolated_env(monkeypatch):
    """Do not let a developer's real .env leak into UI tests."""
    monkeypatch.setattr("src.config.load_dotenv", lambda *args, **kwargs: None)
    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_CHUNK_TOKENS"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_resolve_settings_applies_ui_overrides(isolated_env):
    isolated_env.setenv("GEMINI_API_KEY", "test-key-000000000000")
    settings = app_module.resolve_settings(
        {"CHUNK_TOKENS": "1234", "CHUNK_OVERLAP_TOKENS": "100", "MAX_PAGES": "7"}
    )
    assert isinstance(settings, Settings)
    assert settings.chunk_tokens == 1234
    assert settings.chunk_overlap_tokens == 100
    assert settings.max_pages == 7


def test_app_explains_a_missing_key(isolated_env):
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in error.value for error in at.error)
    assert any("Secrets" in info.value for info in at.info)


def test_app_boots_with_a_key(isolated_env):
    isolated_env.setenv("GEMINI_API_KEY", "test-key-000000000000")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value.startswith("📄")
    assert at.sidebar.selectbox[0].value == "auto"
    assert at.sidebar.checkbox[0].value is True


def test_render_result_shows_every_section():
    from src.schemas import PageNote, SectionSummary, SummaryResult, Usage

    result = SummaryResult(
        source="report.pdf",
        title="Quarterly Report",
        strategy="map_reduce",
        model="gemini-3.1-flash-lite",
        page_count=13,
        overview="Revenue grew while costs stayed flat.",
        key_points=["Revenue up 12%"],
        sections=[SectionSummary(title="Results", summary="Revenue rose.", pages=[1, 2])],
        action_items=["Approve phase two"],
        keywords=["revenue"],
        page_notes=[PageNote(page=1, note="Introduces the report")],
        usage=Usage(input_tokens=100, output_tokens=50, total_tokens=150, calls=3),
    )

    def script(module, summary) -> None:
        module.render_result(summary, download_stem="report")

    at = AppTest.from_function(
        script, args=(app_module, result), default_timeout=30
    ).run()
    assert not at.exception
    assert at.subheader[0].value == "Quarterly Report"
    markdown = "\n".join(element.value for element in at.markdown)
    assert "Revenue grew while costs stayed flat." in markdown
    assert "Results" in markdown and "pp.1-2" in markdown
    assert "[ ] Approve phase two" in markdown
    assert len(at.dataframe) == 1
    assert at.dataframe[0].value.iloc[0]["page"] == "p.1"
    assert len(at.get("download_button")) == 2
