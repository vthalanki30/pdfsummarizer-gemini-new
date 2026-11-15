"""Token estimation behaviour."""

from __future__ import annotations

from src.tokens import CHARS_PER_TOKEN, estimate_tokens, exact_tokens


def test_empty_text_is_zero():
    assert estimate_tokens("") == 0


def test_short_text_is_at_least_one():
    assert estimate_tokens("hi") == 1


def test_prose_lands_near_four_chars_per_token():
    text = "The quick brown fox jumps over the lazy dog. " * 20
    assert estimate_tokens(text) == round(len(text) / CHARS_PER_TOKEN)


def test_whitespace_heavy_text_costs_more_tokens():
    prose = "abcdefghijklmnopqrstuvwxyz" * 10
    table = "a b c d e f g h i j k l m n o p q r s t " * 10
    assert estimate_tokens(table) > estimate_tokens(prose)


def test_estimate_grows_monotonically():
    samples = ["one", "one two", "one two three", "one two three four"]
    counts = [estimate_tokens(sample) for sample in samples]
    assert counts == sorted(counts)


def test_exact_tokens_short_circuits_empty_text():
    assert exact_tokens("", "gemini-3.1-flash-lite") == 0


def test_exact_tokens_never_raises_on_failure(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("google"):
            raise ImportError("simulated missing sdk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert exact_tokens("some text", "gemini-3.1-flash-lite") is None
