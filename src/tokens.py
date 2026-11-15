"""Token accounting.

Gemini does not use an OpenAI tokenizer, so shipping ``tiktoken`` would give a
confidently wrong number. Instead we estimate from character volume (~4 chars
per token for prose, fewer for whitespace-heavy tables) and, where an exact
count changes behaviour, ask Gemini's ``countTokens`` endpoint.
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate for budgeting and chunk sizing.

    Deliberately errs high: underestimating a chunk costs a failed request,
    overestimating costs a little extra headroom.
    """
    if not text:
        return 0
    chars = len(text)
    whitespace = sum(text.count(ch) for ch in " \n\t")
    # Dense whitespace (tables, code) tokenises less efficiently than prose.
    ratio = CHARS_PER_TOKEN
    if chars and whitespace / chars > 0.25:
        ratio = 3.2
    return max(1, math.ceil(chars / ratio))


def exact_tokens(text: str, model: str, *, api_key: str | None = None, timeout: float = 20.0) -> int | None:
    """Exact token count from Gemini, or ``None`` if unavailable.

    Network failures and unsupported models degrade to ``None`` so callers can
    fall back to :func:`estimate_tokens` instead of crashing a summarisation.
    """
    if not text:
        return 0
    try:
        from google import genai

        client = genai.Client(api_key=api_key) if api_key else genai.Client()
        response = client.models.count_tokens(model=model, contents=text)
        total = getattr(response, "total_tokens", None)
        return int(total) if total is not None else None
    except Exception:  # noqa: BLE001 - accounting must never break the pipeline
        return None
