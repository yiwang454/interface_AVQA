"""Helpers for detecting malformed multimodal-model completions."""

from __future__ import annotations

import re
from typing import Any, Mapping


_OUTPUT_TOKEN_KEYS = (
    "completion_tokens",
    "output_tokens",
    "candidatesTokenCount",
    "candidates_token_count",
    "outputTokenCount",
)
_CJK_CHARACTER_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def completion_token_count(usage: Mapping[str, Any] | None) -> int | None:
    """Return a provider-neutral generated-token count when one is available."""
    if not usage:
        return None
    for key in _OUTPUT_TOKEN_KEYS:
        value = usage.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _text_unit_count(text: str) -> int:
    """Approximate words while treating contiguous CJK characters as text units."""
    cjk_count = len(_CJK_CHARACTER_RE.findall(text))
    non_cjk_text = _CJK_CHARACTER_RE.sub(" ", text)
    return cjk_count + len(re.findall(r"\w+", non_cjk_text, flags=re.UNICODE))


def degenerate_response_reason(
    response_text: str | None,
    token_usage: Mapping[str, Any] | None,
    *,
    max_tokens: int,
) -> str | None:
    """Return a retry reason only when the response is empty after stripping."""

    stripped = str(response_text or "").strip()
    if not stripped:
        return "response is empty after strip"

    # Disabled: completion-token counts are not reliable enough to judge the
    # amount of text in a non-empty response. Keep every non-empty response.
    # completion_tokens = completion_token_count(token_usage)
    # meaningful_output_floor = max(128, max_tokens // 4)
    # text_units = _text_unit_count(stripped)
    # if (
    #     completion_tokens is not None
    #     and completion_tokens >= meaningful_output_floor
    #     and text_units * 50 <= completion_tokens
    # ):
    #     return (
    #         f"response has only {text_units} text units for "
    #         f"completion_tokens={completion_tokens}"
    #     )
    return None
