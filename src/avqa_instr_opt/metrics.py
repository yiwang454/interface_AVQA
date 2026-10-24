"""Answer normalization, option formatting, and the exact-match metric.

The general normalizer follows the legacy ReAct evaluator, while Thinking
responses use DailyOmni ``eval_results.py`` final-line parsing after
``</think>``. Programs and optimization metrics share these functions so an
answer is scored consistently throughout the repository.
"""

from __future__ import annotations

import json
import re
from typing import Any

OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def extract_last_option(response: str) -> str:
    """Extract the last A-F option from the final response line.

    Kept aligned with DailyOmni's canonical eval_results.py parser so
    optimization-time supervision and offline evaluation use the same rule.
    """
    response = response.strip()
    if not response:
        return ""
    last_line = response.split("\n")[-1].strip()
    answer_match = re.search(
        r"(?i:\b(?:final\s+answer|correct\s+answer|answer)\b\s*(?:is|:)?\s*\**)\s*([A-F])\b",
        last_line,
    )
    if answer_match:
        return answer_match.group(1).upper()
    matches = re.findall(r"\b([A-F])\b", last_line)
    if not matches:
        return ""
    return matches[-1].upper()


def _strip_fences(text: str) -> str:
    """Remove optional markdown code fences from model output."""
    value = text.strip()
    if not value.startswith("```"):
        return value

    lines = value.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _first_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object embedded in text, if present."""
    value = _strip_fences(text)
    decoder = json.JSONDecoder()
    for idx, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_option_letter(answer: str) -> str:
    """Normalize model output (bare letter, JSON, or prose) to an option letter."""
    value = str(answer or "").strip()
    if not value:
        return ""

    # Qwen Thinking supervision must never fall back to an option mentioned in
    # chain-of-thought. Match DailyOmni eval_results.py: use only the text after
    # </think>, then parse the last response line with extract_last_option().
    think_split = re.split(r"</think>", value, maxsplit=1, flags=re.IGNORECASE)
    if len(think_split) == 2:
        return extract_last_option(think_split[1])

    tagged = re.search(r"<answer>\s*([A-Za-z])\s*</answer>", value, flags=re.IGNORECASE)
    if tagged:
        return tagged.group(1).upper()

    payload = _first_json_object(value)
    if payload is not None and "answer" in payload:
        nested = normalize_option_letter(str(payload.get("answer") or ""))
        if nested:
            return nested

    upper = value.upper()
    direct = re.match(r"^\s*([A-F])(?:\s*[\.)\]:-]|\s|$)", upper)
    if direct:
        return direct.group(1)

    for pattern in (
        r"\b(?:FINAL|ANSWER|OPTION|CHOICE|CHOOSE|SELECT|LABEL)\s*(?:IS|:|=)?\s*([A-F])\b",
        r"\b([A-F])\s+IS\s+(?:THE\s+)?(?:CORRECT|BEST|ANSWER)",
    ):
        match = re.search(pattern, upper)
        if match:
            return match.group(1)

    isolated = re.findall(r"\b([A-F])\b", upper)
    return isolated[0] if len(isolated) == 1 else ""


def format_options(options_json: str) -> str:
    """Format an options JSON array as stable labeled lines."""
    try:
        parsed = json.loads(options_json)
    except json.JSONDecodeError:
        return options_json
    if not isinstance(parsed, list):
        return options_json

    lines: list[str] = []
    for idx, option in enumerate(parsed):
        label = OPTION_LABELS[idx] if idx < len(OPTION_LABELS) else f"O{idx + 1}"
        option_text = str(option).strip()
        if re.match(r"^[A-Z][\.)]\s", option_text):
            lines.append(option_text)
        else:
            lines.append(f"{label}. {option_text}")
    return "\n".join(lines)


def prediction_option_letter(pred: Any) -> str:
    """Extract the scored option, preferring a raw Thinking response when present."""
    response_text = str(getattr(pred, "response_text", "") or "")
    prediction_source = (
        response_text
        if re.search(r"</think>", response_text, flags=re.IGNORECASE)
        else str(getattr(pred, "answer", "") or "")
    )
    return normalize_option_letter(prediction_source)


def avqa_metric(example: Any, pred: Any, trace: Any = None) -> float:
    """Exact-match metric on option letter."""
    gold = normalize_option_letter(str(example.answer))
    got = prediction_option_letter(pred)
    return 1.0 if gold == got else 0.0
