"""Official Vertex AI Gemini ``generateContent`` transport for local AVQA media.

Local video and audio are encoded with Vertex's native ``inlineData`` shape,
so public runs do not depend on a project-specific Cloud Storage mirror.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.parse
from pathlib import Path
from threading import Lock
from typing import Any

import requests

from .response_quality import degenerate_response_reason


_PRINTED_FIRST_PROMPT = False
_PRINT_LOCK = Lock()


def _mime_type(path: str) -> str:
    overrides = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    return (
        overrides.get(Path(path).suffix.lower())
        or mimetypes.guess_type(path)[0]
        or "application/octet-stream"
    )


def inline_data_from_path(path: str, mime_type: str | None = None) -> dict[str, Any]:
    """Encode a local media file using Gemini's native ``inlineData`` shape."""
    with open(path, "rb") as handle:
        data = base64.b64encode(handle.read()).decode("utf-8")
    return {"inlineData": {"mimeType": mime_type or _mime_type(path), "data": data}}


def _response_text_parts(payload: dict[str, Any]) -> tuple[str, str]:
    candidates = payload.get("candidates") or []
    if not candidates:
        return "", ""
    parts = candidates[0].get("content", {}).get("parts", [])
    visible: list[str] = []
    thoughts: list[str] = []
    for part in parts:
        text = part.get("text") if isinstance(part, dict) else None
        if isinstance(text, str):
            (thoughts if part.get("thought") else visible).append(text)
    return "".join(visible), "".join(thoughts)


def _usage(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usageMetadata")
    return usage if isinstance(usage, dict) else {}


def _safe_error(exc: BaseException, limit: int = 1200) -> str:
    text = str(exc).replace("\n", " ")
    text = re.sub(r"(?i)\bbearer\s+[^\s,;)}\]]+", "Bearer <redacted>", text)
    text = re.sub(
        r"(?i)(api[_-]?key|authorization|bearer|token|password|secret)"
        r"(\s*[=:]\s*|\s+)([^\s,;)}\]]+)",
        r"\1\2<redacted>",
        text,
    )
    return text if len(text) <= limit else text[:limit] + "... <truncated>"


def _service_account_access_token(credentials_path: str) -> str:
    """Mint a Vertex OAuth token from the service-account mechanism used by sibling."""
    try:
        from google.auth.transport.requests import Request as GoogleAuthRequest
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(GoogleAuthRequest())
        return str(credentials.token)
    except ImportError as exc:
        raise RuntimeError(
            "Official Vertex Gemini requires google-auth for service-account credentials."
        ) from exc


def _vertex_headers(api_key: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key.strip():
        headers["x-goog-api-key"] = api_key
        return headers
    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not path:
        raise RuntimeError(
            "Official Vertex Gemini requires a Google API key or GOOGLE_APPLICATION_CREDENTIALS."
        )
    headers["Authorization"] = f"Bearer {_service_account_access_token(path)}"
    return headers


def _request_url_and_headers(
    *, base_url: str, model: str, api_key: str,
) -> tuple[str, dict[str, str]]:
    if not base_url.strip():
        raise ValueError("Vertex Gemini profile has no base URL; set GEMINI_BASE_URL.")
    return (
        f"{base_url.rstrip('/')}/models/{urllib.parse.quote(model, safe='')}:generateContent",
        _vertex_headers(api_key),
    )


def _redact_inline_data(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_inline_data(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                f"<base64 redacted; {len(item)} chars>"
                if key == "data" and isinstance(item, str)
                else _redact_inline_data(item)
            )
            for key, item in value.items()
        }
    return value


def _maybe_print_first_prompt(data: dict[str, Any], url: str, enabled: bool) -> None:
    global _PRINTED_FIRST_PROMPT
    if not enabled or _PRINTED_FIRST_PROMPT:
        return
    with _PRINT_LOCK:
        if _PRINTED_FIRST_PROMPT:
            return
        print("[Gemini first prompt]", flush=True)
        print(f"Request URL: {url}", flush=True)
        print(json.dumps(_redact_inline_data(data), ensure_ascii=False, indent=2), flush=True)
        _PRINTED_FIRST_PROMPT = True


def call_gemini_messages(
    contents: list[dict[str, Any]],
    *,
    system_prompt: str = "",
    model: str,
    api_key: str,
    base_url: str,
    timeout: int = 180,
    max_retries: int = 3,
    retry_delay_s: float = 5.0,
    include_thoughts: bool = True,
    thinking_budget: int | None = None,
    return_thinking: bool = False,
    temperature: float = 0.6,
    gemini_seed: int | None = None,
    top_p: float = 0.95,
    top_k: int | None = None,
    max_tokens: int = 1024,
    retry_degenerate_response: bool = True,
    print_first_prompt: bool = False,
) -> tuple[str, dict[str, Any]] | tuple[str, dict[str, Any], str]:
    """Call the official Vertex Gemini ``generateContent`` endpoint."""
    url, headers = _request_url_and_headers(
        base_url=base_url, model=model, api_key=api_key,
    )
    data: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
        },
    }
    if top_k is not None:
        data["generationConfig"]["topK"] = top_k
    if gemini_seed is not None:
        data["generationConfig"]["seed"] = gemini_seed
    if system_prompt:
        data["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if thinking_budget is not None or include_thoughts:
        thinking_config: dict[str, Any] = {}
        if thinking_budget is not None:
            thinking_config["thinkingBudget"] = thinking_budget
        if include_thoughts:
            thinking_config["includeThoughts"] = True
        data["generationConfig"]["thinkingConfig"] = thinking_config
    _maybe_print_first_prompt(data, url, print_first_prompt)

    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            if not response.ok:
                raise RuntimeError(
                    f"Gemini API error {response.status_code}: {response.text[:2000]}"
                )
            response_json = response.json()
            response_text, thinking_text = _response_text_parts(response_json)
            token_usage = _usage(response_json)
            reason = (
                degenerate_response_reason(response_text, token_usage, max_tokens=max_tokens)
                if retry_degenerate_response
                else None
            )
            if reason and attempt < max_retries:
                delay = retry_delay_s * (2 ** (attempt - 1))
                print(
                    f"[warn] Gemini API attempt {attempt}/{max_retries} returned a "
                    f"degenerate caption: {reason}. Retrying in {delay}s.",
                    flush=True,
                )
                time.sleep(delay)
                continue
            if reason:
                print(
                    f"[warn] Gemini API exhausted {max_retries} attempts after a "
                    f"degenerate caption: {reason}. Keeping the final response.",
                    flush=True,
                )
            if return_thinking:
                return response_text, token_usage, thinking_text
            return response_text, token_usage
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                delay = retry_delay_s * (2 ** (attempt - 1))
                print(
                    f"[warn] Gemini API attempt {attempt}/{max_retries} failed: "
                    f"{_safe_error(exc)}. Retrying in {delay}s.",
                    flush=True,
                )
                time.sleep(delay)

    if last_error is not None:
        raise RuntimeError(
            f"Gemini API call failed: {_safe_error(last_error, 2000)}"
        ) from last_error
    raise RuntimeError("Gemini API call failed")
