"""Low-level Qwen3-Omni (OpenAI-compatible) transport.

Ported from react-agent-avqa-optimize's ``dspy_avqa/qwen3omni_api.py``. Kept:
the streaming/non-streaming call path, the empty-response detection and
degenerate-response retry loop, and the video/audio media-ordering helpers
that ``clients.reliable.ReliableQwenExecutor`` drives during batch fallback.

Dropped relative to the source module: the ReAct-era convenience wrappers
(``call_qwen``, ``call_qwen_serial``, ``call_qwen3omni``) and the regex-based
choice-letter parser (``parse_qwen_content``/``parse_qwen_response``). Answer
parsing for this repo goes through the single ``metrics.normalize_option_letter``
implementation instead, so direct-Qwen and caption-then-text answers are
normalized identically to how ``avqa_metric`` scores them.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .response_quality import degenerate_response_reason

class EmptyQwenResponseError(RuntimeError):
    """Raised when every Qwen attempt returns empty visible content."""

    def __init__(self, message: str, response_attempts: list[dict[str, Any]]) -> None:
        """Initialize the error with structured response-attempt diagnostics."""
        super().__init__(message)
        self.response_attempts = response_attempts


def safe_exception_summary(exc: BaseException, *, limit: int = 1200) -> str:
    """Return a bounded, credential-safe exception summary for experiment logs."""
    message = str(exc)
    # OpenAI-compatible errors can echo request fragments. Do not let a
    # credential or a base64 media URL escape into a run log.
    message = re.sub(r"data:[^,\s]+,[A-Za-z0-9+/=]+", "<redacted:data-url>", message)
    message = re.sub(
        r"(?i)(api[_-]?key|authorization|bearer|token|password|secret)"
        r"(\s*[=:]\s*|\s+)([^\s,;)}\]]+)",
        r"\1\2<redacted>",
        message,
    )
    message = re.sub(r"(?i)(https?://)[^/@\s:]+:[^/@\s]+@", r"\1<redacted>@", message)
    if len(message) > limit:
        message = message[:limit] + "... <truncated>"
    return f"{type(exc).__name__}: {message}"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _client(*, api_key: str, base_url: str) -> OpenAI:
    """Build a client only from an explicitly resolved Qwen profile.

    Local OpenAI-compatible servers commonly ignore authentication, but the
    OpenAI Python client still requires a non-empty placeholder.  The Qwen YAML
    therefore uses ``qwen_api_key: EMPTY`` and that literal value is valid.
    """
    if not api_key or not api_key.strip():
        raise ValueError(
            "Qwen profile has no api_key; set qwen_api_key: EMPTY in the model "
            "YAML for a local server that does not authenticate."
        )
    if not base_url or not base_url.strip():
        raise ValueError("Qwen profile has no base_url; set qwen_base_url in the model YAML.")
    return OpenAI(api_key=api_key, base_url=base_url)


def to_data_url(
    path: str,
    mime_type: Optional[str] = None,
    *,
    media_url_mode: str | None = None,
) -> str:
    media_url_mode = (
        media_url_mode or os.getenv("QWEN_MEDIA_URL_MODE", "data_url")
    ).strip().lower()
    if media_url_mode in {"file", "file_url", "file://"}:
        return f"file://{Path(path).resolve()}"

    guessed_type = mime_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{guessed_type};base64,{data}"


def _extract_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text", item.get("content"))
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    return str(content)


def _extract_usage(response: Any) -> Dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {
            key: getattr(usage, key)
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            )
            if getattr(usage, key, None) is not None
        }
    return {key: value for key, value in data.items() if value is not None}


def _extract_stream_delta(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""
    return _extract_text_content(getattr(delta, "content", None))


def _extract_stream_reasoning(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return ""

    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning is not None:
        return _extract_text_content(reasoning)

    if hasattr(delta, "model_dump"):
        delta_data = delta.model_dump(exclude_none=True)
    elif isinstance(delta, dict):
        delta_data = delta
    else:
        delta_data = {}
    return _extract_text_content(delta_data.get("reasoning_content"))


def _extract_message_reasoning(message: Any) -> str:
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        return _extract_text_content(reasoning)

    if hasattr(message, "model_dump"):
        message_data = message.model_dump(exclude_none=True)
    elif isinstance(message, dict):
        message_data = message
    else:
        message_data = {}
    return _extract_text_content(message_data.get("reasoning_content"))


def _extract_finish_reason(response: Any) -> str | None:
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    finish_reason = getattr(choices[0], "finish_reason", None)
    return str(finish_reason) if finish_reason is not None else None


def _messages_with_video_first(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Place video items before audio items without moving non-media content."""
    reordered_messages: List[Dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            reordered_messages.append(message)
            continue

        media_positions = [
            index
            for index, item in enumerate(content)
            if isinstance(item, dict)
            and item.get("type") in {"audio_url", "video_url"}
        ]
        media_items = [content[index] for index in media_positions]
        media_items.sort(key=lambda item: item.get("type") != "video_url")
        reordered_content = list(content)
        for index, item in zip(media_positions, media_items):
            reordered_content[index] = item
        reordered_messages.append({**message, "content": reordered_content})
    return reordered_messages


def call_qwen_messages(
    messages: List[Dict[str, Any]],
    *,
    model: str,
    api_key: str,
    base_url: str,
    timeout: int = 180,
    max_retries: int = 3,
    retry_delay_s: float = 1.0,
    temperature: float = 0.6,
    qwen_seed: Optional[int] = None,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 1024,
    fps: float = 2.0,
    max_frames: int = 128,
    use_audio_in_video: Optional[bool] = None,
    repetition_penalty: Optional[float] = None,
    enable_thinking: Optional[bool] = None,
    include_enable_thinking: Optional[bool] = None,
    stream: Optional[bool] = None,
    return_thinking: bool = False,
    retry_degenerate_response: bool = False,
    empty_response_retry_min_tokens: int = 0,
    empty_response_retry_temperature: Optional[float] = None,
    empty_response_retry_seed_step: int = 0,
    empty_response_retry_video_first: bool = False,
    raise_on_empty_response: bool = False,
    stream_empty_fallback: bool = True,
) -> Tuple[str, Dict[str, Any]] | Tuple[str, Dict[str, Any], str]:
    last_err: Exception | None = None
    resolved_stream = _env_flag("QWEN_STREAM", True) if stream is None else stream
    response_attempts: list[dict[str, Any]] = []
    empty_response_retry_count = 0
    for attempt in range(1, max_retries + 1):
        try:
            is_empty_response_retry = (
                retry_degenerate_response and empty_response_retry_count > 0
            )
            effective_temperature = temperature
            effective_seed = qwen_seed
            effective_min_tokens = 0
            if is_empty_response_retry:
                if empty_response_retry_temperature is not None:
                    effective_temperature = empty_response_retry_temperature
                if effective_seed is not None:
                    effective_seed += empty_response_retry_seed_step * empty_response_retry_count
                effective_min_tokens = min(
                    max(0, empty_response_retry_min_tokens),
                    max_tokens,
                )

            use_video_first_retry = (
                is_empty_response_retry and empty_response_retry_video_first
            )
            effective_messages = (
                _messages_with_video_first(messages)
                if use_video_first_retry
                else messages
            )

            extra_body: Dict[str, Any] = {
                "top_k": top_k,
            }
            if effective_min_tokens:
                extra_body["min_tokens"] = effective_min_tokens
            if repetition_penalty is not None:
                extra_body["repetition_penalty"] = repetition_penalty
            # vLLM samples video according to the server-side
            # ``--media-io-kwargs`` and injects the effective sampled FPS into
            # ``videos_kwargs``. Passing ``fps`` here as a second top-level HF
            # processor kwarg makes current vLLM-Omni/Transformers reject the
            # request with "Keyword argument fps was passed two times". Keep
            # the client-side frame ceiling, but let the serving configuration
            # own sampling FPS.
            mm_processor_kwargs: Dict[str, Any] = {
                "max_frames": max_frames,
            }
            if use_audio_in_video is not None:
                mm_processor_kwargs["use_audio_in_video"] = use_audio_in_video
            extra_body["mm_processor_kwargs"] = mm_processor_kwargs
            include_thinking_flag = (
                _env_flag("QWEN_INCLUDE_ENABLE_THINKING", True)
                if include_enable_thinking is None else include_enable_thinking
            )
            if enable_thinking is not None and include_thinking_flag:
                extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = enable_thinking

            request_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": effective_messages,
                "stream": resolved_stream,
                "temperature": effective_temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "extra_body": extra_body,
            }
            if effective_seed is not None:
                request_kwargs["seed"] = effective_seed
            if resolved_stream:
                request_kwargs["stream_options"] = {"include_usage": True}

            completion = _client(api_key=api_key, base_url=base_url).chat.completions.create(
                **request_kwargs
            )

            if resolved_stream:
                content_chunks: list[str] = []
                reasoning_chunks: list[str] = []
                token_usage: Dict[str, Any] = {}
                finish_reason: str | None = None
                for chunk in completion:
                    content = _extract_stream_delta(chunk)
                    if content:
                        content_chunks.append(content)
                    reasoning = _extract_stream_reasoning(chunk)
                    if reasoning:
                        reasoning_chunks.append(reasoning)
                    usage = _extract_usage(chunk)
                    if usage:
                        token_usage = usage
                    finish_reason = _extract_finish_reason(chunk) or finish_reason
                response_text = "".join(content_chunks)
                reasoning_text = "".join(reasoning_chunks).strip()
                if stream_empty_fallback and not response_text.strip() and not reasoning_text:
                    retry_kwargs = dict(request_kwargs)
                    retry_kwargs["stream"] = False
                    retry_kwargs.pop("stream_options", None)
                    retry_completion = _client(api_key=api_key, base_url=base_url).chat.completions.create(
                        **retry_kwargs
                    )
                    message = retry_completion.choices[0].message
                    response_text = _extract_text_content(getattr(message, "content", None))
                    reasoning_text = _extract_message_reasoning(message).strip()
                    token_usage = _extract_usage(retry_completion) or token_usage
                    finish_reason = _extract_finish_reason(retry_completion) or finish_reason
            else:
                message = completion.choices[0].message
                response_text = _extract_text_content(getattr(message, "content", None))
                reasoning_text = _extract_message_reasoning(message).strip()
                token_usage = _extract_usage(completion)
                finish_reason = _extract_finish_reason(completion)

            degenerate_reason = (
                degenerate_response_reason(
                    response_text,
                    token_usage,
                    max_tokens=max_tokens,
                )
                if retry_degenerate_response
                else None
            )
            response_attempts.append(
                {
                    "attempt": attempt,
                    "seed": effective_seed,
                    "temperature": effective_temperature,
                    "max_tokens": max_tokens,
                    "min_tokens": effective_min_tokens,
                    "video_first_retry": use_video_first_retry,
                    "completion_tokens": token_usage.get("completion_tokens"),
                    "content_chars": len(response_text),
                    "reasoning_chars": len(reasoning_text),
                    "finish_reason": finish_reason,
                    "degenerate_reason": degenerate_reason,
                }
            )
            if degenerate_reason and attempt < max_retries:
                print(
                    f"[warn] Qwen API attempt {attempt}/{max_retries} returned a degenerate "
                    f"response: {degenerate_reason}. Retrying in {retry_delay_s}s.",
                    flush=True,
                )
                empty_response_retry_count += 1
                time.sleep(retry_delay_s)
                continue
            if degenerate_reason:
                message = (
                    f"Qwen API exhausted {max_retries} attempts after a degenerate "
                    f"response: {degenerate_reason}"
                )
                print(f"[warn] {message}.", flush=True)
                if raise_on_empty_response:
                    raise EmptyQwenResponseError(message, response_attempts)

            token_usage = dict(token_usage)
            token_usage["qwen_max_tokens"] = max_tokens
            token_usage["qwen_finish_reason"] = finish_reason
            if len(response_attempts) > 1:
                token_usage["qwen_response_attempts"] = response_attempts

            if return_thinking:
                return response_text, token_usage, reasoning_text
            return response_text, token_usage
        except EmptyQwenResponseError:
            raise
        except Exception as exc:
            last_err = exc
            if attempt < max_retries:
                time.sleep(retry_delay_s)

    assert last_err is not None
    print(
        f"[error] Qwen API call failed after {max_retries} attempt(s): "
        f"{safe_exception_summary(last_err)}",
        flush=True,
    )
    raise RuntimeError(f"Qwen API call failed: {last_err}") from last_err
