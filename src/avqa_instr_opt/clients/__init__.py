"""MLLM/LLM client components: Qwen3-Omni transport and text-only LMs."""

from __future__ import annotations

from .qwen_omni import EmptyQwenResponseError, call_qwen_messages, to_data_url
from .reliable import (
    PRIMARY_QWEN_PROFILE,
    RETRY_EMBEDDED_AUDIO_QWEN_PROFILE,
    RETRY_VIDEO_FIRST_QWEN_PROFILE,
    QwenRequestProfile,
    ReliableQwenBatchResult,
    ReliableQwenExecutor,
    active_qwen_request_profile,
    qwen_request_profile,
    qwen_request_slot,
    response_is_empty,
)
from .response_quality import degenerate_response_reason
from .text_lm import build_text_lm, resolve_text_lm_config

__all__ = [
    "EmptyQwenResponseError",
    "call_qwen_messages",
    "to_data_url",
    "QwenRequestProfile",
    "PRIMARY_QWEN_PROFILE",
    "RETRY_VIDEO_FIRST_QWEN_PROFILE",
    "RETRY_EMBEDDED_AUDIO_QWEN_PROFILE",
    "ReliableQwenBatchResult",
    "ReliableQwenExecutor",
    "active_qwen_request_profile",
    "qwen_request_profile",
    "qwen_request_slot",
    "response_is_empty",
    "degenerate_response_reason",
    "build_text_lm",
    "resolve_text_lm_config",
]
