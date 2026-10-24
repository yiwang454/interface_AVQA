"""Factory for text-only LMs used by the caption-to-reasoner workflow.

Used for two roles that never touch video/audio:

- ``text_reasoner``: the fixed (non-optimized) reader in
  ``programs.caption_then_text.CaptionThenTextQAProgram`` that answers from a
  caption plus the question/options.
- ``prompt_lm``: the GPT model Stage 2 optimizers (COPRO/MIPROv2) use to
  propose new instruction candidates -- DSPy's ``prompt_model``/reflection LM
  role. Kept here, not in Stage 2, because it is the same kind of client as
  ``text_reasoner`` and Stage 2 only needs to import ``build_text_lm``.

Model-name normalization (bare ``gpt-4.1`` -> ``openai/gpt-4.1``) mirrors
``optimize.py``'s ``_normalize_openai_model_name``/``build_gepa_reflection_lm``
in the source repo, so the same ``TEXT_REASONER_MODEL=gpt-4.1``-style env
values behave the same way here.
"""

from __future__ import annotations

import os
from typing import Any

import dspy

from ..clients.gemini import call_gemini_messages
from ..config import GeminiModelProfile, resolve_gemini_profile


def _normalize_model_name(model: str) -> str:
    return model if "/" in model else f"openai/{model}"


def _supports_reasoning_effort(model: str) -> bool:
    """Return whether the routed model accepts LiteLLM's reasoning parameter.

    GPT-4.1 is not an OpenAI reasoning-model family.  LiteLLM rejects an
    explicitly supplied ``reasoning_effort`` for it, including the sibling
    config's ``reasoning_effort: none`` value.  Keep the parameter available
    for models such as GPT-5 while omitting it for GPT-4.1 and dated/size
    variants of that family.
    """
    provider, separator, model_name = model.lower().partition("/")
    return not (
        separator
        and provider == "openai"
        and (model_name == "gpt-4.1" or model_name.startswith("gpt-4.1-"))
    )


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip() or value.strip().upper() == "EMPTY":
        return None
    return value.strip()


def resolve_text_lm_config(role: str, *, default_model: str) -> dict[str, Any]:
    """Resolve a text LM's model/kwargs from ``{ROLE}_*`` environment variables."""
    prefix = role.strip().upper()
    if _optional_env(f"{prefix}_BACKEND") == "gemini":
        profile = resolve_gemini_profile(f"{role}_gemini", default_top_k=None)
        return {
            "backend": "gemini",
            "model": profile.model,
            "api_key": profile.api_key,
            "api_base": profile.base_url,
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "top_k": profile.top_k,
            "max_tokens": profile.max_tokens,
            "seed": profile.seed,
            "timeout": float(profile.timeout),
            "num_retries": profile.max_retries,
            "include_thoughts": profile.include_thoughts,
        }
    model = _optional_env(f"{prefix}_MODEL") or default_model
    config: dict[str, Any] = {"model": _normalize_model_name(model)}

    api_key = _optional_env(f"{prefix}_API_KEY")
    if api_key is not None:
        config["api_key"] = api_key
    api_base = _optional_env(f"{prefix}_API_BASE")
    if api_base is not None:
        config["api_base"] = api_base

    temperature = _optional_env(f"{prefix}_TEMPERATURE")
    if temperature is not None:
        config["temperature"] = float(temperature)
    max_tokens = _optional_env(f"{prefix}_MAX_TOKENS")
    if max_tokens is not None:
        config["max_tokens"] = int(max_tokens)
    top_p = _optional_env(f"{prefix}_TOP_P")
    if top_p is not None:
        config["top_p"] = float(top_p)
    reasoning_effort = _optional_env(f"{prefix}_REASONING_EFFORT")
    if reasoning_effort is not None and _supports_reasoning_effort(config["model"]):
        config["reasoning_effort"] = reasoning_effort
    seed = _optional_env(f"{prefix}_SEED")
    if seed is not None:
        config["seed"] = int(seed)
    timeout = _optional_env(f"{prefix}_TIMEOUT")
    if timeout is not None:
        config["timeout"] = float(timeout)
    max_retries = _optional_env(f"{prefix}_MAX_RETRIES")
    if max_retries is not None:
        config["num_retries"] = int(max_retries)

    return config


class GeminiTextReasonerLM:
    """Small text-only adapter for Gemini's native ``generateContent`` API."""

    def __init__(self, profile: GeminiModelProfile) -> None:
        self.profile = profile
        self.model = profile.model

    def __call__(self, *, messages: list[dict[str, Any]], **_: Any) -> list[str]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role") or "user").strip().lower()
            content = str(message.get("content") or "")
            if role == "system":
                system_parts.append(content)
            else:
                contents.append({
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": content}],
                })
        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]
        response_text, _usage, _thinking = call_gemini_messages(
            contents,
            system_prompt="\n\n".join(part for part in system_parts if part),
            model=self.profile.model,
            api_key=self.profile.api_key or "",
            base_url=self.profile.base_url or "",
            timeout=self.profile.timeout,
            max_retries=self.profile.max_retries,
            retry_delay_s=self.profile.retry_delay_s,
            include_thoughts=self.profile.include_thoughts,
            thinking_budget=self.profile.thinking_budget,
            return_thinking=True,
            temperature=self.profile.temperature,
            gemini_seed=self.profile.seed,
            top_p=self.profile.top_p,
            top_k=self.profile.top_k,
            max_tokens=self.profile.max_tokens,
            retry_degenerate_response=False,
            print_first_prompt=self.profile.print_first_prompt,
        )
        return [response_text]


def build_text_lm(role: str, *, default_model: str = "openai/gpt-4.1") -> Any:
    """Build the configured OpenAI-compatible or native-Gemini text LM."""
    if _optional_env(f"{role.strip().upper()}_BACKEND") == "gemini":
        return GeminiTextReasonerLM(
            resolve_gemini_profile(f"{role}_gemini", default_top_k=None)
        )
    config = resolve_text_lm_config(role, default_model=default_model)
    return dspy.LM(**config)
