"""YAML-loaded task-model profiles and resolved-config artifacts.

Qwen YAML values are materialized under role-prefixed runtime variables before
profile resolution.  This is adapted from ``call_qwen_perception``'s prefix
family in react-agent-avqa-optimize's ``dspy_avqa/tools.py`` so direct-Qwen and
captioner-Qwen can use independent profiles.  The low-level transport does not
fall back to provider credentials or a public endpoint; callers pass the fully
resolved YAML profile explicitly.

``redact_sensitive``/``save_resolved_experiment_config`` are ported verbatim
from ``dspy_avqa/experiment_config.py`` -- Stage 3 launcher scripts call
``save_resolved_experiment_config`` once per run so a resolved, secret-free
snapshot of every client/profile setting sits next to the results, matching
the reproducibility convention already used across the source repo's
experiment records.
"""

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_SENSITIVE_KEY_PARTS = ("api_key", "password", "secret", "credential")


def load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}.")
    return data


def first_present(*values: Any, default: Any = None) -> Any:
    return next((value for value in values if value is not None), default)


def _env_bool(value: Any) -> str:
    if isinstance(value, str):
        return "false" if value.strip().lower() in {"0", "false", "no", "off"} else "true"
    return "true" if bool(value) else "false"


def _load_qwen_config_yaml(path: Path | None, *, prefix: str) -> dict[str, Any]:
    """Qwen-only subset of sibling runner._load_qwen_config_yaml.

    Explicit YAML values are applied to the role environment before profile
    resolution; omitted values retain environment/default settings.
    """
    if path is None:
        return {}
    config = load_mapping(path)
    sections = [config.get(key) or {} for key in ("task", "model", "sampling_params")]
    if not all(isinstance(section, dict) for section in sections):
        raise ValueError("Qwen config task/model/sampling_params must be mappings")
    task, model, sampling = sections
    for key in ("video_only", "enable_thinking", "video_first", "use_audio_in_video"):
        if task.get(key) is not None:
            os.environ[f"{prefix}_{key.upper()}"] = _env_bool(task[key])
    mappings = (
        (task, {"timeout": "TIMEOUT", "max_retries": "MAX_RETRIES", "qwen_delay_s": "DELAY_S",
                "batch_size": "BATCH_SIZE"}),
        (model, {"qwen_model": "MODEL", "qwen_base_url": "BASE_URL", "qwen_api_key": "API_KEY"}),
        (sampling, {key: key.upper() for key in (
            "temperature", "top_p", "top_k", "max_tokens", "fps", "max_frames",
            "seed", "repetition_penalty")}),
    )
    for section, mapping in mappings:
        for key, suffix in mapping.items():
            if section.get(key) is not None:
                os.environ[f"{prefix}_{suffix}"] = str(section[key])
    if (task.get("reliable_max_batch_retries") is not None
            and not os.environ.get("QWEN_RELIABLE_MAX_BATCH_RETRIES", "").strip()):
        os.environ["QWEN_RELIABLE_MAX_BATCH_RETRIES"] = str(task["reliable_max_batch_retries"])
    # Transport settings belong to the selected Qwen YAML.  Store them under
    # the requested profile prefix so direct and captioner configs remain
    # independent; resolve_qwen_profile falls back to shared QWEN_* values.
    transport = config.get("transport") or {}
    if not isinstance(transport, dict):
        raise ValueError("Qwen config transport must be a mapping")
    for key, value in transport.items():
        if key not in {"media_url_mode", "stream", "include_enable_thinking"}:
            raise ValueError(f"Unknown transport setting: {key}")
        os.environ[f"{prefix}_{key.upper()}"] = _env_bool(value) if key != "media_url_mode" else str(value)
    return config


def load_perception_config_yaml(path: Path | None) -> dict[str, Any]:
    """Load a direct-task Qwen or official-Vertex Gemini configuration."""
    if path is None:
        return {}
    config = load_mapping(path)
    model = config.get("model") or {}
    if not isinstance(model, dict):
        raise ValueError("Perception config model section must be a mapping")
    backend = str(model.get("backend") or "").strip().lower()
    if "gemini_model" in model or backend == "gemini":
        os.environ["DIRECT_BACKEND"] = "gemini"
        return _load_gemini_config(config, prefix="DIRECT_GEMINI", label="direct Gemini")
    os.environ["DIRECT_BACKEND"] = "qwen"
    return _load_qwen_config_yaml(path, prefix="QWEN")


def load_captioner_config_yaml(path: Path | None) -> dict[str, Any]:
    """Load either a Qwen or official-Vertex Gemini captioner configuration."""
    if path is None:
        return {}
    config = load_mapping(path)
    model = config.get("model") or {}
    if not isinstance(model, dict):
        raise ValueError("Captioner config model section must be a mapping")
    backend = str(model.get("backend") or "").strip().lower()
    if "gemini_model" in model or backend == "gemini":
        os.environ["CAPTIONER_BACKEND"] = "gemini"
        return _load_gemini_captioner_config(config)
    os.environ["CAPTIONER_BACKEND"] = "qwen"
    return _load_qwen_config_yaml(path, prefix="CAPTIONER_QWEN")


def _load_gemini_captioner_config(config: dict[str, Any]) -> dict[str, Any]:
    """Load the Gemini YAML schema into a role-specific profile."""
    return _load_gemini_config(config, prefix="CAPTIONER_GEMINI", label="Gemini captioner")


def _load_gemini_config(
    config: dict[str, Any], *, prefix: str, label: str
) -> dict[str, Any]:
    """Load the shared official-Vertex Gemini YAML schema under a role prefix."""
    sections = [config.get(key) or {} for key in ("task", "model", "sampling_params")]
    if not all(isinstance(section, dict) for section in sections):
        raise ValueError(f"{label} task/model/sampling_params must be mappings")
    task, model, sampling = sections
    for key in ("video_only", "print_first_prompt", "include_thoughts"):
        if task.get(key) is not None:
            os.environ[f"{prefix}_{key.upper()}"] = _env_bool(task[key])
    mappings = (
        (task, {"timeout": "TIMEOUT", "max_retries": "MAX_RETRIES",
                "retry_delay_s": "RETRY_DELAY_S", "batch_size": "BATCH_SIZE"}),
        (model, {"gemini_model": "MODEL", "gemini_base_url": "BASE_URL",
                 "gemini_api_key": "API_KEY", "gemini_media_mode": "MEDIA_MODE",
                 "gemini_local_data_root": "LOCAL_DATA_ROOT",
                 "gemini_gcs_data_root": "GCS_DATA_ROOT"}),
        (sampling, {"temperature": "TEMPERATURE", "top_p": "TOP_P",
                    "top_k": "TOP_K", "max_tokens": "MAX_TOKENS",
                    "gemini_seed": "SEED", "seed": "SEED",
                    "thinking_budget": "THINKING_BUDGET"}),
    )
    for section, mapping in mappings:
        for key, suffix in mapping.items():
            if section.get(key) is not None:
                os.environ[f"{prefix}_{suffix}"] = str(section[key])
            elif key == "top_k" and key in section:
                # ``top_k: null`` explicitly omits this optional Vertex field
                # instead of inheriting a shared GEMINI_TOP_K value.
                os.environ[f"{prefix}_{suffix}"] = "EMPTY"
    return config


def load_text_reasoner_config_yaml(path: Path | None) -> dict[str, Any]:
    """Load the text-only reasoner config into ``TEXT_REASONER_*`` variables.

    The accepted ``task/model/sampling_params`` layout matches the existing
    caption-reasoner YAML and the sibling reasoner-config convention.
    ``task.caption_dir`` is returned as source configuration; the caption
    runner consumes it only in cached mode.
    """
    if path is None:
        return {}
    config = load_mapping(path)
    sections = [config.get(key) or {} for key in ("task", "model", "sampling_params")]
    if not all(isinstance(section, dict) for section in sections):
        raise ValueError("Text reasoner config task/model/sampling_params must be mappings")
    task, model, sampling = sections
    backend = str(model.get("backend") or "").strip().lower()
    if backend == "gemini" or "gemini_model" in model:
        # Vertex Gemini is a native generateContent endpoint, not an
        # OpenAI-compatible text endpoint. Do not silently route it as GPT.
        os.environ["TEXT_REASONER_BACKEND"] = "gemini"
        return _load_gemini_config(
            config, prefix="TEXT_REASONER_GEMINI", label="Gemini text reasoner"
        )
    os.environ["TEXT_REASONER_BACKEND"] = "openai"
    mappings = (
        (task, {"timeout": "TIMEOUT", "max_retries": "MAX_RETRIES"}),
        (model, {"openai_model": "MODEL", "model_name": "MODEL",
                 "openai_base_url": "API_BASE", "base_url": "API_BASE",
                 "openai_api_key": "API_KEY", "api_key": "API_KEY"}),
        (sampling, {"temperature": "TEMPERATURE", "top_p": "TOP_P",
                    "max_tokens": "MAX_TOKENS", "seed": "SEED",
                    "reasoning_effort": "REASONING_EFFORT"}),
    )
    for section, mapping in mappings:
        for key, suffix in mapping.items():
            if section.get(key) is not None:
                os.environ[f"TEXT_REASONER_{suffix}"] = str(section[key])
    return config


def load_optimizer_lm_config_yaml(path: Path | None, *, role: str) -> dict[str, Any]:
    """Load a sibling-style reasoner YAML for a prompt/reflection LM role.

    ``provider`` is retained as provenance; DSPy routes bare GPT model names
    through its OpenAI-compatible provider. Endpoint and credentials remain
    role-specific environment settings unless explicitly included in YAML.
    """
    if path is None:
        return {}
    config = load_mapping(path)
    allowed_root = {"model", "sampling_params"}
    unknown_root = sorted(set(config) - allowed_root)
    if unknown_root:
        raise ValueError(f"Unknown optimizer LM config key(s): {', '.join(unknown_root)}")
    model = config.get("model") or {}
    sampling = config.get("sampling_params") or {}
    if not isinstance(model, dict) or not isinstance(sampling, dict):
        raise ValueError("Optimizer LM model/sampling_params must be mappings")
    allowed_model = {"provider", "name", "api_base", "api_key", "openai_base_url", "openai_api_key"}
    allowed_sampling = {"temperature", "top_p", "max_tokens", "reasoning_effort", "seed"}
    unknown_model = sorted(set(model) - allowed_model)
    unknown_sampling = sorted(set(sampling) - allowed_sampling)
    if unknown_model or unknown_sampling:
        raise ValueError(
            "Unknown optimizer LM field(s): " + ", ".join(unknown_model + unknown_sampling)
        )
    prefix = role.strip().upper()
    mappings = (
        (model, {"name": "MODEL", "api_base": "API_BASE", "openai_base_url": "API_BASE",
                 "api_key": "API_KEY", "openai_api_key": "API_KEY"}),
        (sampling, {"temperature": "TEMPERATURE", "top_p": "TOP_P",
                    "max_tokens": "MAX_TOKENS", "reasoning_effort": "REASONING_EFFORT",
                    "seed": "SEED"}),
    )
    for section, mapping in mappings:
        for key, suffix in mapping.items():
            if section.get(key) is not None:
                os.environ[f"{prefix}_{suffix}"] = str(section[key])
    return config


def load_inference_config(args) -> tuple[QwenModelProfile | GeminiModelProfile, dict]:
    """Resolve the direct-Qwen or direct-Gemini YAML and common CLI overrides."""
    config = load_perception_config_yaml(args.config)
    backend = os.environ.get("DIRECT_BACKEND", "qwen").strip().lower()
    if backend not in {"qwen", "gemini"}:
        raise ValueError(f"Unsupported DIRECT_BACKEND: {backend!r}")
    overrides = (
        (("openai_model", "MODEL"), ("openai_base_url", "BASE_URL"),
         ("openai_api_key", "API_KEY"), ("temperature", "TEMPERATURE"),
         ("max_tokens", "MAX_TOKENS"), ("qwen_seed", "SEED"),
         ("retry_delay_s", "DELAY_S"))
        if backend == "qwen" else
        (("temperature", "TEMPERATURE"), ("max_tokens", "MAX_TOKENS"),
         ("qwen_seed", "SEED"), ("retry_delay_s", "RETRY_DELAY_S"))
    )
    prefix = "QWEN" if backend == "qwen" else "DIRECT_GEMINI"
    for field, suffix in overrides:
        value = getattr(args, field)
        if value is not None:
            os.environ[prefix + "_" + suffix] = str(value)
    task = config.get("task") or {}
    args.input_jsonl = first_present(args.input_jsonl, task.get("input_jsonl"))
    if args.input_jsonl is None:
        raise ValueError("Missing required config/CLI value: input_jsonl")
    args.input_jsonl = Path(args.input_jsonl)
    args.workers = int(first_present(args.workers,
        os.getenv("DSPY_AVQA_INFERENCE_NUM_THREADS"), default=4))
    args.batch_size = int(first_present(args.batch_size, default=args.workers))
    profile = (resolve_qwen_profile("direct_qwen") if backend == "qwen"
               else resolve_gemini_profile("direct_gemini"))
    return profile, config


def _env_optional_value(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip() or value.strip().upper() == "EMPTY":
        return None
    return value.strip()


def _role_env_name(role: str, suffix: str) -> str:
    return f"{role.strip().upper()}_{suffix}"


def _role_env_value(role: str, suffix: str, default: str) -> str:
    """Return {ROLE}_{SUFFIX}, falling back to QWEN_{SUFFIX}, then default."""
    value = _env_optional_value(_role_env_name(role, suffix))
    if value is not None:
        return value
    if role.strip().upper() != "QWEN":
        fallback = _env_optional_value(f"QWEN_{suffix}")
        if fallback is not None:
            return fallback
    return default


def _role_env_optional_value(role: str, suffix: str) -> str | None:
    value = _env_optional_value(_role_env_name(role, suffix))
    if value is not None:
        return value
    if role.strip().upper() != "QWEN":
        return _env_optional_value(f"QWEN_{suffix}")
    return None


def _role_env_api_key(role: str) -> str | None:
    """Resolve a Qwen API-key field while preserving the local ``EMPTY`` token."""
    names = [_role_env_name(role, "API_KEY")]
    if role.strip().upper() != "QWEN":
        names.append("QWEN_API_KEY")
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _role_env_optional_int(role: str, suffix: str) -> int | None:
    value = _role_env_optional_value(role, suffix)
    return int(value) if value is not None else None


def _role_env_optional_float(role: str, suffix: str) -> float | None:
    value = _role_env_optional_value(role, suffix)
    return float(value) if value is not None else None


def _role_env_flag(role: str, suffix: str, default: str = "false") -> bool:
    return _role_env_value(role, suffix, default).strip().lower() in {"1", "true", "yes", "on"}


def _gemini_env_optional_value(suffix: str, *, role: str = "captioner_gemini") -> str | None:
    role_name = f"{role.strip().upper()}_{suffix}"
    if role_name in os.environ:
        return _env_optional_value(role_name)
    return _env_optional_value(f"GEMINI_{suffix}")


def _gemini_env_value(suffix: str, default: str, *, role: str = "captioner_gemini") -> str:
    return _gemini_env_optional_value(suffix, role=role) or default


@dataclass(frozen=True)
class QwenModelProfile:
    """Resolved Qwen3-Omni model/server/sampling settings for one role."""

    role: str
    model: str
    base_url: str | None
    api_key: str | None
    timeout: int
    max_retries: int
    retry_delay_s: float
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    fps: float
    max_frames: int
    seed: int | None
    repetition_penalty: float | None
    enable_thinking: bool | None
    video_only: bool
    video_first: bool
    use_audio_in_video: bool
    media_url_mode: str
    stream: bool
    include_enable_thinking: bool

    def redacted(self) -> dict[str, Any]:
        """Return this profile as a dict with the API key masked."""
        return redact_sensitive(vars(self))


@dataclass(frozen=True)
class GeminiModelProfile:
    """Resolved official Vertex Gemini settings."""

    role: str
    model: str
    base_url: str | None
    api_key: str | None
    timeout: int
    max_retries: int
    retry_delay_s: float
    temperature: float
    top_p: float
    top_k: int | None
    max_tokens: int
    seed: int | None
    video_only: bool
    include_thoughts: bool
    print_first_prompt: bool
    thinking_budget: int | None = None
    media_mode: str = "inline"
    local_data_root: str | None = None
    gcs_data_root: str | None = None

    def redacted(self) -> dict[str, Any]:
        return redact_sensitive(vars(self))


CaptionerModelProfile = QwenModelProfile | GeminiModelProfile


def resolve_gemini_profile(
    role: str = "captioner_gemini", *, default_top_k: int | None = None
) -> GeminiModelProfile:
    """Resolve an official Vertex Gemini profile with role-scoped overrides."""
    media_mode = _gemini_env_value("MEDIA_MODE", "inline", role=role).strip().lower()
    if media_mode not in {"inline", "gcs"}:
        raise ValueError(
            f"Unsupported Gemini media mode {media_mode!r}; expected inline or gcs."
        )
    seed = _gemini_env_optional_value("SEED", role=role)
    thinking_budget = _gemini_env_optional_value("THINKING_BUDGET", role=role)
    scoped_top_k = os.environ.get(f"{role.strip().upper()}_TOP_K")
    if scoped_top_k is not None and (not scoped_top_k.strip() or scoped_top_k.strip().upper() == "EMPTY"):
        top_k = default_top_k
    else:
        raw_top_k = _gemini_env_optional_value("TOP_K", role=role)
        top_k = int(raw_top_k) if raw_top_k is not None else default_top_k
    return GeminiModelProfile(
        role=role,
        model=_gemini_env_value("MODEL", "gemini-2.5-flash", role=role),
        base_url=_gemini_env_optional_value("BASE_URL", role=role),
        api_key=_gemini_env_optional_value("API_KEY", role=role),
        timeout=int(_gemini_env_value("TIMEOUT", "180", role=role)),
        max_retries=int(_gemini_env_value("MAX_RETRIES", "3", role=role)),
        retry_delay_s=float(_gemini_env_value("RETRY_DELAY_S", "5", role=role)),
        temperature=float(_gemini_env_value("TEMPERATURE", "0.6", role=role)),
        top_p=float(_gemini_env_value("TOP_P", "0.95", role=role)),
        top_k=top_k,
        max_tokens=int(_gemini_env_value("MAX_TOKENS", "1024", role=role)),
        seed=int(seed) if seed is not None else None,
        video_only=_gemini_env_value("VIDEO_ONLY", "true", role=role).strip().lower()
        not in {"0", "false", "no", "off"},
        include_thoughts=_gemini_env_value("INCLUDE_THOUGHTS", "true", role=role).strip().lower()
        in {"1", "true", "yes", "on"},
        print_first_prompt=_gemini_env_value("PRINT_FIRST_PROMPT", "false", role=role).strip().lower()
        in {"1", "true", "yes", "on"},
        thinking_budget=(int(thinking_budget) if thinking_budget is not None else None),
        media_mode=media_mode,
        local_data_root=_gemini_env_optional_value("LOCAL_DATA_ROOT", role=role),
        gcs_data_root=_gemini_env_optional_value("GCS_DATA_ROOT", role=role),
    )


def resolve_captioner_profile() -> CaptionerModelProfile:
    backend = os.environ.get("CAPTIONER_BACKEND", "qwen").strip().lower()
    if backend == "gemini":
        return resolve_gemini_profile()
    if backend != "qwen":
        raise ValueError(f"Unsupported CAPTIONER_BACKEND: {backend!r}")
    return resolve_qwen_profile("captioner_qwen")


def captioner_backend(profile: CaptionerModelProfile) -> str:
    return "gemini" if isinstance(profile, GeminiModelProfile) else "qwen"


def resolve_qwen_profile(role: str) -> QwenModelProfile:
    """Resolve a Qwen model/server profile for ``role`` (e.g. ``direct_qwen``).

    YAML loading has already populated ``{ROLE}_{SUFFIX}`` or the shared
    ``QWEN_{SUFFIX}``; the prefix fallback lets direct and captioner profiles
    share the same resolver without introducing a transport-level fallback.
    """
    video_only = _role_env_flag(role, "VIDEO_ONLY", "false")
    use_audio_in_video = (not video_only) and _role_env_flag(role, "USE_AUDIO_IN_VIDEO", "false")
    return QwenModelProfile(
        role=role,
        model=_role_env_value(role, "MODEL", "qwen3-omni-flash"),
        base_url=_role_env_optional_value(role, "BASE_URL"),
        api_key=_role_env_api_key(role),
        timeout=int(_role_env_value(role, "TIMEOUT", "180")),
        max_retries=int(_role_env_value(role, "MAX_RETRIES", "3")),
        retry_delay_s=float(_role_env_value(role, "DELAY_S", "1.0")),
        temperature=float(_role_env_value(role, "TEMPERATURE", "0.6")),
        top_p=float(_role_env_value(role, "TOP_P", "0.95")),
        top_k=int(_role_env_value(role, "TOP_K", "20")),
        max_tokens=int(_role_env_value(role, "MAX_TOKENS", "1024")),
        fps=float(_role_env_value(role, "FPS", "2.0")),
        max_frames=int(_role_env_value(role, "MAX_FRAMES", "128")),
        seed=_role_env_optional_int(role, "SEED"),
        repetition_penalty=_role_env_optional_float(role, "REPETITION_PENALTY"),
        enable_thinking=(
            None
            if _role_env_optional_value(role, "ENABLE_THINKING") is None
            else _role_env_flag(role, "ENABLE_THINKING", "false")
        ),
        video_only=video_only,
        video_first=_role_env_flag(role, "VIDEO_FIRST", "false"),
        use_audio_in_video=use_audio_in_video,
        media_url_mode=_role_env_value(role, "MEDIA_URL_MODE", "data_url").strip().lower(),
        stream=_role_env_flag(role, "STREAM", "true"),
        include_enable_thinking=_role_env_flag(role, "INCLUDE_ENABLE_THINKING", "true"),
    )


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return {key: _serializable(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def redact_sensitive(value: Any) -> Any:
    """Recursively replace secret values while retaining whether they were set."""
    value = _serializable(value)
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = key.lower()
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS) or normalized == "token" or normalized.endswith("_token"):
                if item in {"<unset>", "<redacted:unset>"}:
                    redacted[key] = "<redacted:unset>"
                elif item in {"<set>", "<redacted:set>"}:
                    redacted[key] = "<redacted:set>"
                else:
                    redacted[key] = "<redacted:set>" if item else "<redacted:unset>"
            else:
                redacted[key] = redact_sensitive(item)
        return redacted
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def save_resolved_experiment_config(
    output_dir: Path,
    payload: dict[str, Any],
    *,
    print_config: bool = False,
) -> Path:
    """Save (and optionally print) a redacted resolved experiment mapping."""
    resolved = redact_sensitive(payload)
    rendered = yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "resolved_experiment_config.yaml"
    output_path.write_text(rendered, encoding="utf-8")
    if print_config:
        print("Resolved experiment config:")
        print(rendered, end="")
    return output_path
