"""Prompt YAML loading, dot-path lookup, and optimizer-visible overrides.

Ported from react-agent-avqa-optimize's ``dspy_avqa/prompt_config.py``. This
is the mechanism Stage 2's optimizer wrapper depends on: a candidate
instruction is applied by entering ``prompt_overrides({...})`` around one
``forward()`` call, never by mutating the loaded YAML in place, so concurrent
DSPy worker threads evaluating different candidates never race on shared
state.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import yaml


DEFAULT_PROMPT_YAML = Path(__file__).resolve().parent / "yamls" / "daily_qa_single_mllm.yaml"

_PROMPT_YAML_PATH: Path | None = None
_PROMPTS: dict[str, Any] = {}
_PROMPT_OVERRIDES: ContextVar[dict[tuple[str, ...], str]] = ContextVar(
    "AVQA_INSTR_OPT_PROMPT_OVERRIDES", default={}
)


def load_prompt_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load the nested prompt mapping directly, as in the sibling repository."""
    global _PROMPT_YAML_PATH, _PROMPTS
    yaml_path = Path(path) if path is not None else DEFAULT_PROMPT_YAML
    with yaml_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Prompt config must be a YAML mapping: {yaml_path}")
    _PROMPT_YAML_PATH = yaml_path
    _PROMPTS = config
    return config


def load_named_prompt(path: Path, template_name: str, system_template_name: str | None = None) -> tuple[str, str]:
    """Load one legacy flat named template without making it the active mapping."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Prompt config must be a YAML mapping: {path}")
    prompt = config.get(template_name)
    if not isinstance(prompt, str):
        raise ValueError(f"Prompt template not found: {template_name} ({path})")
    system = config.get(system_template_name) if system_template_name else config.get("system_prompt", "")
    if system_template_name and not isinstance(system, str):
        raise ValueError(f"System prompt template not found: {system_template_name} ({path})")
    return prompt, system.rstrip("\n") if isinstance(system, str) else ""


def split_text_reasoner_prompt(prompt: str) -> tuple[str, str]:
    """Split the legacy caption-QA prompt into an optimizable instruction and fixed shell.

    The split is deliberately anchored at ``Audio-visual caption:`` so rendering
    the two pieces reproduces the existing baseline prompt exactly while keeping
    all runtime placeholders outside the optimizer-controlled text.
    """
    marker = "Audio-visual caption:"
    if marker not in prompt:
        raise ValueError(
            "Text reasoner prompt must contain the fixed 'Audio-visual caption:' marker"
        )
    instruction, remainder = prompt.split(marker, 1)
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("Text reasoner user instruction is empty")
    return instruction, "{user_instruction}\n" + marker + remainder


def load_caption_reasoner_prompts(
    *,
    caption_prompt_yaml: Path,
    caption_template_name: str,
    reasoner_prompt_yaml: Path,
    reasoner_template_name: str,
    caption_system_template_name: str | None = None,
    reasoner_system_template_name: str | None = None,
) -> dict[str, Any]:
    """Activate flat baseline templates through the internal dot-path contract."""
    global _PROMPT_YAML_PATH, _PROMPTS
    caption, caption_system = load_named_prompt(
        caption_prompt_yaml, caption_template_name, caption_system_template_name
    )
    reasoner, reasoner_system = load_named_prompt(
        reasoner_prompt_yaml, reasoner_template_name, reasoner_system_template_name
    )
    reasoner_instruction, reasoner_template = split_text_reasoner_prompt(reasoner)
    _PROMPT_YAML_PATH = reasoner_prompt_yaml
    _PROMPTS = {
        "captioner": {
            "system_prompt": caption_system,
            "user_instruction": caption,
            "caption_prompt_template": "{user_instruction}",
        },
        "text_reasoner": {
            "system_prompt": reasoner_system,
            "user_instruction": reasoner_instruction,
            "prompt_template": reasoner_template,
        },
    }
    return _PROMPTS


def load_text_reasoner_prompt(
    *, reasoner_prompt_yaml: Path, reasoner_template_name: str,
    reasoner_system_template_name: str | None = None,
) -> dict[str, Any]:
    """Activate only a cached-caption reasoner template."""
    global _PROMPT_YAML_PATH, _PROMPTS
    reasoner, system = load_named_prompt(
        reasoner_prompt_yaml, reasoner_template_name, reasoner_system_template_name
    )
    reasoner_instruction, reasoner_template = split_text_reasoner_prompt(reasoner)
    _PROMPT_YAML_PATH = reasoner_prompt_yaml
    _PROMPTS = {"text_reasoner": {
        "system_prompt": system,
        "user_instruction": reasoner_instruction,
        "prompt_template": reasoner_template,
    }}
    return _PROMPTS


def active_prompt_yaml_path() -> Path:
    """Return the active prompt YAML path, loading the default if needed."""
    if _PROMPT_YAML_PATH is None:
        load_prompt_config()
    assert _PROMPT_YAML_PATH is not None
    return _PROMPT_YAML_PATH


def prompt_config() -> dict[str, Any]:
    """Return the active prompt mapping, loading the default if needed."""
    if not _PROMPTS:
        load_prompt_config()
    return _PROMPTS


@contextmanager
def prompt_overrides(overrides: dict[tuple[str, ...], str]):
    """Temporarily override selected prompt values for one program rollout."""
    current = dict(_PROMPT_OVERRIDES.get())
    current.update(overrides)
    token = _PROMPT_OVERRIDES.set(current)
    try:
        yield
    finally:
        _PROMPT_OVERRIDES.reset(token)


def prompt_value(*keys: str) -> str:
    """Return a string prompt value by nested YAML keys."""
    override_key = tuple(keys)
    overrides = _PROMPT_OVERRIDES.get()
    if override_key in overrides:
        return overrides[override_key]

    value: Any = prompt_config()
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            joined = ".".join(keys)
            raise KeyError(f"Missing prompt config key: {joined}")
        value = value[key]
    if not isinstance(value, str):
        joined = ".".join(keys)
        raise ValueError(f"Prompt config value must be a string: {joined}")
    return value


def render_prompt(*keys: str, **kwargs: Any) -> str:
    """Render a configured prompt template with Python format variables."""
    return prompt_value(*keys).format(**kwargs).strip()
