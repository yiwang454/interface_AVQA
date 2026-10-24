"""Cached-caption compatibility helpers adapted from deepseek_selfimprove."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CachedCaption:
    text: str
    lookup_id: str
    source_format: str


def _load_caption_pair(path: Path) -> tuple[str, str, str]:
    """Support nested audio/video captions and single caption/response records."""
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", "", "unreadable"
    if not isinstance(data, dict):
        return "", "", "invalid"
    audio = data.get("audio") or {}
    video = data.get("video") or {}
    audio_text = audio.get("caption", "") if isinstance(audio, dict) else ""
    video_text = video.get("caption", "") if isinstance(video, dict) else ""
    if audio_text or video_text:
        return str(audio_text), str(video_text), "audio_video"
    caption = data.get("caption") or data.get("response")
    if isinstance(caption, str) and caption.strip():
        return "", caption, "caption_or_response"
    return "", "", "empty"


def cached_caption_lookup_ids(cut: dict[str, Any]) -> list[str]:
    """Try recording_id first, then cut/sample id, matching the legacy baseline."""
    supervision = (cut.get("supervisions") or [{}])[0]
    custom = supervision.get("custom") or {}
    sample_id = str(cut.get("id") or supervision.get("id") or "")
    recording_id = str(supervision.get("recording_id") or custom.get("video_id") or sample_id)
    return list(dict.fromkeys(value for value in (recording_id, sample_id) if value))


def load_cached_caption(cut: dict[str, Any], caption_dir: Path) -> CachedCaption:
    """Load a caption or fail explicitly instead of reasoning over empty context."""
    attempted = []
    for lookup_id in cached_caption_lookup_ids(cut):
        path = caption_dir / f"{lookup_id}.json"
        attempted.append(path.name)
        if not path.is_file():
            continue
        audio, video, source_format = _load_caption_pair(path)
        # QA_PROMPT_TEMPLATE_CAPTION_OMNI consumes ``video_caption`` in the
        # legacy runner. A lone audio caption remains a useful fallback.
        text = video.strip() or audio.strip()
        if text:
            return CachedCaption(text=text, lookup_id=lookup_id, source_format=source_format)
    raise ValueError(
        f"Cached caption missing or empty for cut_id={cut.get('id')}; tried {', '.join(attempted)}"
    )
