"""Batch data loading and output assembly.

Ported (trimmed) from react-agent-avqa-optimize's ``dspy_avqa/data.py``. This
is the Stage 1 -> Stage 3 output contract: any launcher script builds rows
with ``build_result_row`` and writes them with ``write_results_jsonl``, which
keeps ``question_data``/``metadata`` shaped the way the existing
``visualize_dspy_output.py``-style tooling in react-agent-avqa-optimize
expects, so results stay comparable across repos.

Dropped relative to the source module: ``DEFAULT_DEBUG_ID_LIST``/``debug_filter``
(hardcoded to that repo's specific debug sample IDs) and
``load_audio_caption_concat`` (reads a directory of *pre-generated* captions;
this repo's ``programs.caption_then_text`` calls the captioner itself instead
of consuming an external cache -- Stage 2 can add that back if a cached-caption
experiment condition is needed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_input_state(
    cut: dict[str, Any], *, require_audio: bool = True
) -> dict[str, Any]:
    """Map one Lhotse-style cut row to AVQA model inputs."""
    supervisions = cut.get("supervisions") or []
    if not supervisions:
        raise ValueError("Cut has no supervisions")

    supervision = supervisions[0]
    custom = supervision.get("custom") or {}
    recording = cut.get("recording") or {}
    sources = recording.get("sources") or []

    source_video = None
    if sources and isinstance(sources[0], dict):
        source_video = sources[0].get("source")

    question = supervision.get("text") or custom.get("question") or ""
    options = custom.get("options") or custom.get("Choice") or []
    video_path = custom.get("video_path") or source_video
    audio_path = custom.get("audio_path")

    question_id = str(cut.get("id") or "")
    if not question_id:
        raise ValueError("Cut id is empty")

    if not question or not options or not video_path:
        raise ValueError(
            f"Missing required AVQA fields in cut_id={cut.get('id')}: "
            f"question={bool(question)}, options={bool(options)}, video_path={bool(video_path)}"
        )
    if require_audio and not audio_path:
        raise ValueError(f"Missing required audio_path in cut_id={cut.get('id')}")

    return {
        "question_id": question_id,
        "question": question,
        "options": options,
        "video_path": video_path,
        "audio_path": audio_path,
        "video_id": custom.get("video_id"),
    }


def json_safe(value: Any) -> Any:
    """Convert nested SDK/model objects into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return json_safe(method())
            except Exception:
                pass

    return str(value)


def build_result_row(cut: dict[str, Any], response_text: str) -> dict[str, Any]:
    """Build a process_cut_task-style output row."""
    supervisions = cut.get("supervisions") or []
    supervision = supervisions[0]
    custom = supervision.get("custom") or {}
    cut_id = cut.get("id")

    question_data = {
        "question_id": supervision.get("id"),
        "task_type": custom.get("task_type"),
        "question": supervision.get("text"),
        "options": custom.get("options"),
        "answer": custom.get("answer"),
        "response": response_text,
        "fps": None,
    }

    metadata = {
        "video_id": cut_id,
        "duration": custom.get("duration"),
        "domain": custom.get("domain"),
        "sub_category": custom.get("sub_category"),
    }

    return {
        "video_id": cut_id,
        "metadata": metadata,
        "question_data": question_data,
    }


def maybe_dump_question_data(output_dir: Path | None, row: dict[str, Any]) -> None:
    """Optionally save per-cut question_data JSON."""
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{row['video_id']}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(row["question_data"]), f, ensure_ascii=False, indent=2, default=str)


def write_results_jsonl(rows: list[dict[str, Any]], output_jsonl: Path) -> None:
    """Write rows as JSONL."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False, default=str) + "\n")
