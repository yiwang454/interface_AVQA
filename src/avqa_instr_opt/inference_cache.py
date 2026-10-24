"""Per-cut question JSON and directory locking adapted from sibling runner/optimize."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .data import build_result_row, json_safe
from .metrics import normalize_option_letter


def atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_aggregate_outputs(
    examples: list[dict], rows_by_id: dict[str, dict], output_dir: Path, output_jsonl: Path
) -> list[dict]:
    """Atomically rebuild ordered JSONL and the retained grouped JSON artifact."""
    rows = [rows_by_id[e["inputs"]["question_id"]] for e in examples
            if e["inputs"]["question_id"] in rows_by_id]
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + output_jsonl.name + ".", dir=output_jsonl.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_jsonl)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    grouped: dict[str, dict] = {}
    for row in rows:
        grouped.setdefault(str(row["video_id"]), {
            "video_id": row["video_id"], "metadata": row["metadata"], "questions": []
        })["questions"].append(row["question_data"])
    aggregate = sorted(grouped.values(), key=lambda item: str(item["video_id"]))
    for item in aggregate:
        item["questions"].sort(key=lambda q: str(q.get("question_id", "")))
    atomic_write_json(output_dir / "output_test.json", aggregate)
    return rows


def question_cache_path(output_dir: Path, cut: dict) -> Path:
    sample_id = str(cut["id"])
    if (not sample_id or sample_id in {".", ".."} or "/" in sample_id or "\\" in sample_id
            or sample_id in {"run_metadata", "inference_only_metadata", "output_test"}):
        raise ValueError(f"Unsafe or reserved sample ID: {sample_id!r}")
    return output_dir / f"{sample_id}.json"


@contextmanager
def final_eval_directory_lock(output_dir: Path):
    """Same advisory single-writer lock as sibling optimize._final_eval_directory_lock."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".final_eval.lock").open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Final test inference is already running in {output_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def save_question_data(output_dir: Path, cut: dict, row: dict) -> None:
    # Standalone files contain question_data, exactly as sibling maybe_dump_question_data.
    atomic_write_json(question_cache_path(output_dir, cut), row["question_data"])


def load_cached_row_from_question_json(output_dir: Path, cut: dict) -> dict | None:
    path = question_cache_path(output_dir, cut)
    if not path.exists():
        return None
    try:
        question_data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if not isinstance(question_data, dict):
        return None
    response = str(question_data.get("response") or "")
    answer = normalize_option_letter(response)
    expected = build_result_row(cut, "")["question_data"]
    options = expected.get("options") or []
    status = question_data.get("status", "ok")
    if (not response.strip() or response.lstrip().startswith("[ERROR]")
            or status not in {"ok", "invalid_answer"}
            or (status == "ok" and (not answer or answer not in "ABCDEF"[:len(options)]))):
        return None
    if any(question_data.get(key) != expected[key] for key in
           ("question_id", "question", "options", "answer")):
        return None
    row = build_result_row(cut, response)
    row["question_data"] = question_data
    row.update(
        question_id=str(cut["id"]), response_text=question_data.get("response_text", response),
        answer=answer, gold=normalize_option_letter(expected["answer"]), status=status,
        correct=status == "ok" and answer == normalize_option_letter(expected["answer"]),
        thinking=question_data.get("think", ""), token_usage=question_data.get("token_usage", {}),
        request_profile=question_data.get("request_profile"),
    )
    trace = question_data.get("turn_trace")
    if isinstance(trace, list):
        row["turn_trace"] = trace
        row["caption_mode"] = question_data.get("caption_mode")
        if trace:
            row["caption"] = trace[0].get("tool_observation", "")
            row["caption_token_usage"] = trace[0].get("perception_token_usage") or {}
        if len(trace) > 1:
            row["reasoner_token_usage"] = trace[1].get("reasoner_token_usage") or {}
    return row
