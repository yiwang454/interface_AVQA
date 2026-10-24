"""Fixed caption -> one-round text-reasoner Daily-Omni baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from .captions import CachedCaption, load_cached_caption
from .clients.reliable import ReliableQwenExecutor, active_qwen_request_profile, response_is_empty
from .clients.text_lm import build_text_lm, resolve_text_lm_config
from .config import (GeminiModelProfile, captioner_backend, load_captioner_config_yaml,
                     load_text_reasoner_config_yaml, redact_sensitive,
                     resolve_captioner_profile, save_resolved_experiment_config)
from .data import build_result_row
from .inference import prepare_examples, sha256, write_json
from .inference_cache import (final_eval_directory_lock, load_cached_row_from_question_json,
                              save_question_data, write_aggregate_outputs)
from .metrics import avqa_metric, normalize_option_letter
from .programs.caption_then_text import CaptionThenTextQAProgram
from .prompt_config import load_caption_reasoner_prompts, load_text_reasoner_prompt
from .timing import run_with_completion_timer


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return Path(value).expanduser().resolve()


def _cached_caption_row_valid(row: dict | None, caption_mode: str) -> bool:
    if row is None:
        return False
    qd = row.get("question_data") or {}
    trace = qd.get("turn_trace") or []
    status = qd.get("status", "ok")
    completed_response = bool(str(qd.get("response_text", qd.get("response", "")) or "").strip())
    return bool(
        qd.get("caption_mode") == caption_mode
        and status in {"ok", "invalid_answer"}
        and completed_response
        and len(trace) == 2
        and trace[0].get("planner_action") == "tool"
        and trace[0].get("tool_name") == "ask_caption"
        and str(trace[0].get("tool_observation", "") or "").strip()
        and trace[1].get("planner_action") == "final"
        and (
            status == "invalid_answer"
            or normalize_option_letter(trace[1].get("final_answer", ""))
        )
    )


def _turn_trace(
    *, pred, inputs: dict, caption_mode: str, caption_source: CachedCaption | None,
    caption_profile: dict | None, caption_prompt_yaml: Path | None,
    caption_template_name: str | None,
) -> list[dict]:
    caption = str(getattr(pred, "caption", "") or "")
    response = str(getattr(pred, "response_text", "") or "")
    answer = normalize_option_letter(response)
    live_backend = str((caption_profile or {}).get("backend") or "qwen")
    request_profile = active_qwen_request_profile() if live_backend == "qwen" else None
    first = {
        "turn_id": 1, "planner_action": "tool", "tool_name": "ask_caption",
        "raw_tool_name": "ask_caption",
        "perception_backend": live_backend if caption_mode == "live" else "cached",
        "tool_args": {"video_path": inputs["video_path"], "audio_path": inputs["audio_path"]},
        "tool_observation": caption,
        "perception_system_prompt": None, "perception_prompt": None,
        "perception_model": caption_profile.get("model") if caption_profile else None,
        "perception_thinking": str(getattr(pred, "caption_thinking", "") or ""),
        "perception_token_usage": getattr(pred, "caption_token_usage", {}) or {},
        "caption_cache_hit": caption_mode == "cached", "caption_cache_path": None,
        "caption_cache_lookup_id": caption_source.lookup_id if caption_source else None,
        "caption_cache_source_format": caption_source.source_format if caption_source else None,
        "caption_prompt_yaml": str(caption_prompt_yaml) if caption_prompt_yaml else None,
        "caption_prompt_template_name": caption_template_name,
        "reliable_qwen_profile": asdict(request_profile) if request_profile else None,
        "qwen_response_empty": not caption.strip() if live_backend == "qwen" else None,
        "caption_response_empty": not caption.strip(),
        "planner_raw": None, "planner_calls": {}, "planner_parse_error": None,
        "final_answer": None, "video_id": inputs.get("video_id"),
    }
    second = {
        "turn_id": 2, "planner_action": "final", "tool_name": None,
        "raw_tool_name": None, "tool_args": None, "tool_observation": None,
        "planner_raw": response, "planner_calls": {}, "planner_parse_error": None,
        "final_answer": answer, "video_id": inputs.get("video_id"),
        "reasoner_thinking": str(getattr(pred, "reasoner_thinking", "") or ""),
        "reasoner_token_usage": getattr(pred, "reasoner_token_usage", {}) or {},
    }
    return [first, second]


def _result_row(
    example: dict, pred, *, caption_mode: str, caption_source: CachedCaption | None,
    caption_profile: dict | None, caption_prompt_yaml: Path | None,
    caption_template_name: str | None,
) -> dict:
    inputs = example["inputs"]
    raw = str(getattr(pred, "response_text", "") or "")
    caption = str(getattr(pred, "caption", "") or "")
    answer = normalize_option_letter(raw)
    valid = answer in "ABCDEF"[:len(json.loads(inputs["options_json"]))] if answer else False
    status = "empty" if (not caption.strip() or response_is_empty(raw)) else ("ok" if valid else "invalid_answer")
    response = f"{answer}. {raw}" if valid else raw
    if status == "empty":
        response = "[ERROR] captioner or reasoner returned no visible content"
    row = build_result_row(example["cut"], response)
    trace = _turn_trace(
        pred=pred, inputs=inputs, caption_mode=caption_mode, caption_source=caption_source,
        caption_profile=caption_profile, caption_prompt_yaml=caption_prompt_yaml,
        caption_template_name=caption_template_name)
    correct = bool(status == "ok" and avqa_metric(
        SimpleNamespace(answer=example["gold"]), SimpleNamespace(answer=answer)))
    qd = row["question_data"]
    qd.update(response=response, response_text=raw, caption_mode=caption_mode,
              status=status, predicted_answer=answer, correct=correct,
              token_usage=getattr(pred, "reasoner_token_usage", {}) or {}, turn_trace=trace)
    if getattr(pred, "reasoner_thinking", ""):
        qd["think"] = pred.reasoner_thinking
    row.update(question_id=inputs["question_id"], response_text=raw, answer=answer,
               gold=example["gold"], correct=correct, status=status,
               caption=caption, caption_mode=caption_mode, turn_trace=trace,
               caption_token_usage=getattr(pred, "caption_token_usage", {}) or {},
               reasoner_token_usage=getattr(pred, "reasoner_token_usage", {}) or {})
    return row


def _error_row(example: dict, *, caption_mode: str, exc: Exception) -> dict:
    row = build_result_row(example["cut"], f"[ERROR] {type(exc).__name__}")
    inputs = example["inputs"]
    trace = [{
        "turn_id": 1, "planner_action": "error", "tool_name": "ask_caption",
        "tool_args": {"video_path": inputs["video_path"], "audio_path": inputs["audio_path"]},
        "tool_observation": None, "final_answer": None, "tool_error": type(exc).__name__,
        "video_id": inputs.get("video_id"),
    }]
    row["question_data"].update(caption_mode=caption_mode, status="error", turn_trace=trace)
    row.update(question_id=inputs["question_id"], response_text="", answer="",
               gold=example["gold"], correct=False, status="error", caption="",
               caption_mode=caption_mode, turn_trace=trace, error_type=type(exc).__name__)
    return row


def evaluate(
    examples: list[dict], *, program: CaptionThenTextQAProgram, caption_mode: str,
    caption_dir: Path | None, caption_profile: dict | None,
    caption_prompt_yaml: Path | None, caption_template_name: str | None,
    output_dir: Path, output_jsonl: Path, workers: int, batch_size: int,
    reliable: ReliableQwenExecutor,
) -> dict:
    rows_by_id, remaining = {}, []
    for example in examples:
        cached = load_cached_row_from_question_json(output_dir, example["cut"])
        if _cached_caption_row_valid(cached, caption_mode):
            rows_by_id[example["inputs"]["question_id"]] = cached
        else:
            remaining.append(example)
    cached_count = len(rows_by_id)
    print(f"Cached samples: {cached_count}; remaining samples: {len(remaining)}", flush=True)

    def run_item(example):
        source = None
        try:
            if caption_mode == "cached":
                assert caption_dir is not None
                source = load_cached_caption(example["cut"], caption_dir)
                inputs = example["inputs"]
                pred = program.forward_from_caption(
                    caption=source.text, question=inputs["question"],
                    options_json=inputs["options_json"], video_id=inputs.get("video_id"),
                    question_id=inputs.get("question_id"))
            else:
                pred = program(**example["inputs"])
            row = _result_row(
                example, pred, caption_mode=caption_mode, caption_source=source,
                caption_profile=caption_profile, caption_prompt_yaml=caption_prompt_yaml,
                caption_template_name=caption_template_name)
        except Exception as exc:
            row = _error_row(example, caption_mode=caption_mode, exc=exc)
        save_question_data(output_dir, example["cut"], row)
        return row

    write_aggregate_outputs(examples, rows_by_id, output_dir, output_jsonl)
    for start in range(0, len(remaining), batch_size):
        items = remaining[start:start + batch_size]
        live_backend = str((caption_profile or {}).get("backend") or "qwen")
        if caption_mode == "live" and live_backend == "qwen":
            batch = reliable.run_batch(
                items, run_item=run_item,
                has_empty_qwen_response=lambda row: bool(
                    row["status"] == "empty" and not str(row.get("caption") or "").strip()),
                max_workers=workers)
            results = batch.results
        elif workers == 1:
            results = [run_item(item) for item in items]
        else:
            indexed = [None] * len(items)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_item, item): index for index, item in enumerate(items)}
                for future in as_completed(futures):
                    indexed[futures[future]] = future.result()
            results = indexed
        for row in results:
            rows_by_id[row["question_id"]] = row
        write_aggregate_outputs(examples, rows_by_id, output_dir, output_jsonl)
        print(f"Completed {len(rows_by_id)}/{len(examples)}", flush=True)
    counts = {key: 0 for key in ("correct", "ok", "empty", "invalid_answer", "error")}
    for row in rows_by_id.values():
        counts[row["status"]] += 1
        counts["correct"] += int(row["correct"])
    failed = counts["empty"] + counts["error"]
    return {**counts, "total": len(examples), "cached_samples": cached_count,
            "new_samples": len(remaining), "accuracy": counts["correct"] / len(examples),
            "status": "completed_with_errors" if failed else "completed"}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--caption-mode", choices=("live", "cached"), required=True)
    cli.add_argument("--caption-dir")
    cli.add_argument("--captioner-config-yaml", type=Path)
    cli.add_argument("--reasoner-config-yaml", type=Path, required=True)
    cli.add_argument("--caption-prompt-yaml", type=Path)
    cli.add_argument("--caption-template-name", default="QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3")
    cli.add_argument("--caption-system-template-name")
    cli.add_argument("--reasoner-prompt-yaml", type=Path, required=True)
    cli.add_argument("--reasoner-template-name", default="QA_PROMPT_TEMPLATE_CAPTION_OMNI")
    cli.add_argument("--reasoner-system-template-name")
    cli.add_argument("--input-jsonl", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--output-jsonl", type=Path)
    cli.add_argument("--inference-num-threads", type=int, default=1)
    cli.add_argument("--inference-batch-size", type=int)
    cli.add_argument("--expected-count", type=int, default=1197)
    cli.add_argument("--limit", type=int, help="First N examples for a smoke run; omit for the full input")
    cli.add_argument("--debug", action="store_true")
    cli.add_argument("--debug-limit", type=int, default=4)
    cli.add_argument("--print-config", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--allow-dirty", action="store_true")
    return cli


def main(argv=None) -> int:
    def operation() -> int:
        args = parser().parse_args(argv)
        with final_eval_directory_lock(args.output_dir):
            return _main(args)

    return run_with_completion_timer("Caption-reasoner inference", operation)


def _main(args) -> int:
    if args.debug:
        args.limit = args.debug_limit
    workers = args.inference_num_threads
    batch_size = args.inference_batch_size or workers
    if (workers < 1 or batch_size < 1 or args.expected_count < 0
            or (args.limit is not None and args.limit < 1)):
        raise ValueError("Invalid count/worker/batch arguments")
    output_jsonl = args.output_jsonl or args.output_dir / "output_test.jsonl"
    caption_dir = _optional_path(args.caption_dir)
    reasoner_source = load_text_reasoner_config_yaml(args.reasoner_config_yaml)
    reasoner_record = json.loads(json.dumps(reasoner_source))
    if args.caption_mode == "cached":
        caption_dir = caption_dir or _optional_path((reasoner_source.get("task") or {}).get("caption_dir"))
        if caption_dir is None or not caption_dir.is_dir():
            raise ValueError("cached mode requires an existing caption directory")
        load_text_reasoner_prompt(
            reasoner_prompt_yaml=args.reasoner_prompt_yaml,
            reasoner_template_name=args.reasoner_template_name,
            reasoner_system_template_name=args.reasoner_system_template_name)
        captioner_source, caption_profile, caption_prompt_yaml = {}, None, None
    else:
        (reasoner_record.get("task") or {}).pop("caption_dir", None)
        if args.captioner_config_yaml is None or args.caption_prompt_yaml is None:
            raise ValueError("live mode requires --captioner-config-yaml and --caption-prompt-yaml")
        captioner_source = load_captioner_config_yaml(args.captioner_config_yaml)
        caption_profile_obj = resolve_captioner_profile()
        caption_profile = asdict(caption_profile_obj)
        caption_profile["api_key"] = "<set>" if caption_profile_obj.api_key else "<unset>"
        caption_profile["backend"] = captioner_backend(caption_profile_obj)
        load_caption_reasoner_prompts(
            caption_prompt_yaml=args.caption_prompt_yaml,
            caption_template_name=args.caption_template_name,
            caption_system_template_name=args.caption_system_template_name,
            reasoner_prompt_yaml=args.reasoner_prompt_yaml,
            reasoner_template_name=args.reasoner_template_name,
            reasoner_system_template_name=args.reasoner_system_template_name)
        caption_prompt_yaml = args.caption_prompt_yaml.resolve()
        if not args.dry_run:
            if not caption_profile_obj.base_url:
                raise ValueError("Live captioner requires a resolved base URL")
            if (
                isinstance(caption_profile_obj, GeminiModelProfile)
                and not caption_profile_obj.api_key
                and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            ):
                raise ValueError(
                    "Official Vertex Gemini requires GEMINI_API_KEY or "
                    "GOOGLE_APPLICATION_CREDENTIALS"
                )
    reasoner_effective = resolve_text_lm_config("text_reasoner", default_model="openai/gpt-4.1")
    reasoner_public = redact_sensitive(reasoner_effective)
    examples = prepare_examples(args.input_jsonl, args.expected_count, args.limit)
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    if dirty and not args.allow_dirty:
        raise ValueError("Working tree is dirty; commit changes or use --allow-dirty")
    reliable = ReliableQwenExecutor()
    if args.caption_mode == "live" and captioner_backend(caption_profile_obj) == "qwen":
        caption_transport = {
            "media_url_mode": caption_profile_obj.media_url_mode,
            "stream": caption_profile_obj.stream,
            "include_enable_thinking": caption_profile_obj.include_enable_thinking,
        }
    elif args.caption_mode == "live":
        caption_transport = {
            "api": "generateContent", "provider": "vertex",
            "media": (
                "inlineData" if caption_profile_obj.media_mode == "inline" else "fileData"
            ),
        }
    else:
        caption_transport = None
    caption_reliability = {
        "max_batch_retries": reliable.max_batch_retries,
        "fallback_max_workers": reliable.fallback_max_workers,
        "primary": asdict(reliable.primary_profile),
    }
    caption_provenance = ({
        "caption_dir": str(caption_dir), "prompt": None, "model": None,
    } if args.caption_mode == "cached" else {
        "caption_dir": None,
        "prompt": {"yaml": str(caption_prompt_yaml), "template_name": args.caption_template_name},
        "model": {"config_yaml": str(args.captioner_config_yaml.resolve()),
                  "source_config": redact_sensitive(captioner_source),
                  "effective_profile": caption_profile,
                  "transport": caption_transport,
                  "reliability": (caption_reliability if caption_profile["backend"] == "qwen" else None)},
    })
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "git_dirty": dirty, "python": sys.executable,
        "versions": {name: version(name) for name in ("dspy", "openai", "pyyaml")},
        "mode": "dry_run" if args.dry_run else "live", "workflow": "caption_then_text_qa",
        "caption_mode": args.caption_mode, "instruction_optimization": False,
        "input_jsonl": str(args.input_jsonl.resolve()), "input_sha256": sha256(args.input_jsonl),
        "reasoner": {"config_yaml": str(args.reasoner_config_yaml.resolve()),
                     "source_config": redact_sensitive(reasoner_record),
                     "effective": reasoner_public,
                     "prompt": {"yaml": str(args.reasoner_prompt_yaml.resolve()),
                                "template_name": args.reasoner_template_name}},
        "caption": caption_provenance, "workers": workers, "batch_size": batch_size,
        "limit": args.limit, "selected_count": len(examples), "expected_count": args.expected_count,
    }
    signature_payload = {
        "workflow": manifest["workflow"], "caption_mode": args.caption_mode,
        "input_sha256": manifest["input_sha256"], "reasoner": manifest["reasoner"],
        "caption": caption_provenance,
        "reasoner_prompt_sha256": sha256(args.reasoner_prompt_yaml),
        "caption_prompt_sha256": (
            sha256(args.caption_prompt_yaml) if args.caption_mode == "live" else None),
    }
    manifest["resume_signature"] = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    metadata_path = args.output_dir / "run_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("mode") != "dry_run" and previous.get("resume_signature") != manifest["resume_signature"]:
            raise ValueError("Resume configuration differs; use a new output directory")
        if previous.get("mode") == "dry_run":
            write_json(metadata_path, manifest)
    else:
        write_json(metadata_path, manifest)
    save_resolved_experiment_config(args.output_dir, {
        "mode": "inference", "arguments": vars(args),
        "source_configs": {
            "captioner": {"path": args.captioner_config_yaml if args.caption_mode == "live" else None,
                          "config": captioner_source if args.caption_mode == "live" else None},
            "reasoner": {"path": args.reasoner_config_yaml, "config": reasoner_record},
            "caption_prompt": ({"path": caption_prompt_yaml, "template_name": args.caption_template_name}
                               if args.caption_mode == "live" else None),
            "reasoner_prompt": {"path": args.reasoner_prompt_yaml,
                                "template_name": args.reasoner_template_name},
        },
        "effective": {"caption_mode": args.caption_mode, "caption": caption_provenance,
                      "reasoner": reasoner_public, "caption_dir": str(caption_dir) if caption_dir else None,
                      "inference_num_threads": workers, "inference_batch_size": batch_size},
    }, print_config=args.print_config)
    if args.dry_run:
        print(f"Preflight passed: {len(examples)} selected examples; no API calls")
        return 0
    invocation = dict(manifest, status="running")
    write_json(args.output_dir / "inference_only_metadata.json", invocation)
    program = CaptionThenTextQAProgram(
        captioner_profile=caption_profile_obj if args.caption_mode == "live" else None,
        text_lm=build_text_lm("text_reasoner"))
    try:
        summary = evaluate(
            examples, program=program, caption_mode=args.caption_mode, caption_dir=caption_dir,
            caption_profile=caption_profile, caption_prompt_yaml=caption_prompt_yaml,
            caption_template_name=args.caption_template_name if args.caption_mode == "live" else None,
            output_dir=args.output_dir, output_jsonl=output_jsonl, workers=workers,
            batch_size=batch_size, reliable=reliable)
    except BaseException as exc:
        invocation.update(status="interrupted_or_failed", error_type=type(exc).__name__,
                          finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "inference_only_metadata.json", invocation)
        raise
    invocation.update(summary, finished_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output_dir / "inference_only_metadata.json", invocation)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
