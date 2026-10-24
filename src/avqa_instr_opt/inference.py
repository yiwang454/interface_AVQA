"""Fixed-prompt Daily-Omni QA evaluation; no instruction optimization."""

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

from .clients.reliable import (
    PRIMARY_EMBEDDED_AUDIO_QWEN_PROFILE,
    ReliableQwenExecutor,
    active_qwen_request_profile,
    response_is_empty,
)
from .clients.qwen_omni import safe_exception_summary
from .config import GeminiModelProfile, QwenModelProfile, load_inference_config, save_resolved_experiment_config
from .data import build_input_state, build_result_row, read_jsonl
from .metrics import avqa_metric, normalize_option_letter
from .programs.direct_qwen import DirectQwenAVQAProgram
from .programs.direct_gemini import DirectGeminiAVQAProgram
from .prompt_config import DEFAULT_PROMPT_YAML, load_prompt_config, render_prompt, prompt_value
from .timing import run_with_completion_timer
from .inference_cache import (atomic_write_json, final_eval_directory_lock,
    load_cached_row_from_question_json, question_cache_path, save_question_data,
    write_aggregate_outputs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    atomic_write_json(path, value)


def prepare_examples(
    path: Path,
    expected_count: int,
    limit: int | None,
    *,
    require_audio: bool = True,
) -> list[dict]:
    """Validate the entire input before selecting a deterministic smoke subset."""
    cuts = read_jsonl(path)
    if not cuts or (expected_count and len(cuts) != expected_count):
        raise ValueError(f"Input has {len(cuts)} rows; expected {expected_count or 'non-empty input'}")
    examples, seen = [], set()
    for cut in cuts:
        state = build_input_state(cut, require_audio=require_audio)
        qid = state["question_id"]
        if qid in seen:
            raise ValueError(f"Duplicate cut ID: {qid}")
        seen.add(qid)
        question_cache_path(Path("."), cut)
        options = state.pop("options")
        if not isinstance(options, list) or not 2 <= len(options) <= 6:
            raise ValueError(f"Invalid options for {qid}")
        gold = normalize_option_letter(cut["supervisions"][0]["custom"].get("answer", ""))
        if gold not in "ABCDEF"[:len(options)] or not gold:
            raise ValueError(f"Invalid gold answer for {qid}")
        for field in ("video_path", "audio_path"):
            if field == "audio_path" and not state[field]:
                continue
            media_path = Path(state[field]).resolve()
            stat = media_path.stat()
            if not media_path.is_file() or not stat.st_size:
                raise ValueError(f"Invalid media file for {qid}: {media_path}")
            state[field] = str(media_path)
        state["options_json"] = json.dumps(options, ensure_ascii=False)
        examples.append({"cut": cut, "inputs": state, "gold": gold})
    return examples if limit is None else examples[:limit]


def evaluate(examples: list[dict], program, executor: ReliableQwenExecutor | None,
             output_dir: Path, workers: int, batch_size: int, output_jsonl: Path | None = None) -> dict:
    """Resume valid per-cut JSON; regenerate errors and rebuild ordered aggregate outputs."""
    output_jsonl = output_jsonl or output_dir / "output_test.jsonl"
    rows_by_id, remaining = {}, []
    for example in examples:
        cached = load_cached_row_from_question_json(output_dir, example["cut"])
        if cached is None:
            remaining.append(example)
        else:
            rows_by_id[example["inputs"]["question_id"]] = cached
    cached_count = len(rows_by_id)
    print(f"Cached samples: {cached_count}; remaining samples: {len(remaining)}", flush=True)
    attempted_profiles = []
    def run_item(example):
        inputs = example["inputs"]
        row = build_result_row(example["cut"], "")
        try:
            pred = program(**inputs)
            response = str(pred.response_text or "")
            answer = normalize_option_letter(response)
            status = "empty" if response_is_empty(response) else (
                "ok" if answer and answer in "ABCDEF"[:len(json.loads(inputs["options_json"]))]
                else "invalid_answer"
            )
            row.update(response_text=response, answer=answer, status=status,
                       thinking=getattr(pred, "thinking", ""),
                       token_usage=getattr(pred, "token_usage", {}))
        except Exception as exc:
            # Exception strings can contain request credentials; persist type only.
            print(
                "[error] Final-eval task call failed for "
                f"question_id={inputs['question_id']}: {safe_exception_summary(exc)}",
                flush=True,
            )
            row.update(response_text="", answer="", status="error", error_type=type(exc).__name__)
        row["question_data"]["response"] = row["response_text"]
        row["question_data"]["token_usage"] = row.get("token_usage", {})
        if row.get("thinking"):
            row["question_data"]["think"] = row["thinking"]
        row["question_id"] = inputs["question_id"]
        row["gold"] = example["gold"]
        row["correct"] = bool(row["status"] == "ok" and avqa_metric(
            SimpleNamespace(answer=example["gold"]), SimpleNamespace(answer=row["answer"])))
        profile = active_qwen_request_profile()
        row["request_profile"] = asdict(profile) if profile else None
        row["question_data"].update(
            response_text=row["response_text"], status=row["status"],
            predicted_answer=row["answer"], correct=row["correct"],
            request_profile=row["request_profile"],
        )
        if row["status"] in {"empty", "error"}:
            row["question_data"]["response"] = "[ERROR] " + row.get("error_type", "Model returned no visible content")
        save_question_data(output_dir, example["cut"], row)
        return row


    def write_aggregates():
        write_aggregate_outputs(examples, rows_by_id, output_dir, output_jsonl)

    write_aggregates()
    for start in range(0, len(remaining), batch_size):
        current_batch = remaining[start:start + batch_size]
        if executor is not None:
            batch = executor.run_batch(
                current_batch, run_item=run_item,
                has_empty_qwen_response=lambda row: row["status"] == "empty", max_workers=workers)
            attempted_profiles.append(batch.attempted_profiles)
            rows = batch.results
        else:
            # Gemini transport handles its own retry policy and has no Qwen
            # media-layout fallback. Preserve input ordering in aggregate files.
            rows = [None] * len(current_batch)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_item, example): index
                           for index, example in enumerate(current_batch)}
                for future in as_completed(futures):
                    rows[futures[future]] = future.result()
        for row in rows:
            rows_by_id[row["question_id"]] = row
        write_aggregates()
        print(f"Completed {len(rows_by_id)}/{len(examples)}", flush=True)
    counts = {key: 0 for key in ("correct", "ok", "empty", "invalid_answer", "error")}
    for row in rows_by_id.values():
        counts[row["status"]] += 1
        counts["correct"] += int(row["correct"])
    failed = counts["empty"] + counts["error"]
    return {**counts, "total": len(examples), "accuracy": counts["correct"] / len(examples),
            "cached_samples": cached_count, "new_samples": len(remaining),
            "status": "completed_with_errors" if failed else "completed",
            "attempted_profiles_by_batch": attempted_profiles}


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--perception-config-yaml", dest="config", type=Path, required=True)
    cli.add_argument("--prompt-yaml", dest="prompt_config", type=Path, default=DEFAULT_PROMPT_YAML)
    cli.add_argument("--output-dir", dest="output_dir", type=Path, required=True)
    cli.add_argument("--output-jsonl", type=Path, help="Defaults to OUTPUT_DIR/output_test.jsonl")
    cli.add_argument("--print-config", action="store_true")
    cli.add_argument("--debug", action="store_true")
    cli.add_argument("--debug-limit", type=int, default=4)
    cli.add_argument("--input-jsonl", type=Path)
    cli.add_argument("--openai-model")
    cli.add_argument("--openai-base-url")
    cli.add_argument("--openai-api-key")
    cli.add_argument("--temperature", type=float)
    cli.add_argument("--max-tokens", type=int)
    cli.add_argument("--qwen_seed", "--qwen-seed", type=int)
    cli.add_argument("--retry-delay-s", type=float)
    cli.add_argument("--launcher-log-name", help=argparse.SUPPRESS)
    cli.add_argument("--expected-count", type=int, default=1197, help="0 disables exact count check")
    cli.add_argument("--limit", type=int, help="First N examples for a smoke run; omit for the full input")
    cli.add_argument("--inference-num-threads", "--workers", dest="workers", type=int)
    cli.add_argument("--inference-batch-size", "--batch-size", dest="batch_size", type=int)
    cli.add_argument("--dry-run", action="store_true", help="Save preflight metadata without API calls")
    cli.add_argument("--allow-dirty", action="store_true", help="Development only; record dirty state")
    return cli


def main(argv=None) -> int:
    def operation() -> int:
        args = parser().parse_args(argv)
        with final_eval_directory_lock(args.output_dir):
            return _main(args)

    return run_with_completion_timer("Direct inference", operation)


def _main(args) -> int:
    profile, source_config = load_inference_config(args)
    if args.debug:
        args.limit = args.debug_limit
    if args.output_jsonl is not None and args.output_jsonl.suffix != ".jsonl":
        raise ValueError("--output-jsonl must have a .jsonl suffix")
    if (args.workers < 1 or args.batch_size < 1 or args.expected_count < 0
            or (args.limit is not None and args.limit < 1)):
        raise ValueError("Invalid count/worker/batch arguments")
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    if dirty and not args.allow_dirty:
        raise ValueError("Working tree is dirty; commit changes or use --allow-dirty for development")
    is_gemini = isinstance(profile, GeminiModelProfile)
    if not is_gemini and (profile.video_only or profile.video_first):
        raise ValueError(
            "This runner requires an audio-visual primary with text last; "
            "media layout is owned by ReliableQwenExecutor"
        )
    if profile.max_retries < 1:
        raise ValueError("Model max_retries must be >= 1")
    if not args.dry_run:
        if is_gemini:
            has_vertex_credentials = bool(
                os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            )
            if not profile.base_url or (not profile.api_key and not has_vertex_credentials):
                raise ValueError(
                    "Official Vertex Gemini requires GEMINI_BASE_URL plus either "
                    "GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS."
                )
        elif not profile.api_key or not profile.base_url:
            raise ValueError(
                "Qwen model YAML must provide qwen_base_url and qwen_api_key "
                "(use qwen_api_key: EMPTY for an unauthenticated local server)"
            )
    prompts = load_prompt_config(args.prompt_config)
    render_prompt("direct_qwen", "qa_prompt_template",
                  user_instruction=prompt_value("direct_qwen", "user_instruction"),
                  question="preflight", formatted_options="A. one\nB. two", options=["A. one", "B. two"])
    prompt_value("direct_qwen", "system_prompt")
    examples = prepare_examples(
        args.input_jsonl,
        args.expected_count,
        args.limit,
        require_audio=not (is_gemini and profile.video_only),
    )
    executor = None if is_gemini else ReliableQwenExecutor(
        primary_profile=(
            PRIMARY_EMBEDDED_AUDIO_QWEN_PROFILE
            if profile.use_audio_in_video
            else None
        )
    )
    public_profile = asdict(profile)
    public_profile["api_key"] = "<set>" if profile.api_key else "<unset>"
    # Do not use the generic token-key redactor: max_tokens is experiment data.
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "git_dirty": dirty, "python": sys.executable,
        "versions": {name: version(name) for name in ("dspy", "openai", "pyyaml")},
        "input_jsonl": str(args.input_jsonl.resolve()), "input_sha256": sha256(args.input_jsonl),
        "prompt_yaml": str(args.prompt_config.resolve()), "prompt_sha256": sha256(args.prompt_config),
        "perception_yaml": str(args.config.resolve()), "perception_sha256": sha256(args.config),
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))},
        "profile": public_profile,
        "workers": args.workers, "batch_size": args.batch_size, "limit": args.limit,
        "selected_count": len(examples), "expected_count": args.expected_count,
        "mode": "dry_run" if args.dry_run else "live", "instruction_optimization": False,
    }
    if is_gemini:
        manifest["transport"] = {
            "backend": "gemini", "api": "generateContent",
            "provider": "vertex",
            "media": "inlineData" if profile.media_mode == "inline" else "fileData",
            "retry_degenerate_response": True, "include_thoughts": profile.include_thoughts,
            "thinking_budget": profile.thinking_budget,
        }
        manifest["reliability"] = None
    else:
        manifest["transport"] = {
            "media_url_mode": profile.media_url_mode,
            "stream": profile.stream,
            "include_enable_thinking": profile.include_enable_thinking,
            "retry_degenerate_response": False, "stream_empty_fallback": True,
        }
        manifest["reliability"] = {"max_batch_retries": executor.max_batch_retries,
                                   "fallback_max_workers": executor.fallback_max_workers,
                                   "primary": asdict(executor.primary_profile)}
    # Cache compatibility follows effective experimental settings, not commit identity.
    signature_payload = {"input_sha256": manifest["input_sha256"], "profile": public_profile,
                         "prompts": prompts["direct_qwen"], "transport": manifest["transport"],
                         "reliability": manifest["reliability"]}
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    manifest["resume_signature"] = signature
    metadata_path = args.output_dir / "run_metadata.json"
    if metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("mode") != "dry_run" and previous.get("resume_signature") != signature:
            raise ValueError("Resume configuration differs or predates compatible cache metadata; use a new output directory")
        if previous.get("mode") == "dry_run" and not args.dry_run:
            write_json(metadata_path, manifest)
    else:
        write_json(metadata_path, manifest)
    if not args.dry_run:
        manifest["status"] = "running"
        write_json(args.output_dir / "inference_only_metadata.json", manifest)
    save_resolved_experiment_config(args.output_dir, {
        "mode": "inference",
        "arguments": vars(args),
        "effective": {"perception": public_profile, "transport": manifest["transport"],
                      "reliability": manifest["reliability"]},
        "perception_config_path": str(args.config.resolve()), "perception_config": source_config,
        "resolved_profile": public_profile, "transport": manifest["transport"],
        "reliability": manifest["reliability"], "prompt_config": str(args.prompt_config.resolve()),
        "source_configs": {"perception": {"path": str(args.config.resolve()), "config": source_config},
                           "prompt": {"path": str(args.prompt_config.resolve())}},
        "inference": {"num_threads": args.workers, "batch_size": args.batch_size,
                      "output_jsonl": str(args.output_jsonl or args.output_dir / "output_test.jsonl")},
        "BATCH_SIZE": args.batch_size, "NUM_WORKERS": args.workers,
        "INPUT_JSONL": str(args.input_jsonl.resolve()), "OUTPUT_DIR_OVERRIDE": os.getenv("OUTPUT_DIR_OVERRIDE"),
        "output_dir": str(args.output_dir.resolve()), "limit": args.limit,
        "expected_count": args.expected_count, "dry_run": args.dry_run,
    }, print_config=args.print_config)
    if args.dry_run:
        print(f"Preflight passed: {len(examples)} selected examples; commit={commit}; no API calls")
        return 0
    try:
        program = (DirectGeminiAVQAProgram(profile=profile) if is_gemini
                   else DirectQwenAVQAProgram(profile=profile))
        summary = evaluate(examples, program, executor,
                           args.output_dir, args.workers, args.batch_size,
                           args.output_jsonl)
    except BaseException as exc:
        manifest.update(status="interrupted_or_failed", error_type=type(exc).__name__,
                        finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output_dir / "inference_only_metadata.json", manifest)
        raise
    manifest.update(summary)
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output_dir / "inference_only_metadata.json", manifest)
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
