"""Optimize direct-MLLM or caption-reasoner user instructions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import dspy

from .caption_inference import evaluate as evaluate_caption
from .captions import load_cached_caption
from .clients.reliable import ReliableQwenExecutor
from .clients.text_lm import build_text_lm, resolve_text_lm_config
from .config import (
    GeminiModelProfile,
    captioner_backend,
    load_captioner_config_yaml,
    load_optimizer_lm_config_yaml,
    load_perception_config_yaml,
    load_text_reasoner_config_yaml,
    redact_sensitive,
    resolve_captioner_profile,
    resolve_gemini_profile,
    resolve_qwen_profile,
    save_resolved_experiment_config,
)
from .inference import evaluate as evaluate_direct
from .inference import prepare_examples, sha256, write_json
from .inference_cache import final_eval_directory_lock
from .optimization import (
    CaptionInputReasonerProgram,
    PromptTargetProgram,
    load_optimizer_config,
    optimize_with_copro,
    optimize_with_gepa,
    optimize_with_miprov2,
    validate_optimizer_config,
    validate_optimize_targets,
    write_prompt_artifacts,
)
from .programs.caption_then_text import CaptionResponseCache, CaptionThenTextQAProgram
from .programs.direct_gemini import DirectGeminiAVQAProgram
from .programs.direct_qwen import DirectQwenAVQAProgram
from .prompt_config import (
    load_caption_reasoner_prompts,
    load_prompt_config,
    load_text_reasoner_prompt,
)
from .timing import run_with_completion_timer


_RUN_PATH_FIELDS = {
    "trainset_jsonl", "valset_jsonl", "output_program", "initial_program",
    "metadata_json", "optimizer_log_dir", "optimized_prompt_config_yaml",
    "trajectory_jsonl", "final_eval_output_jsonl", "final_eval_output_dir",
    "gepa_log_dir",
}
_RUN_INT_FIELDS = {"train_limit", "gepa_seed"}
_RUN_FIELDS = _RUN_PATH_FIELDS | _RUN_INT_FIELDS | {"algorithm", "optimize_targets"}


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return Path(value).expanduser().resolve()


def _expand(value: str, *, field: str) -> str:
    expanded = os.path.expandvars(value)
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in optimizer run.{field}: {value}")
    return expanded


def _apply_run_config(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    run = config.get("run") or {}
    if not isinstance(run, dict):
        raise ValueError("Optimizer config run section must be a mapping")
    unknown = sorted(set(run) - _RUN_FIELDS)
    if unknown:
        raise ValueError("Unknown optimizer run key(s): " + ", ".join(unknown))
    if run.get("algorithm"):
        if args.algorithm and args.algorithm != run["algorithm"]:
            raise ValueError("CLI algorithm differs from optimizer config run.algorithm")
        args.algorithm = run["algorithm"]
    if args.optimize_targets is None and run.get("optimize_targets") is not None:
        value = run["optimize_targets"]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("run.optimize_targets must be a list of strings")
        args.optimize_targets = tuple(value)
    for field in _RUN_PATH_FIELDS:
        value = run.get(field)
        if value is not None and getattr(args, field, None) is None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"run.{field} must be a non-empty path string")
            setattr(args, field, Path(_expand(value, field=field)))
    if args.train_limit is None and run.get("train_limit") is not None:
        args.train_limit = int(run["train_limit"])
    if args.gepa_seed is not None:
        config["seed"] = args.gepa_seed
    elif run.get("gepa_seed") is not None and "seed" not in config:
        config["seed"] = int(run["gepa_seed"])
    if run.get("gepa_log_dir") is not None and "log_dir" not in config:
        config["log_dir"] = _expand(str(run["gepa_log_dir"]), field="gepa_log_dir")
    return run


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--algorithm", choices=("gepa", "copro", "miprov2"))
    cli.add_argument("--workflow", choices=("direct", "caption_reasoner"), required=True)
    cli.add_argument("--caption-mode", choices=("live", "cached"))
    cli.add_argument("--optimize-target", action="append", dest="optimize_targets")
    cli.add_argument("--optimizer-config", type=Path, required=True)
    cli.add_argument("--optimizer-lm-config-yaml", type=Path, required=True)
    cli.add_argument(
        "--optimizer-lm-seed",
        type=int,
        help="Override the prompt/reflection LM seed after loading its YAML.",
    )
    cli.add_argument(
        "--copro-init-temperature",
        type=float,
        help="Override COPRO init_temperature after loading its optimizer YAML.",
    )
    cli.add_argument("--trainset-jsonl", type=Path)
    cli.add_argument("--valset-jsonl", type=Path)
    cli.add_argument("--train-limit", type=int)
    cli.add_argument("--output-dir", type=Path)
    cli.add_argument("--output-program", type=Path)
    cli.add_argument("--initial-program", type=Path)
    cli.add_argument("--metadata-json", type=Path)
    cli.add_argument("--optimizer-log-dir", type=Path)
    cli.add_argument("--optimized-prompt-config-yaml", type=Path)
    cli.add_argument("--trajectory-jsonl", type=Path)
    cli.add_argument("--final-eval-input-jsonl", type=Path)
    cli.add_argument("--final-eval-output-jsonl", type=Path)
    cli.add_argument("--final-eval-output-dir", type=Path)
    cli.add_argument("--gepa-log-dir", type=Path)
    cli.add_argument("--gepa-seed", type=int)
    cli.add_argument("--perception-config-yaml", type=Path)
    cli.add_argument("--openai-model", help="Optional direct-Qwen model override")
    cli.add_argument("--openai-base-url", help="Optional direct-Qwen API base URL override")
    cli.add_argument("--openai-api-key", help="Optional direct-Qwen API key override")
    cli.add_argument("--prompt-yaml", type=Path)
    cli.add_argument("--caption-dir")
    cli.add_argument("--captioner-config-yaml", type=Path)
    cli.add_argument("--caption-prompt-yaml", type=Path)
    cli.add_argument("--caption-template-name", default="QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3")
    cli.add_argument("--reasoner-config-yaml", type=Path)
    cli.add_argument("--reasoner-prompt-yaml", type=Path)
    cli.add_argument("--reasoner-template-name", default="QA_PROMPT_TEMPLATE_CAPTION_OMNI")
    cli.add_argument("--num-threads", type=int, default=1)
    cli.add_argument(
        "--optimizer-num-threads", type=int,
        help="Override optimizer task-evaluation concurrency from its YAML.",
    )
    cli.add_argument("--final-eval-num-threads", type=int, default=1)
    cli.add_argument("--final-eval-batch-size", type=int)
    cli.add_argument("--expected-final-count", type=int, default=1197)
    cli.add_argument("--optimization-train-only", action="store_true")
    cli.add_argument("--inference-only", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--allow-dirty", action="store_true")
    cli.add_argument("--print-config", action="store_true")
    return cli


def _qwen_profile(role: str):
    return resolve_qwen_profile(role)


def _examples(
    path: Path, limit: int | None = None, *, require_audio: bool = True
) -> list[dict]:
    return prepare_examples(
        path, expected_count=0, limit=limit, require_audio=require_audio
    )


def _dspy_examples(examples: list[dict], *, caption_dir: Path | None = None):
    result = []
    for example in examples:
        inputs = dict(example["inputs"])
        if caption_dir is not None:
            inputs["caption"] = load_cached_caption(example["cut"], caption_dir).text
        item = dspy.Example(**inputs, answer=example["gold"])
        result.append(item.with_inputs(*inputs.keys()))
    return result


def _public_profile(profile: Any | None):
    if profile is None:
        return None
    payload = asdict(profile)
    payload["api_key"] = "<set>" if profile.api_key else "<unset>"
    return payload


def _build_task(args: argparse.Namespace):
    source: dict[str, Any] = {}
    caption_dir = None
    caption_cache = None
    caption_profile = None
    if args.workflow == "direct":
        if args.caption_mode is not None:
            raise ValueError("--caption-mode is only valid for caption_reasoner workflow")
        if args.perception_config_yaml is None or args.prompt_yaml is None:
            raise ValueError("direct workflow requires perception and prompt YAMLs")
        source["perception"] = load_perception_config_yaml(args.perception_config_yaml)
        for field, suffix in (("openai_model", "MODEL"), ("openai_base_url", "BASE_URL"),
                              ("openai_api_key", "API_KEY")):
            value = getattr(args, field)
            if value is not None:
                os.environ[f"QWEN_{suffix}"] = str(value)
        load_prompt_config(args.prompt_yaml)
        direct_backend = os.environ.get("DIRECT_BACKEND", "qwen").strip().lower()
        if direct_backend == "gemini":
            task_profile = resolve_gemini_profile("direct_gemini")
            base_program = DirectGeminiAVQAProgram(profile=task_profile)
        elif direct_backend == "qwen":
            task_profile = _qwen_profile("direct_qwen")
            base_program = DirectQwenAVQAProgram(profile=task_profile)
        else:
            raise ValueError(f"Unsupported direct backend: {direct_backend!r}")
        caption_mode = None
    else:
        if args.caption_mode is None:
            raise ValueError("caption_reasoner workflow requires --caption-mode")
        if args.reasoner_config_yaml is None or args.reasoner_prompt_yaml is None:
            raise ValueError("caption_reasoner requires reasoner config and prompt YAMLs")
        reasoner_source = load_text_reasoner_config_yaml(args.reasoner_config_yaml)
        source["reasoner"] = reasoner_source
        caption_mode = args.caption_mode
        if caption_mode == "cached":
            caption_dir = _optional_path(args.caption_dir) or _optional_path(
                (reasoner_source.get("task") or {}).get("caption_dir")
            )
            if caption_dir is None or not caption_dir.is_dir():
                raise ValueError("cached caption optimization requires an existing caption directory")
            load_text_reasoner_prompt(
                reasoner_prompt_yaml=args.reasoner_prompt_yaml,
                reasoner_template_name=args.reasoner_template_name,
            )
        else:
            if args.captioner_config_yaml is None or args.caption_prompt_yaml is None:
                raise ValueError("live caption optimization requires captioner config and prompt YAMLs")
            source["captioner"] = load_captioner_config_yaml(args.captioner_config_yaml)
            caption_profile = resolve_captioner_profile()
            load_caption_reasoner_prompts(
                caption_prompt_yaml=args.caption_prompt_yaml,
                caption_template_name=args.caption_template_name,
                reasoner_prompt_yaml=args.reasoner_prompt_yaml,
                reasoner_template_name=args.reasoner_template_name,
            )
            caption_cache = CaptionResponseCache()
        task_profile = caption_profile
        caption_program = CaptionThenTextQAProgram(
            captioner_profile=caption_profile, text_lm=build_text_lm("text_reasoner"),
            caption_cache=caption_cache,
        )
        base_program = (
            CaptionInputReasonerProgram(caption_program)
            if caption_mode == "cached" else caption_program
        )
    return {
        "base_program": base_program, "task_profile": task_profile,
        "caption_profile": caption_profile, "caption_dir": caption_dir,
        "caption_cache": caption_cache, "caption_mode": caption_mode,
        "source_configs": source,
    }


def _save_trajectories(program: dspy.Module, examples: list[dspy.Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples):
            try:
                prediction = program(**example.inputs())
                row = {
                    "example_index": index, "question_id": getattr(example, "question_id", None),
                    "gold": example.answer, "answer": getattr(prediction, "answer", ""),
                    "caption": getattr(prediction, "caption", None),
                    "response_text": getattr(prediction, "response_text", None),
                    "score": float(getattr(prediction, "answer", "") == example.answer),
                }
            except Exception as exc:
                row = {"example_index": index, "question_id": getattr(example, "question_id", None),
                       "gold": example.answer, "score": 0.0, "error_type": type(exc).__name__}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_final_eval(args: argparse.Namespace, task: dict, program: PromptTargetProgram):
    if args.final_eval_input_jsonl is None:
        return None
    output_dir = args.final_eval_output_dir or args.output_dir / "final_eval"
    output_jsonl = args.final_eval_output_jsonl or output_dir / "output_test.jsonl"
    if args.caption_mode == "live":
        # Final evaluation is a fresh phase: retain the optimized instruction but
        # do not reuse train/validation captions generated during search.
        base = program._base_program
        if isinstance(base, CaptionThenTextQAProgram):
            base.caption_cache = CaptionResponseCache()
    direct_video_only = (
        args.workflow == "direct"
        and isinstance(task["task_profile"], GeminiModelProfile)
        and task["task_profile"].video_only
    )
    examples = prepare_examples(
        args.final_eval_input_jsonl,
        args.expected_final_count,
        limit=None,
        require_audio=not direct_video_only,
    )
    with final_eval_directory_lock(output_dir):
        signature_payload = {
            "workflow": args.workflow,
            "caption_mode": args.caption_mode,
            "input_sha256": sha256(args.final_eval_input_jsonl),
            "optimized_prompts": {
                target: program.candidate_text(target) for target in program.target_paths
            },
            "captioner_profile": _public_profile(task["task_profile"]),
            "caption_dir": str(task["caption_dir"]) if task["caption_dir"] else None,
            "text_reasoner": (
                resolve_text_lm_config("text_reasoner", default_model="openai/gpt-4.1")
                if args.workflow == "caption_reasoner" else None
            ),
        }
        signature = hashlib.sha256(
            json.dumps(redact_sensitive(signature_payload), sort_keys=True).encode("utf-8")
        ).hexdigest()
        run_metadata_path = output_dir / "run_metadata.json"
        if run_metadata_path.exists():
            previous = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if previous.get("resume_signature") != signature:
                raise ValueError(
                    "Final-eval resume configuration differs; use a new output directory"
                )
        else:
            write_json(run_metadata_path, redact_sensitive({
                "workflow": args.workflow, "caption_mode": args.caption_mode,
                "instruction_optimization": True, "algorithm": args.algorithm,
                "input_jsonl": str(args.final_eval_input_jsonl.resolve()),
                "resume_signature": signature, "configuration": signature_payload,
            }))
        if args.workflow == "direct":
            return evaluate_direct(
                examples, program, ReliableQwenExecutor(), output_dir,
                args.final_eval_num_threads,
                args.final_eval_batch_size or args.final_eval_num_threads,
                output_jsonl,
            )
        return evaluate_caption(
            examples, program=program, caption_mode=args.caption_mode,
            caption_dir=task["caption_dir"], caption_profile=(
                {**(_public_profile(task["caption_profile"]) or {}),
                 "backend": captioner_backend(task["caption_profile"])}
                if task["caption_profile"] is not None else None
            ),
            caption_prompt_yaml=args.caption_prompt_yaml if args.caption_mode == "live" else None,
            caption_template_name=args.caption_template_name if args.caption_mode == "live" else None,
            output_dir=output_dir, output_jsonl=output_jsonl,
            workers=args.final_eval_num_threads,
            batch_size=args.final_eval_batch_size or args.final_eval_num_threads,
            reliable=ReliableQwenExecutor(),
        )


def _main(argv=None) -> int:
    args = parser().parse_args(argv)
    optimizer_config = load_optimizer_config(args.optimizer_config)
    run_config = _apply_run_config(args, optimizer_config)
    if args.algorithm is None:
        raise ValueError("algorithm is required in CLI or optimizer config run section")
    if args.copro_init_temperature is not None:
        if args.algorithm != "copro":
            raise ValueError("--copro-init-temperature is only valid with --algorithm copro")
        if args.copro_init_temperature < 0:
            raise ValueError("--copro-init-temperature must be non-negative")
        optimizer_config["init_temperature"] = args.copro_init_temperature
    if args.optimizer_num_threads is not None:
        if args.optimizer_num_threads < 1:
            raise ValueError("--optimizer-num-threads must be >= 1")
        optimizer_config["num_threads"] = args.optimizer_num_threads
    validate_optimizer_config(optimizer_config, args.algorithm)
    if args.algorithm == "miprov2" and importlib.util.find_spec("optuna") is None:
        raise RuntimeError(
            "MIPROv2 requires optuna; install with `pip install -e '.[optimizers]'`"
        )
    args.optimize_targets = tuple(args.optimize_targets or ())
    if args.trainset_jsonl is None or args.valset_jsonl is None:
        if not args.inference_only:
            raise ValueError("trainset and valset JSONLs are required")
    if args.output_dir is None:
        args.output_dir = args.output_program.parent if args.output_program else None
    if args.output_dir is None:
        raise ValueError("output directory or output program is required")
    args.output_dir = args.output_dir.resolve()
    args.output_program = args.output_program or args.output_dir / f"compiled_{args.algorithm}.json"
    args.initial_program = args.initial_program or args.output_dir / f"initial_{args.algorithm}_program.json"
    args.metadata_json = args.metadata_json or args.output_dir / f"compiled_{args.algorithm}_metadata.json"
    if args.num_threads < 1 or args.final_eval_num_threads < 1:
        raise ValueError("thread counts must be >= 1")

    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True).strip())
    if dirty and not args.allow_dirty:
        raise ValueError("Working tree is dirty; commit changes or use --allow-dirty")

    task = _build_task(args)
    validate_optimize_targets(
        args.optimize_targets, workflow=args.workflow, caption_mode=task["caption_mode"],
        algorithm=args.algorithm,
    )
    lm_role = "gepa_reflection" if args.algorithm == "gepa" else "optimizer_prompt"
    optimizer_lm_source = load_optimizer_lm_config_yaml(
        args.optimizer_lm_config_yaml, role=lm_role
    )
    if args.optimizer_lm_seed is not None:
        os.environ[f"{lm_role.upper()}_SEED"] = str(args.optimizer_lm_seed)
    optimizer_lm = build_text_lm(lm_role, default_model="openai/gpt-5.4")
    dspy.configure(lm=optimizer_lm)

    direct_video_only = (
        args.workflow == "direct"
        and isinstance(task["task_profile"], GeminiModelProfile)
        and task["task_profile"].video_only
    )
    train_raw = (
        _examples(
            args.trainset_jsonl,
            args.train_limit,
            require_audio=not direct_video_only,
        )
        if args.trainset_jsonl
        else []
    )
    val_raw = (
        _examples(args.valset_jsonl, require_audio=not direct_video_only)
        if args.valset_jsonl
        else []
    )
    trainset = _dspy_examples(train_raw, caption_dir=task["caption_dir"])
    valset = _dspy_examples(val_raw, caption_dir=task["caption_dir"])
    program = PromptTargetProgram(task["base_program"], args.optimize_targets)

    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
        "git_dirty": dirty, "python": sys.executable,
        "versions": {name: version(name) for name in ("dspy", "gepa", "openai", "pyyaml")},
        "algorithm": args.algorithm, "workflow": args.workflow,
        "caption_mode": task["caption_mode"], "optimize_targets": list(args.optimize_targets),
        "trainset_jsonl": str(args.trainset_jsonl.resolve()) if args.trainset_jsonl else None,
        "valset_jsonl": str(args.valset_jsonl.resolve()) if args.valset_jsonl else None,
        "trainset_sha256": sha256(args.trainset_jsonl) if args.trainset_jsonl else None,
        "valset_sha256": sha256(args.valset_jsonl) if args.valset_jsonl else None,
        "train_examples": len(trainset), "val_examples": len(valset),
        "optimizer_config_yaml": str(args.optimizer_config.resolve()),
        "optimizer_config": optimizer_config, "run_config": run_config,
        "optimizer_lm": {
            "config_yaml": str(args.optimizer_lm_config_yaml.resolve()),
            "source_config": optimizer_lm_source,
            "effective": resolve_text_lm_config(lm_role, default_model="openai/gpt-5.4"),
        },
        "task_models": {
            "qwen": (
                _public_profile(task["task_profile"])
                if args.workflow == "direct" and
                not isinstance(task["task_profile"], GeminiModelProfile) else None
            ),
            "direct": (
                {**(_public_profile(task["task_profile"]) or {}),
                 "backend": getattr(task["base_program"], "task_backend", "qwen")}
                if args.workflow == "direct" else None
            ),
            "captioner": (
                {**(_public_profile(task["caption_profile"]) or {}),
                 "backend": captioner_backend(task["caption_profile"])}
                if task["caption_profile"] is not None else None
            ),
            "text_reasoner": (
                resolve_text_lm_config("text_reasoner", default_model="openai/gpt-4.1")
                if args.workflow == "caption_reasoner" else None
            ),
        },
        "caption_cache_policy": (
            "candidate_instruction_and_sample" if args.caption_mode == "live" else
            "external_caption_directory" if args.caption_mode == "cached" else None
        ),
        "caption_dir": str(task["caption_dir"]) if task["caption_dir"] else None,
        "source_configs": task["source_configs"], "status": "preflight",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.metadata_json, redact_sensitive(metadata))
    save_resolved_experiment_config(args.output_dir, {
        "mode": "optimization", "arguments": vars(args), "metadata": metadata,
    }, print_config=args.print_config)
    if args.dry_run:
        print(
            f"Optimization preflight passed: algorithm={args.algorithm}, workflow={args.workflow}, "
            f"targets={','.join(args.optimize_targets)}, train={len(trainset)}, val={len(valset)}; "
            "no model calls"
        )
        return 0

    if args.inference_only:
        if args.final_eval_input_jsonl is None:
            raise ValueError("--inference-only requires --final-eval-input-jsonl")
        program.load(str(args.output_program))
        summary = _run_final_eval(args, task, program)
        metadata.update(status="inference_only_completed", final_eval=summary,
                        finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.metadata_json, redact_sensitive(metadata))
        return 0 if summary is None or summary.get("status") == "completed" else 1

    args.initial_program.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(args.initial_program))
    metadata["status"] = "running"
    write_json(args.metadata_json, redact_sensitive(metadata))
    try:
        if args.algorithm == "gepa":
            optimized = optimize_with_gepa(
                program, trainset, valset=valset, reflection_lm=optimizer_lm,
                config=optimizer_config,
            )
        elif args.algorithm == "copro":
            optimized = optimize_with_copro(
                program, trainset, prompt_lm=optimizer_lm, config=optimizer_config,
                checkpoint_output_dir=args.output_dir,
            )
        else:
            optimized = optimize_with_miprov2(
                program, trainset, valset=valset, prompt_lm=optimizer_lm,
                config=optimizer_config,
            )
        artifacts = write_prompt_artifacts(
            optimized, algorithm=args.algorithm, output_dir=args.output_dir,
            optimizer_log_dir=args.optimizer_log_dir,
        )
        if args.output_program != Path(artifacts["compiled_program"]):
            optimized.save(str(args.output_program))
            artifacts["compiled_program"] = str(args.output_program)
        if args.optimized_prompt_config_yaml:
            args.optimized_prompt_config_yaml.parent.mkdir(parents=True, exist_ok=True)
            Path(artifacts["optimized_prompt_config_yaml"]).replace(
                args.optimized_prompt_config_yaml
            )
            artifacts["optimized_prompt_config_yaml"] = str(args.optimized_prompt_config_yaml)
        if args.trajectory_jsonl:
            _save_trajectories(optimized, trainset, args.trajectory_jsonl)
            artifacts["trajectory_jsonl"] = str(args.trajectory_jsonl)
        final_eval = None if args.optimization_train_only else _run_final_eval(args, task, optimized)
    except BaseException as exc:
        metadata.update(status="interrupted_or_failed", error_type=type(exc).__name__,
                        finished_at=datetime.now(timezone.utc).isoformat())
        write_json(args.metadata_json, redact_sensitive(metadata))
        raise
    metadata.update(
        status="completed", artifacts=artifacts, final_eval=final_eval,
        caption_cache=(task["caption_cache"].snapshot() if task["caption_cache"] else None),
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(args.metadata_json, redact_sensitive(metadata))
    return 0 if final_eval is None or final_eval.get("status") == "completed" else 1


def main(argv=None) -> int:
    return run_with_completion_timer("Instruction optimization", lambda: _main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
