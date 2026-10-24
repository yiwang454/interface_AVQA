"""Shared prompt-target adapters and optimizer implementations.

The runtime programs read prompt text from :mod:`avqa_instr_opt.prompt_config`
instead of invoking DSPy predictors.  ``PromptTargetProgram`` therefore exposes
one carrier predictor per optimizable prompt, while this module's GEPA adapter
evaluates the real external-model workflow and constructs target-aware reflection
records from its predictions.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import dspy
import yaml
from tqdm.auto import tqdm

from .clients.reliable import ReliableQwenExecutor
from .clients.qwen_omni import safe_exception_summary
from .data import json_safe
from .metrics import avqa_metric, normalize_option_letter, prediction_option_letter
from .prompt_config import prompt_config, prompt_overrides


OPTIMIZE_TARGETS = frozenset({
    "direct_qwen.user_instruction",
    "captioner.user_instruction",
    "text_reasoner.user_instruction",
})

_WORKFLOW_TARGETS = {
    ("direct", None): frozenset({"direct_qwen.user_instruction"}),
    ("caption_reasoner", "cached"): frozenset({"text_reasoner.user_instruction"}),
    ("caption_reasoner", "live"): frozenset({
        "captioner.user_instruction", "text_reasoner.user_instruction",
    }),
}


class PromptComponentSignature(dspy.Signature):
    """Carrier signature whose instructions hold an optimizer candidate."""

    prompt_context = dspy.InputField(desc="Description of the prompt component's runtime role.")
    prompt_text = dspy.OutputField(desc="The prompt component text.")


def target_keys(target_path: str) -> tuple[str, ...]:
    return tuple(part for part in target_path.split(".") if part)


def get_nested(mapping: dict[str, Any], target_path: str) -> Any:
    value: Any = mapping
    for key in target_keys(target_path):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(target_path)
        value = value[key]
    return value


def set_nested(mapping: dict[str, Any], target_path: str, value: str) -> None:
    keys = target_keys(target_path)
    current = mapping
    for key in keys[:-1]:
        child = current.setdefault(key, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {target_path}; {key} is not a mapping")
        current = child
    current[keys[-1]] = value


def validate_optimize_targets(
    target_paths: tuple[str, ...], *, workflow: str, caption_mode: str | None,
    algorithm: str,
) -> None:
    if not target_paths:
        raise ValueError("At least one optimize target is required")
    if len(set(target_paths)) != len(target_paths):
        raise ValueError(f"Duplicate optimize targets: {target_paths}")
    allowed = _WORKFLOW_TARGETS.get((workflow, caption_mode))
    if allowed is None:
        raise ValueError(f"Unsupported workflow/caption mode: {workflow}/{caption_mode}")
    invalid = sorted(set(target_paths) - allowed)
    if invalid:
        raise ValueError(
            f"Unsupported target(s) for {workflow}/{caption_mode}: {', '.join(invalid)}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )
    if algorithm in {"copro", "miprov2"} and len(target_paths) != 1:
        raise ValueError(f"{algorithm} supports exactly one optimize target in the first version")
    for path in target_paths:
        value = get_nested(prompt_config(), path)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Optimize target must resolve to non-empty text: {path}")


class PromptTargetProgram(dspy.Module):
    """Expose prompt-config strings as optimizer-visible carrier predictors."""

    def __init__(self, base_program: dspy.Module, target_paths: tuple[str, ...]):
        super().__init__()
        object.__setattr__(self, "_base_program", base_program)
        self.target_paths = target_paths
        self.component_names = tuple(f"prompt_component_{index}" for index in range(len(target_paths)))
        for target_path, name in zip(target_paths, self.component_names):
            signature = PromptComponentSignature.with_instructions(
                str(get_nested(prompt_config(), target_path)).strip()
            )
            setattr(self, name, dspy.Predict(signature))

    def component_name(self, target_path: str) -> str:
        return self.component_names[self.target_paths.index(target_path)]

    def candidate_text(self, target_path: str) -> str:
        component = getattr(self, self.component_name(target_path))
        return str(component.signature.instructions or "").strip()

    def forward(self, *args: Any, **kwargs: Any) -> dspy.Prediction:
        overrides = {
            target_keys(path): self.candidate_text(path) for path in self.target_paths
        }
        with prompt_overrides(overrides):
            return self._base_program(*args, **kwargs)

    def forward_from_caption(self, *args: Any, **kwargs: Any) -> dspy.Prediction:
        """Preserve the caption runner's shared cached-caption interface."""
        overrides = {
            target_keys(path): self.candidate_text(path) for path in self.target_paths
        }
        with prompt_overrides(overrides):
            method = getattr(self._base_program, "forward_from_caption")
            return method(*args, **kwargs)

    def named_predictors(self):
        return [(name, getattr(self, name)) for name in self.component_names]

    def predictors(self):
        return [getattr(self, name) for name in self.component_names]


class CaptionInputReasonerProgram(dspy.Module):
    """Use an example-provided cached caption and call only the text reasoner."""

    def __init__(self, base_program: Any):
        super().__init__()
        object.__setattr__(self, "_base_program", base_program)

    def forward(self, *, caption: str, question: str, options_json: str,
                video_id: str | None = None, question_id: str | None = None,
                **_: Any) -> dspy.Prediction:
        return self._base_program.forward_from_caption(
            caption=caption, question=question, options_json=options_json,
            video_id=video_id, question_id=question_id,
        )

    def forward_from_caption(self, **kwargs: Any) -> dspy.Prediction:
        return self._base_program.forward_from_caption(**kwargs)


def optimizer_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    return float(avqa_metric(example, prediction, trace))


def _scored_prediction_answer(prediction: Any) -> str:
    """Return exactly the option letter used by the optimization metric."""
    return prediction_option_letter(prediction)


def gepa_feedback_metric(
    example: Any, prediction: Any, trace: Any = None,
    pred_name: str | None = None, pred_trace: Any = None,
) -> dspy.Prediction:
    """Answer-only score with target-specific, workflow-safe feedback."""
    score = optimizer_metric(example, prediction)
    gold = normalize_option_letter(str(getattr(example, "answer", "") or ""))
    got = _scored_prediction_answer(prediction)
    target = str(pred_name or "")
    state = "correct" if score == 1.0 else "incorrect"
    prefix = f"Downstream AVQA answer is {state} (score={score:g}); gold={gold}, predicted={got or '<empty>'}."
    if target == "captioner.user_instruction":
        guidance = (
            "Revise only the generic caption instruction. Request factual audio-visual evidence, "
            "speech, timing and cross-modal relations useful for later QA; never answer the "
            "multiple-choice question or encode example-specific facts."
        )
    elif target == "text_reasoner.user_instruction":
        guidance = (
            "Revise only the caption-based QA instruction. Use the supplied caption, question and "
            "choices carefully and preserve the single option-letter output contract."
        )
    else:
        guidance = (
            "Revise only the direct multimodal QA instruction. Ground the decision in the supplied "
            "audio/video, question and choices and preserve the single option-letter output contract."
        )
    return dspy.Prediction(score=score, feedback=prefix + "\n" + guidance)


def _reflection_template(target_path: str) -> str:
    if target_path == "captioner.user_instruction":
        role = (
            "This is a caption-only instruction. Improve factual coverage of visible actions, "
            "audible events, clear speech, timing and audio-visual relationships. It must not "
            "choose an answer option."
        )
    elif target_path == "text_reasoner.user_instruction":
        role = (
            "This is a text-only reasoner instruction. It receives a fixed caption plus a question "
            "and choices, and must return one option letter."
        )
    else:
        role = (
            "This is a direct multimodal QA instruction. It receives audio/video plus a question "
            "and choices, and must return one option letter."
        )
    return f"""I provided an assistant with the following instruction:
```
<curr_instructions>
```

Below are examples containing model outputs and downstream answer feedback:
```
<inputs_outputs_feedback>
```

{role}

Infer reusable improvements for unseen examples. Do not copy any example-specific answer,
option, caption, timestamp, person, object, scene, event, or wording into the instruction.
Keep all runtime inputs and output-format requirements intact.

Provide only the new instruction within a single ``` block."""


_QWEN_LENGTH_FINISH_REASONS = frozenset({
    "length", "max_token", "max_tokens", "max_output_tokens", "token_limit",
})
_QWEN_OUTPUT_TOKEN_USAGE_KEYS = ("completion_tokens", "output_tokens")


def _mapping_value(value: Any) -> dict[str, Any]:
    """Return provider metadata as a mapping without breaking GEPA diagnostics."""
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
        except Exception:
            return {}
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _qwen_call_hit_max_tokens(token_usage: Any) -> bool:
    """Detect a Qwen caption generation that stopped at its output cap."""
    usage = _mapping_value(token_usage)
    attempt_usages = [usage]
    attempts = usage.get("qwen_response_attempts")
    if isinstance(attempts, list):
        attempt_usages.extend(_mapping_value(attempt) for attempt in attempts)

    for attempt_usage in attempt_usages:
        finish_reason = str(
            attempt_usage.get("qwen_finish_reason")
            or attempt_usage.get("finish_reason")
            or ""
        ).strip().lower()
        if finish_reason in _QWEN_LENGTH_FINISH_REASONS:
            return True

        max_tokens = _positive_int(
            attempt_usage.get("qwen_max_tokens", attempt_usage.get("max_tokens"))
        )
        if max_tokens is None:
            continue
        for token_key in _QWEN_OUTPUT_TOKEN_USAGE_KEYS:
            completion_tokens = _positive_int(attempt_usage.get(token_key))
            if completion_tokens is not None and completion_tokens >= max_tokens:
                return True
    return False


def _qwen_captioner_max_token_stats(predictions: list[Any]) -> dict[str, Any]:
    """Aggregate output-cap hits for one-shot Qwen captions in a GEPA batch."""
    captioner_samples = 0
    hit_samples = 0
    max_tokens_values: set[int] = set()
    for prediction in predictions:
        if not hasattr(prediction, "caption_token_usage"):
            continue
        captioner_samples += 1
        token_usage = _mapping_value(getattr(prediction, "caption_token_usage", None))
        max_tokens = _positive_int(token_usage.get("qwen_max_tokens"))
        if max_tokens is not None:
            max_tokens_values.add(max_tokens)
        hit_samples += int(_qwen_call_hit_max_tokens(token_usage))
    return {
        "captioner_samples": captioner_samples,
        "hit_samples": hit_samples,
        "max_tokens_values": tuple(sorted(max_tokens_values)),
    }


def prediction_has_empty_qwen(prediction: Any) -> bool:
    if hasattr(prediction, "caption"):
        # Cached-caption reasoner predictions always contain a supplied caption; an
        # empty reasoner output is not a Qwen transport/layout failure.
        return not str(getattr(prediction, "caption", "") or "").strip()
    return not str(getattr(prediction, "response_text", "") or "").strip()


def _runtime_task_backend(program: Any) -> str:
    """Find a nested runtime backend without importing program classes."""
    current = program
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        backend = getattr(current, "task_backend", None) or getattr(
            current, "captioner_backend", None
        )
        if backend:
            return str(backend)
        current = getattr(current, "_base_program", None)
    # Direct programs and historical test doubles are Qwen-backed.
    return "qwen"


class PromptTargetGEPAAdapter:
    """GEPA adapter for carrier predictors backed by external runtime programs."""

    def __init__(self, *, student_module: PromptTargetProgram, reflection_lm: Any,
                 num_threads: int | None = None, failure_score: float = 0.0,
                 qwen_executor: ReliableQwenExecutor | None = None,
                 use_qwen_reliability: bool | None = None):
        self.student = student_module
        self.component_names = student_module.component_names
        self.reflection_lm = reflection_lm
        self.num_threads = num_threads or 1
        self.failure_score = failure_score
        self.qwen_executor = qwen_executor or ReliableQwenExecutor()
        self.use_qwen_reliability = (
            _runtime_task_backend(student_module) == "qwen"
            if use_qwen_reliability is None
            else use_qwen_reliability
        )

    def build_program(self, candidate: dict[str, str]) -> PromptTargetProgram:
        program = self.student.deepcopy()
        for name in self.component_names:
            if name in candidate:
                component = getattr(program, name)
                component.signature = component.signature.with_instructions(candidate[name])
        return program

    def _evaluate_one(self, program: dspy.Module, example: dspy.Example):
        try:
            prediction = program(**example.inputs())
            return prediction, optimizer_metric(example, prediction)
        except Exception as exc:
            print(
                "[error] GEPA task evaluation failed for "
                f"question_id={getattr(example, 'question_id', '<unknown>')}: "
                f"{safe_exception_summary(exc)}",
                flush=True,
            )
            prediction = dspy.Prediction(
                answer="", response_text="", caption="", error_type=type(exc).__name__
            )
            return prediction, self.failure_score

    def evaluate(self, batch: list[dspy.Example], candidate: dict[str, str],
                 capture_traces: bool = False):
        from gepa.core.adapter import EvaluationBatch

        program = self.build_program(candidate)

        batch_index = getattr(self, "_progress_batch_index", 0) + 1
        self._progress_batch_index = batch_index
        if capture_traces:
            phase = "GEPA training/reflection"
        elif not getattr(self, "_initial_validation_started", False):
            phase = "GEPA initial validation"
            self._initial_validation_started = True
        else:
            phase = "GEPA candidate validation"

        completed_examples: set[int] = set()
        progress_lock = threading.Lock()
        progress = tqdm(
            total=len(batch),
            desc=f"{phase} batch={batch_index}",
            unit="sample",
            dynamic_ncols=True,
            leave=True,
        )

        def run_item(example):
            try:
                return self._evaluate_one(program, example)
            finally:
                # Reliable-Qwen fallback may invoke the same example again.
                # Count each logical sample once rather than counting retries.
                with progress_lock:
                    example_key = id(example)
                    if example_key not in completed_examples:
                        completed_examples.add(example_key)
                        progress.update(1)

        try:
            if self.use_qwen_reliability:
                result = self.qwen_executor.run_batch(
                    batch, run_item=run_item,
                    has_empty_qwen_response=lambda item: prediction_has_empty_qwen(item[0]),
                    max_workers=self.num_threads,
                )
                result_items = result.results
            elif self.num_threads == 1:
                result_items = [run_item(example) for example in batch]
            else:
                result_items = [None] * len(batch)
                with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                    futures = {
                        pool.submit(run_item, example): index
                        for index, example in enumerate(batch)
                    }
                    for future in as_completed(futures):
                        result_items[futures[future]] = future.result()
        finally:
            progress.close()
        predictions = [item[0] for item in result_items]
        scores = [float(item[1]) for item in result_items]
        self._report_captioner_max_token_hits(
            predictions, capture_traces=capture_traces
        )
        trajectories = None
        if capture_traces:
            trajectories = [
                {"example": example, "prediction": prediction, "score": score, "trace": []}
                for example, prediction, score in zip(batch, predictions, scores)
            ]
        return EvaluationBatch(outputs=predictions, scores=scores, trajectories=trajectories)

    def _target_path(self, component_name: str) -> str:
        return self.student.target_paths[self.component_names.index(component_name)]

    def _report_captioner_max_token_hits(
        self, predictions: list[Any], *, capture_traces: bool,
    ) -> None:
        """Print one output-cap summary after each caption-optimization batch."""
        if "captioner.user_instruction" not in self.student.target_paths:
            return
        if not self.use_qwen_reliability:
            return
        stats = _qwen_captioner_max_token_stats(predictions)
        phase = "reflection_trace" if capture_traces else "evaluation"
        observed = ",".join(map(str, stats["max_tokens_values"])) or "unknown"
        print(
            "[GEPA captioner max-tokens] "
            f"batch={getattr(self, '_progress_batch_index', 0)} phase={phase} "
            f"batch_samples={len(predictions)} "
            f"qwen_captioner_samples={stats['captioner_samples']} "
            f"hit_max_tokens_samples={stats['hit_samples']} "
            f"observed_max_tokens={observed}",
            flush=True,
        )

    def make_reflective_dataset(self, candidate: dict[str, str], eval_batch: Any,
                                components_to_update: list[str]):
        if not eval_batch.trajectories:
            raise RuntimeError("GEPA produced no reflection trajectories")
        datasets: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            target = self._target_path(component)
            records = []
            for item in eval_batch.trajectories:
                example = item["example"]
                prediction = item["prediction"]
                feedback = gepa_feedback_metric(example, prediction, pred_name=target)
                records.append({
                    "Inputs": {
                        "Optimization target": target,
                        "Question": str(getattr(example, "question", "") or ""),
                        "Options": _options_for_reflection(getattr(example, "options_json", "")),
                        "Gold answer": normalize_option_letter(str(getattr(example, "answer", ""))),
                    },
                    "Generated Outputs": {
                        "Caption": str(getattr(prediction, "caption", "") or ""),
                        "Reasoner response": str(getattr(prediction, "response_text", "") or ""),
                        "Thinking": str(getattr(prediction, "thinking", "") or ""),
                        "Predicted answer": _scored_prediction_answer(prediction),
                        "Score": item["score"],
                        "Error type": getattr(prediction, "error_type", None),
                    },
                    "Feedback": feedback.feedback,
                })
            datasets[component] = records
        return datasets

    def stripped_lm_call(self, prompt: str) -> list[str]:
        try:
            outputs = self.reflection_lm(prompt)
        except Exception as exc:
            print(
                "[error] GEPA reflection LM call failed: "
                f"{safe_exception_summary(exc)}",
                flush=True,
            )
            print("".join(traceback.format_tb(exc.__traceback__)), end="", flush=True)
            raise
        return [str(output.get("text", "") if isinstance(output, dict) else output)
                for output in outputs]

    def propose_new_texts(self, candidate: dict[str, str], reflective_dataset: dict,
                          components_to_update: list[str]):
        from dspy.teleprompt.gepa.gepa_utils import InstructionProposalSignature

        proposed = {}
        with dspy.context(lm=self.reflection_lm):
            for component in components_to_update:
                target = self._target_path(component)
                proposed[component] = InstructionProposalSignature.run(
                    lm=lambda prompt: self.stripped_lm_call(prompt)[0],
                    input_dict={
                        "current_instruction_doc": candidate[component],
                        "dataset_with_feedback": reflective_dataset[component],
                        "prompt_template": _reflection_template(target),
                    },
                )["new_instruction"]
        return proposed


def _options_for_reflection(value: Any) -> Any:
    try:
        return json.loads(value) if isinstance(value, str) else json_safe(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return str(value)


def _filtered_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(callable_obj)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD
           for parameter in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def load_optimizer_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) if path.suffix.lower() in {".yaml", ".yml"} else json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Optimizer config must be a mapping: {path}")
    return payload


_OPTIMIZER_CONFIG_FIELDS = {
    "gepa": {
        "auto", "max_full_evals", "max_metric_calls", "reflection_minibatch_size",
        "candidate_selection_strategy", "skip_perfect_score", "use_merge",
        "max_merge_invocations", "num_threads", "failure_score", "seed", "log_dir",
        "reflective_dataset_save_interval", "track_stats", "track_best_outputs",
    },
    "copro": {"breadth", "depth", "init_temperature", "track_stats", "num_threads"},
    "miprov2": {
        "auto", "num_candidates", "num_trials", "max_bootstrapped_demos",
        "max_labeled_demos", "seed", "init_temperature", "num_threads", "max_errors",
        "minibatch", "minibatch_size", "minibatch_full_eval_steps",
        "view_data_batch_size", "track_stats", "log_dir",
    },
}


def validate_optimizer_config(config: dict[str, Any], algorithm: str) -> None:
    allowed = _OPTIMIZER_CONFIG_FIELDS[algorithm] | {"run"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown {algorithm} optimizer config key(s): {', '.join(unknown)}"
        )
    if algorithm == "miprov2" and (
        int(config.get("max_bootstrapped_demos", 0)) != 0
        or int(config.get("max_labeled_demos", 0)) != 0
    ):
        raise ValueError("MIPROv2 is instruction-only; both demo counts must be 0")


def optimize_with_gepa(program: PromptTargetProgram, trainset: list[dspy.Example], *,
                       valset: list[dspy.Example], reflection_lm: Any,
                       config: dict[str, Any]) -> PromptTargetProgram:
    from gepa import optimize

    settings = {key: value for key, value in config.items() if key != "run"}
    max_metric_calls = settings.get("max_metric_calls")
    max_full_evals = settings.get("max_full_evals", 6)
    if max_metric_calls is None and max_full_evals is not None:
        max_metric_calls = int(max_full_evals) * (len(trainset) + len(valset))
    adapter = PromptTargetGEPAAdapter(
        student_module=program, reflection_lm=reflection_lm,
        num_threads=settings.get("num_threads"),
        failure_score=float(settings.get("failure_score", 0.0)),
    )
    seed_candidate = {
        name: program.candidate_text(path)
        for path, name in zip(program.target_paths, program.component_names)
    }
    result = optimize(
        seed_candidate=seed_candidate, trainset=trainset, valset=valset,
        adapter=adapter, reflection_lm=lambda prompt: adapter.stripped_lm_call(prompt)[0],
        candidate_selection_strategy=settings.get("candidate_selection_strategy", "pareto"),
        skip_perfect_score=bool(settings.get("skip_perfect_score", True)),
        reflection_minibatch_size=int(settings.get("reflection_minibatch_size", 3)),
        module_selector="round_robin",
        use_merge=bool(settings.get("use_merge", True)),
        max_merge_invocations=int(settings.get("max_merge_invocations", 5)),
        max_metric_calls=max_metric_calls,
        run_dir=settings.get("log_dir"),
        track_best_outputs=bool(settings.get("track_best_outputs", False)),
        display_progress_bar=True, raise_on_exception=True,
        seed=int(settings.get("seed", 0)),
    )
    optimized = adapter.build_program(result.best_candidate)
    optimized.prompt_target_gepa_result = result
    optimized.prompt_target_candidate_rows = [
        {
            "candidate_index": index,
            "target_path": target,
            "prompt_text": candidate.get(component),
            "full_valset_score": (
                result.val_aggregate_scores[index]
                if index < len(result.val_aggregate_scores) else None
            ),
            "parents": result.parents[index] if index < len(result.parents) else None,
            "discovery_eval_count": (
                result.discovery_eval_counts[index]
                if index < len(result.discovery_eval_counts) else None
            ),
            "whether_selected_as_best": index == result.best_idx,
        }
        for index, candidate in enumerate(result.candidates)
        for target, component in zip(program.target_paths, program.component_names)
    ]
    return optimized


class _COPROCandidateCheckpoint:
    """Durably save a COPRO candidate as soon as its train evaluation returns.

    DSPy's COPRO only exposes its candidate list after ``compile`` finishes.  A
    long-running search can therefore lose every generated instruction when its
    parent process is interrupted.  This recorder is deliberately append-only:
    every completed evaluation gets its own program snapshot and JSONL row.
    """

    def __init__(self, output_dir: Path, *, breadth: int) -> None:
        self.output_dir = output_dir
        self.breadth = max(1, int(breadth))
        self.candidates_dir = output_dir / "optimizer_logs" / "checkpoint_candidates"
        self.instructions_dir = output_dir / "optimizer_logs" / "instructions"
        self.rows_path = output_dir / "copro_candidates_live.jsonl"
        self.state_path = output_dir / "copro_checkpoint_state.json"
        self.count = 0
        self.best_score: float | None = None
        self.best_index: int | None = None
        self._lock = threading.Lock()
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.instructions_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _atomic_text(path: Path, value: str) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    @staticmethod
    def _durable_program_save(program: PromptTargetProgram, path: Path) -> None:
        tmp = path.with_name(f".{path.name}.tmp.json")
        program.save(str(tmp))
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def record(self, candidate: PromptTargetProgram, score: float) -> None:
        """Record one fully evaluated candidate before COPRO moves onward."""
        with self._lock:
            index = self.count
            self.count += 1
            candidate_path = self.candidates_dir / f"candidate_{index:04d}.json"
            self._durable_program_save(candidate, candidate_path)

            rows = []
            for target in candidate.target_paths:
                instruction_path = self.instructions_dir / (
                    f"checkpoint_candidate_{index:04d}_{target.replace('.', '_')}.txt"
                )
                prompt_text = candidate.candidate_text(target)
                self._atomic_text(instruction_path, prompt_text)
                rows.append({
                    "candidate_index": index,
                    "evaluation_index": index + 1,
                    "iteration": index // self.breadth,
                    "search_round": index // self.breadth + 1,
                    "score": float(score),
                    "evaluation_scope": "train125_optimization",
                    "target_path": target,
                    "prompt_text": prompt_text,
                    "instruction_path": str(instruction_path),
                    "program_path": str(candidate_path),
                })
            with self.rows_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            if self.best_score is None or score > self.best_score:
                self.best_score = float(score)
                self.best_index = index
                best_path = self.output_dir / "best_observed_copro_program.json"
                self._durable_program_save(candidate, best_path)
            self._atomic_json(self.state_path, {
                "completed_evaluations": self.count,
                "best_observed_score": self.best_score,
                "best_observed_candidate_index": self.best_index,
                "candidate_rows": str(self.rows_path),
                "note": "Scores are COPRO Train125 optimization-time scores, not validation results.",
            })


def optimize_with_copro(program: PromptTargetProgram, trainset: list[dspy.Example], *,
                        prompt_lm: Any, config: dict[str, Any],
                        checkpoint_output_dir: Path | None = None) -> dspy.Module:
    from dspy.teleprompt import COPRO

    kwargs = {key: config[key] for key in ("breadth", "depth", "init_temperature", "track_stats")
              if key in config}
    optimizer = COPRO(prompt_model=prompt_lm, metric=optimizer_metric, **kwargs)
    eval_kwargs = {"num_threads": config.get("num_threads", 1)}
    if checkpoint_output_dir is None:
        return optimizer.compile(student=program, trainset=trainset, eval_kwargs=eval_kwargs)

    # COPRO imports Evaluate at module scope.  Wrapping that class lets us
    # checkpoint immediately after each candidate's full evaluation without
    # maintaining a fork of DSPy's optimizer implementation.
    import dspy.teleprompt.copro_optimizer as copro_optimizer

    checkpoint = _COPROCandidateCheckpoint(
        checkpoint_output_dir, breadth=int(config.get("breadth", 10)),
    )
    original_evaluate = copro_optimizer.Evaluate

    class CheckpointingEvaluate:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._evaluate = original_evaluate(*args, **kwargs)

        def __call__(self, candidate: PromptTargetProgram, *args: Any, **kwargs: Any) -> Any:
            result = self._evaluate(candidate, *args, **kwargs)
            checkpoint.record(candidate, result.score)
            return result

    copro_optimizer.Evaluate = CheckpointingEvaluate
    try:
        return optimizer.compile(student=program, trainset=trainset, eval_kwargs=eval_kwargs)
    finally:
        copro_optimizer.Evaluate = original_evaluate


def optimize_with_miprov2(program: PromptTargetProgram, trainset: list[dspy.Example], *,
                          valset: list[dspy.Example], prompt_lm: Any,
                          config: dict[str, Any]) -> dspy.Module:
    try:
        import optuna  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "MIPROv2 requires optuna; install the project optimizer dependency first"
        ) from exc
    if int(config.get("max_bootstrapped_demos", 0)) != 0 or int(
        config.get("max_labeled_demos", 0)
    ) != 0:
        raise ValueError("This repository supports MIPROv2 instruction-only mode; demo counts must be 0")
    auto = None if str(config.get("auto", "none")).lower() == "none" else config.get("auto")
    init_kwargs = {
        "metric": optimizer_metric, "prompt_model": prompt_lm,
        "max_bootstrapped_demos": 0, "max_labeled_demos": 0,
        "auto": auto,
        "num_candidates": None if auto is not None else config.get("num_candidates", 6),
        "num_threads": config.get("num_threads"), "max_errors": config.get("max_errors"),
        "seed": config.get("seed", 9), "init_temperature": config.get("init_temperature", 1.0),
        "track_stats": config.get("track_stats", True), "log_dir": config.get("log_dir"),
    }
    optimizer = dspy.MIPROv2(**_filtered_kwargs(dspy.MIPROv2.__init__, init_kwargs))
    compile_kwargs = {
        "valset": valset, "num_trials": None if auto is not None else config.get("num_trials", 10),
        "max_bootstrapped_demos": 0, "max_labeled_demos": 0,
        "seed": config.get("seed", 9), "minibatch": config.get("minibatch", True),
        "minibatch_size": config.get("minibatch_size", 16),
        "minibatch_full_eval_steps": config.get("minibatch_full_eval_steps", 5),
        "view_data_batch_size": config.get("view_data_batch_size", 8),
        "requires_permission_to_run": False,
    }
    return optimizer.compile(
        student=program, trainset=trainset,
        **_filtered_kwargs(optimizer.compile, compile_kwargs),
    )


def optimized_prompt_config(program: PromptTargetProgram) -> dict[str, Any]:
    config = copy.deepcopy(prompt_config())
    for target in program.target_paths:
        set_nested(config, target, program.candidate_text(target))
    return config


def _candidate_rows(program: PromptTargetProgram) -> list[dict[str, Any]]:
    rows = list(getattr(program, "prompt_target_candidate_rows", []) or [])
    if rows:
        return rows
    rows = []
    index = 0
    for attribute, scope in (("candidate_programs", "full"), ("mb_candidate_programs", "minibatch")):
        for candidate in list(getattr(program, attribute, []) or []):
            if not isinstance(candidate, dict) or candidate.get("program") is None:
                continue
            candidate_program = candidate["program"]
            for target in program.target_paths:
                rows.append({
                    "candidate_index": index, "target_path": target,
                    "prompt_text": candidate_program.candidate_text(target),
                    "score": candidate.get("score"), "evaluation_scope": scope,
                    "iteration": candidate.get("depth", candidate.get("trial")),
                    "prefix": candidate.get("prefix"),
                })
            index += 1
    return rows or [{
        "candidate_index": 0, "target_path": target,
        "prompt_text": program.candidate_text(target), "whether_selected_as_best": True,
    } for target in program.target_paths]


def write_prompt_artifacts(program: PromptTargetProgram, *, algorithm: str,
                           output_dir: Path, optimizer_log_dir: Path | None = None) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled = output_dir / f"compiled_{algorithm}.json"
    program.save(str(compiled))
    optimized_yaml = output_dir / f"optimized_prompt_config_{algorithm}.yaml"
    optimized_yaml.write_text(
        yaml.safe_dump(optimized_prompt_config(program), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    rows = _candidate_rows(program)
    candidates = output_dir / f"{algorithm}_candidates.jsonl"
    with candidates.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
    log_dir = optimizer_log_dir or output_dir / "optimizer_logs"
    instructions_dir = log_dir / "instructions"
    instructions_dir.mkdir(parents=True, exist_ok=True)
    normalized = log_dir / "candidates.jsonl"
    with normalized.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            instruction_path = instructions_dir / (
                f"candidate_{int(row.get('candidate_index', row_index)):04d}_"
                f"{str(row.get('target_path', 'prompt')).replace('.', '_')}.txt"
            )
            instruction_path.write_text(str(row.get("prompt_text") or ""), encoding="utf-8")
            enriched = {**row, "instruction_path": str(instruction_path)}
            handle.write(json.dumps(json_safe(enriched), ensure_ascii=False) + "\n")
    return {
        "compiled_program": str(compiled),
        "optimized_prompt_config_yaml": str(optimized_yaml),
        "candidates_jsonl": str(candidates),
        "normalized_candidates_jsonl": str(normalized),
        "optimizer_log_dir": str(log_dir),
    }
