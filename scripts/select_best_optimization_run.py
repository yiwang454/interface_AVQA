#!/usr/bin/env python3
"""Select the validation-best prompt from a completed optimization search."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result file: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    if not rows:
        raise ValueError(f"No result rows in {path}")
    return rows


def _copro_score(run_dir: Path) -> float:
    rows = _jsonl(run_dir / "validation" / "output_test.jsonl")
    values = [row.get("correct") for row in rows]
    if not all(isinstance(value, bool) for value in values):
        raise ValueError(f"COPRO validation rows lack boolean 'correct': {run_dir}")
    return sum(values) / len(values)


def _gepa_score(run_dir: Path) -> float:
    selected = [
        row
        for row in _jsonl(run_dir / "gepa_candidates.jsonl")
        if row.get("whether_selected_as_best") is True
    ]
    if len(selected) != 1 or not isinstance(selected[0].get("full_valset_score"), (int, float)):
        raise ValueError(f"Expected one scored GEPA validation-best candidate: {run_dir}")
    return float(selected[0]["full_valset_score"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", required=True, choices=("copro", "gepa"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    score = _copro_score if args.algorithm == "copro" else _gepa_score
    prompt_name = f"optimized_prompt_config_{args.algorithm}.yaml"
    results = []
    for run_dir in args.run_dirs:
        prompt_path = run_dir / prompt_name
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing optimized prompt: {prompt_path}")
        results.append({"run_dir": str(run_dir.resolve()), "validation_score": score(run_dir)})

    # max() keeps the first run in launcher order when validation scores tie.
    best = max(results, key=lambda result: result["validation_score"])
    best_run = Path(best["run_dir"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_prompt = args.output_dir / "optimized_prompt.yaml"
    shutil.copyfile(best_run / prompt_name, selected_prompt)
    record = {
        "algorithm": args.algorithm,
        "selection_metric": "validation_accuracy",
        "selected_run_dir": str(best_run),
        "selected_prompt": str(selected_prompt.resolve()),
        "validation_score": best["validation_score"],
        "runs": results,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
