#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 --trainset-jsonl TRAIN125.jsonl --valset-jsonl VAL125.jsonl" >&2; }
TRAINSET_JSONL=""; VALSET_JSONL=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --trainset-jsonl) TRAINSET_JSONL="${2:?Missing value for --trainset-jsonl}"; shift 2 ;;
    --valset-jsonl) VALSET_JSONL="${2:?Missing value for --valset-jsonl}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${TRAINSET_JSONL}" && -n "${VALSET_JSONL}" ]] || { usage; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/outputs/gemini_caption_gpt4_1_reasoner_gepa_optimization"
RUNS=()

for seed in 42 2 18; do
  run_dir="${OUTPUT_ROOT}/runs/run_$(( ${#RUNS[@]} + 1 ))"
  RUNS+=("${run_dir}")
  bash "${SCRIPT_DIR}/run_daily_caption_reasoner_optimization.sh" \
    text_reasoner.user_instruction "${run_dir}" "${seed}" \
    "${TRAINSET_JSONL}" "${VALSET_JSONL}" "${NUM_WORKERS:-6}"
done
"${PYTHON_BIN:-python}" "${SCRIPT_DIR}/select_best_optimization_run.py" \
  --algorithm gepa --output-dir "${OUTPUT_ROOT}/selected" "${RUNS[@]}"
