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
OUTPUT_ROOT="${REPO_ROOT}/outputs/daily_gemini2_5_flash_gepa_optimization"
PERCEPTION_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/perception_configs/config_gemini2_5_flash_direct.yaml"
OPTIMIZER_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/optimizer_configs/gepa_2500.yaml"
RUNNER="${SCRIPT_DIR}/run_daily_direct_optimization.sh"
RUNS=()

for seed in 18 42 2; do
  run_dir="${OUTPUT_ROOT}/runs/run_$(( ${#RUNS[@]} + 1 ))"
  RUNS+=("${run_dir}")
  bash "${RUNNER}" gepa "${PERCEPTION_CONFIG}" "${OPTIMIZER_CONFIG}" "${run_dir}" \
    "${seed}" unused "${TRAINSET_JSONL}" "${VALSET_JSONL}" "${NUM_WORKERS:-3}"
done
"${PYTHON_BIN:-python}" "${SCRIPT_DIR}/select_best_optimization_run.py" \
  --algorithm gepa --output-dir "${OUTPUT_ROOT}/selected" "${RUNS[@]}"
