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
OUTPUT_ROOT="${REPO_ROOT}/outputs/daily_qwen3_omni_thinking_copro_optimization"
PERCEPTION_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/perception_configs/config_localqwen_api_think.yaml"
CONFIG_1P0="${REPO_ROOT}/src/avqa_instr_opt/yamls/optimizer_configs/copro_qwen3_omni_thinking_2500_temp1p0.yaml"
CONFIG_1P2="${REPO_ROOT}/src/avqa_instr_opt/yamls/optimizer_configs/copro_qwen3_omni_thinking_2500_temp1p2.yaml"
RUNNER="${SCRIPT_DIR}/run_daily_direct_optimization.sh"
RUNS=()

run_one() {
  local run_name="$1" seed="$2" temperature="$3" optimizer_config="$4"
  local run_dir="${OUTPUT_ROOT}/runs/${run_name}"
  RUNS+=("${run_dir}")
  bash "${RUNNER}" copro "${PERCEPTION_CONFIG}" "${optimizer_config}" "${run_dir}" \
    "${seed}" "${temperature}" "${TRAINSET_JSONL}" "${VALSET_JSONL}" "${NUM_WORKERS:-4}"
}

run_one run_01 18 1.0 "${CONFIG_1P0}"
run_one run_02 2 1.0 "${CONFIG_1P0}"
run_one run_03 18 1.2 "${CONFIG_1P2}"
run_one run_04 2 1.2 "${CONFIG_1P2}"
"${PYTHON_BIN:-python}" "${SCRIPT_DIR}/select_best_optimization_run.py" \
  --algorithm copro --output-dir "${OUTPUT_ROOT}/selected" "${RUNS[@]}"
