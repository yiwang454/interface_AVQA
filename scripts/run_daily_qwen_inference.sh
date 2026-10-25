#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PERCEPTION_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/perception_configs/config_localqwen_api_instruct_control.yaml" \
PROMPT_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/daily_qa_single_mllm.yaml" \
OUTPUT_DIR="${REPO_ROOT}/outputs/qwen3_omni_instruct_baseline" \
  exec bash "${SCRIPT_DIR}/run_daily_direct_inference.sh" "$@"
