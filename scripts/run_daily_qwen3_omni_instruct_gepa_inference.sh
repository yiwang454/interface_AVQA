#!/usr/bin/env bash
set -euo pipefail

# Evaluate the prompt selected on Val125 after the three-seed GEPA search once.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PERCEPTION_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/perception_configs/config_localqwen_api_instruct_control.yaml"
PROMPT_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/optimized_template/qwen3_omni_instruct_gepa.yaml"

PERCEPTION_CONFIG="${PERCEPTION_CONFIG}" \
PROMPT_CONFIG="${PROMPT_CONFIG}" \
OUTPUT_DIR="${REPO_ROOT}/outputs/qwen3_omni_instruct_gepa" \
  exec bash "${SCRIPT_DIR}/run_daily_direct_inference.sh" "$@"
