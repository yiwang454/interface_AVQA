#!/usr/bin/env bash
set -euo pipefail

# The published prompt was selected on Val125 after searching two optimizer
# seeds and two optimizer temperatures. Qwen uses its fixed decoding seed and
# is evaluated once.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PERCEPTION_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/perception_configs/config_localqwen_api_think.yaml" \
PROMPT_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/optimized_template/qwen3_omni_thinking_copro.yaml" \
OUTPUT_DIR="${REPO_ROOT}/outputs/qwen3_omni_thinking_copro" \
  exec bash "${SCRIPT_DIR}/run_daily_direct_inference.sh" "$@"
