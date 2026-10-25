#!/usr/bin/env bash
set -euo pipefail

# Evaluate the prompt selected on Val125 after the three-seed GEPA search in
# three independent full-set runs using the official Gemini backend. All
# repeats use the same prompt and model configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PERCEPTION_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/perception_configs/config_gemini2_5_flash_direct.yaml"
PROMPT_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/optimized_template/gemini2_5_flash_gepa.yaml"

for repeat in 1 2 3; do
  echo "Starting Gemini-2.5-Flash GEPA inference repeat ${repeat}."
  PERCEPTION_CONFIG="${PERCEPTION_CONFIG}" \
  PROMPT_CONFIG="${PROMPT_CONFIG}" \
  OUTPUT_DIR="${REPO_ROOT}/outputs/gemini2_5_flash_gepa/repeat_${repeat}" \
  NUM_WORKERS="${NUM_WORKERS:-3}" BATCH_SIZE="${BATCH_SIZE:-3}" \
    bash "${SCRIPT_DIR}/run_daily_direct_inference.sh" "$@"
done
