#!/usr/bin/env bash
set -euo pipefail

# The published prompt was selected on Val125 after COPRO searches with
# optimizer seeds 2 and 18 at optimizer temperature 1.0. Evaluate that same
# prompt in three independent full-set runs with the official Gemini backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PERCEPTION_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/perception_configs/config_gemini2_5_flash_direct_nonthinking.yaml"
PROMPT_CONFIG="${REPO_ROOT}/src/avqa_instr_opt/yamls/optimized_template/gemini2_5_flash_copro.yaml"

for repeat in 1 2 3; do
  echo "Starting Gemini-2.5-Flash COPRO inference repeat ${repeat}."
  PERCEPTION_CONFIG="${PERCEPTION_CONFIG}" \
  PROMPT_CONFIG="${PROMPT_CONFIG}" \
  OUTPUT_DIR="${REPO_ROOT}/outputs/gemini2_5_flash_copro/repeat_${repeat}" \
  NUM_WORKERS="${NUM_WORKERS:-3}" BATCH_SIZE="${BATCH_SIZE:-3}" \
    bash "${SCRIPT_DIR}/run_daily_direct_inference.sh" "$@"
done
