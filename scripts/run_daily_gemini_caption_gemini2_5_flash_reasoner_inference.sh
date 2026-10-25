#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 --input-jsonl DAILY_OMNI.jsonl" >&2; }
INPUT_JSONL=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --input-jsonl) INPUT_JSONL="${2:?Missing value for --input-jsonl}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage; echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "${INPUT_JSONL}" ]] || { usage; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
exec bash "${SCRIPT_DIR}/run_daily_gemini_caption_reasoner_inference.sh" \
  "${INPUT_JSONL}" \
  "${REPO_ROOT}/src/avqa_instr_opt/yamls/reasoner_configs/config_gemini2_5_flash_reasoner.yaml" \
  "${REPO_ROOT}/src/avqa_instr_opt/yamls/prompt_templates/config_daily_qa_prompt_repetition.yaml" \
  "${REPO_ROOT}/src/avqa_instr_opt/yamls/prompt_templates/config_qwen_caption_prompt.yaml" \
  "${REPO_ROOT}/outputs/gemini_caption_gemini2_5_flash_reasoner"
