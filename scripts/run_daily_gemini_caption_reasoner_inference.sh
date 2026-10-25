#!/usr/bin/env bash
set -euo pipefail

# Shared implementation for one paper-reported Gemini captioner-reasoner run.

if [[ "$#" -ne 5 ]]; then
  echo "Internal usage: $0 INPUT_JSONL REASONER_CONFIG REASONER_PROMPT CAPTION_PROMPT OUTPUT_DIR" >&2
  exit 2
fi

INPUT_JSONL="$1"
REASONER_CONFIG="$2"
REASONER_PROMPT="$3"
CAPTION_PROMPT="$4"
OUTPUT_DIR="$5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

ARGS=(
  --caption-mode live
  --captioner-config-yaml "${REPO_ROOT}/src/avqa_instr_opt/yamls/captioner_configs/config_gemini2_5_flash_captioner.yaml"
  --caption-prompt-yaml "${CAPTION_PROMPT}"
  --caption-template-name QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3
  --reasoner-config-yaml "${REASONER_CONFIG}"
  --reasoner-prompt-yaml "${REASONER_PROMPT}"
  --reasoner-template-name QA_PROMPT_TEMPLATE_CAPTION_OMNI
  --input-jsonl "${INPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --expected-count 1197
  --inference-num-threads "${NUM_WORKERS:-1}"
  --inference-batch-size "${BATCH_SIZE:-${NUM_WORKERS:-1}}"
)

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m avqa_instr_opt.caption_inference "${ARGS[@]}"
