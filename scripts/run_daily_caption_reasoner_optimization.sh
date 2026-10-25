#!/usr/bin/env bash
set -euo pipefail

# Shared implementation for one captioner-reasoner GEPA search run.

if [[ "$#" -ne 6 ]]; then
  echo "Internal usage: $0 TARGET RUN_DIR SEED TRAIN_JSONL VAL_JSONL NUM_WORKERS" >&2
  exit 2
fi

OPTIMIZE_TARGET="$1"
RUN_DIR="$2"
GEPA_SEED="$3"
TRAINSET_JSONL="$4"
VALSET_JSONL="$5"
NUM_WORKERS="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export GEPA_RUN_DIR="${RUN_DIR}"

ARGS=(
  --algorithm gepa
  --workflow caption_reasoner
  --caption-mode live
  --optimize-target "${OPTIMIZE_TARGET}"
  --optimizer-config "${REPO_ROOT}/src/avqa_instr_opt/yamls/optimizer_configs/gepa_2500.yaml"
  --optimizer-lm-config-yaml "${REPO_ROOT}/src/avqa_instr_opt/yamls/reasoner_configs/reasoner_elm_gpt4_1_none.yaml"
  --reasoner-config-yaml "${REPO_ROOT}/src/avqa_instr_opt/yamls/daily_125_elm_caption.yaml"
  --reasoner-prompt-yaml "${REPO_ROOT}/src/avqa_instr_opt/yamls/prompt_templates/config_daily_qa_prompt_repetition.yaml"
  --reasoner-template-name QA_PROMPT_TEMPLATE_CAPTION_OMNI
  --captioner-config-yaml "${REPO_ROOT}/src/avqa_instr_opt/yamls/captioner_configs/config_gemini2_5_flash_captioner.yaml"
  --caption-prompt-yaml "${REPO_ROOT}/src/avqa_instr_opt/yamls/prompt_templates/config_qwen_caption_prompt.yaml"
  --caption-template-name QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3
  --trainset-jsonl "${TRAINSET_JSONL}"
  --valset-jsonl "${VALSET_JSONL}"
  --output-dir "${RUN_DIR}"
  --gepa-seed "${GEPA_SEED}"
  --num-threads "${NUM_WORKERS}"
  --optimizer-num-threads "${NUM_WORKERS}"
  --optimization-train-only
)

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m avqa_instr_opt.optimize "${ARGS[@]}"
