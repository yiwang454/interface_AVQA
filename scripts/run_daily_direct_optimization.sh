#!/usr/bin/env bash
set -euo pipefail

# Shared implementation for one Direct prompt-optimization run. Public
# launchers define the paper search grid and invoke this helper once per run.

if [[ "$#" -ne 9 ]]; then
  echo "Internal usage: $0 ALGORITHM PERCEPTION_CONFIG OPTIMIZER_CONFIG RUN_DIR SEED TEMPERATURE TRAIN_JSONL VAL_JSONL NUM_WORKERS" >&2
  exit 2
fi

ALGORITHM="$1"
PERCEPTION_CONFIG="$2"
OPTIMIZER_CONFIG="$3"
RUN_DIR="$4"
SEED="$5"
TEMPERATURE="$6"
TRAINSET_JSONL="$7"
VALSET_JSONL="$8"
NUM_WORKERS="$9"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COMMON_ARGS=(
  --algorithm "${ALGORITHM}"
  --workflow direct
  --optimize-target direct_qwen.user_instruction
  --perception-config-yaml "${PERCEPTION_CONFIG}"
  --optimizer-config "${OPTIMIZER_CONFIG}"
  --optimizer-lm-config-yaml "${REPO_ROOT}/src/avqa_instr_opt/yamls/reasoner_configs/reasoner_elm_gpt4_1_none.yaml"
  --prompt-yaml "${REPO_ROOT}/src/avqa_instr_opt/yamls/daily_qa_single_mllm.yaml"
  --trainset-jsonl "${TRAINSET_JSONL}"
  --valset-jsonl "${VALSET_JSONL}"
  --output-dir "${RUN_DIR}"
  --optimizer-num-threads "${NUM_WORKERS}"
)

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

case "${ALGORITHM}" in
  copro)
    exec "${PYTHON_BIN}" -m avqa_instr_opt.optimize \
      "${COMMON_ARGS[@]}" \
      --optimizer-lm-seed "${SEED}" \
      --copro-init-temperature "${TEMPERATURE}" \
      --final-eval-input-jsonl "${VALSET_JSONL}" \
      --final-eval-output-dir "${RUN_DIR}/validation" \
      --final-eval-output-jsonl "${RUN_DIR}/validation/output_test.jsonl" \
      --final-eval-num-threads "${NUM_WORKERS}" \
      --expected-final-count 125
    ;;
  gepa)
    export GEPA_RUN_DIR="${RUN_DIR}"
    exec "${PYTHON_BIN}" -m avqa_instr_opt.optimize \
      "${COMMON_ARGS[@]}" \
      --gepa-seed "${SEED}" \
      --optimization-train-only
    ;;
  *)
    echo "Unsupported optimization algorithm: ${ALGORITHM}" >&2
    exit 2
    ;;
esac
