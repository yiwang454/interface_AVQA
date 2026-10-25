#!/usr/bin/env bash
set -euo pipefail

# Shared implementation for the paper's fixed full-dataset Direct inference
# entries. Model config, prompt config, and output directory are selected by
# the public launcher that invokes this helper.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT_DIR}"
exec "${PYTHON_BIN}" -m avqa_instr_opt.inference \
  --perception-config-yaml "${PERCEPTION_CONFIG}" \
  --prompt-yaml "${PROMPT_CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-count 1197 \
  --inference-num-threads "${NUM_WORKERS:-4}" \
  --inference-batch-size "${BATCH_SIZE:-${NUM_WORKERS:-4}}" \
  "$@"
