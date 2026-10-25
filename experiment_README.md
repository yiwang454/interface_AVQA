# ICASSP 2027 reported experiments

This file presents the result and corresponding scripts in the ICASSP 2027 submission.
Daily-Omni contains 1,197 examples split into 125 training, 125 validation, and
947 held-out test examples. All values below are accuracy percentages. `Delta
Test` is relative to the corresponding unoptimized model.

## Setup and data preparation

Install the package and the optional dataset-preparation dependency:

```bash
pip install -e '.[data]'
cp .env.example .env
```

Fill in the provider credentials in `.env`, then export them in the shell used
to run the experiments. Qwen experiments read their OpenAI-compatible local
endpoint from `QWEN_BASE_URL`; `QWEN_API_KEY=EMPTY` is suitable for an
unauthenticated local server.

After downloading the official Daily-Omni metadata, videos, and extracted audio
files, build the full-dataset JSONL used by this repository with explicit input
and output paths:

```bash
set -a
source .env
set +a
python scripts/prepare_daily_omni.py \
  --qa-json /path/to/daily_omni/qa.json \
  --video-root /path/to/daily_omni/Videos \
  --output-jsonl /path/to/daily_omni/daily_omni_cuts_v3.jsonl
```

This command prepares the full benchmark only; it does not reconstruct the
paper's train/validation/test splits.

## Direct inference

| Model | Optimizer | Val (125) | Overall | Test (947) | Delta Test | Launcher |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-Omni-30B-A3B-Instruct | None | 72.80 | 72.18 | 71.81 | - | [`scripts/run_daily_qwen_inference.sh`](scripts/run_daily_qwen_inference.sh) |
| Qwen3-Omni-30B-A3B-Instruct | COPRO | 71.20 | 71.93 | 71.59 | -0.22 | [`scripts/run_daily_qwen3_omni_instruct_copro_inference.sh`](scripts/run_daily_qwen3_omni_instruct_copro_inference.sh) |
| Qwen3-Omni-30B-A3B-Instruct | GEPA | 75.20 | 72.01 | 71.49 | -0.32 | [`scripts/run_daily_qwen3_omni_instruct_gepa_inference.sh`](scripts/run_daily_qwen3_omni_instruct_gepa_inference.sh) |
| Qwen3-Omni-30B-A3B-Thinking | None | 69.60 | 69.76 | 69.06 | - | [`scripts/run_daily_qwen3_omni_thinking_inference.sh`](scripts/run_daily_qwen3_omni_thinking_inference.sh) |
| Qwen3-Omni-30B-A3B-Thinking | COPRO | 71.20 | 69.26 | 68.85 | -0.21 | [`scripts/run_daily_qwen3_omni_thinking_copro_inference.sh`](scripts/run_daily_qwen3_omni_thinking_copro_inference.sh) |
| Qwen3-Omni-30B-A3B-Thinking | GEPA | 72.80 | 69.92 | 70.01 | +0.95 | [`scripts/run_daily_qwen3_omni_thinking_gepa_inference.sh`](scripts/run_daily_qwen3_omni_thinking_gepa_inference.sh) |

| Model | Optimizer | Val (125) | Overall | Test (947) | Delta Test | Launcher |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Gemini-2.5-Flash | None | 77.60 | 77.03 | 75.71 | - | [`scripts/run_daily_gemini2_5_flash_direct_qa_baseline.sh`](scripts/run_daily_gemini2_5_flash_direct_qa_baseline.sh) |
| Gemini-2.5-Flash | COPRO | 72.80 | 78.11 | 78.25 | +2.54 | [`scripts/run_daily_gemini2_5_flash_direct_qa_copro_inference.sh`](scripts/run_daily_gemini2_5_flash_direct_qa_copro_inference.sh) |
| Gemini-2.5-Flash | GEPA | 79.20 | 79.62 | 79.83 | +4.12 | [`scripts/run_daily_gemini2_5_flash_direct_qa_gepa_inference.sh`](scripts/run_daily_gemini2_5_flash_direct_qa_gepa_inference.sh) |

The Gemini baseline, and Gemini COPRO and GEPA inference results are the runs with median held-out test
accuracy among three repetitive full-dataset inference runs, so as to mitigate the impact from closed-source model run-to-run variance. Qwen inference results are fully replicable by fixing the random seeds (and other relevant sampling parameters), so Qwen inference experiments were run only once.

## Captioner-reasoner interface

| System | Captioner | Reasoner | Optimized component | Val (125) | Overall | Test (947) | Launcher |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| Direct inference reference | - | Gemini-2.5-Flash | None | 77.60 | 77.03 | 75.71 | [`scripts/run_daily_gemini2_5_flash_direct_qa_baseline.sh`](scripts/run_daily_gemini2_5_flash_direct_qa_baseline.sh) |
| Captioner-reasoner | Gemini-2.5-Flash | Gemini-2.5-Flash | None | 68.00 | 69.17 | 69.27 | [inference](scripts/run_daily_gemini_caption_gemini2_5_flash_reasoner_inference.sh) |
| Captioner-reasoner | Gemini-2.5-Flash | GPT-4.1 | None | 68.00 | 69.50 | 70.01 | [inference](scripts/run_daily_gemini_caption_gpt4_1_reasoner_inference.sh) |
| Captioner-reasoner | Gemini-2.5-Flash | GPT-4.1 | GEPA on caption | 79.20 | 68.50 | 67.90 | [optimization](scripts/run_daily_gemini_caption_gepa_gpt4_1_reasoner_optimization.sh) / [inference](scripts/run_daily_gemini_caption_gepa_gpt4_1_reasoner_inference.sh) |
| Captioner-reasoner | Gemini-2.5-Flash | GPT-4.1 | GEPA on reasoner | 71.20 | 70.01 | 69.90 | [optimization](scripts/run_daily_gemini_caption_gpt4_1_reasoner_gepa_optimization.sh) / [inference](scripts/run_daily_gemini_caption_gpt4_1_reasoner_gepa_inference.sh) |

Each reported captioner-reasoner result comes from a single full-dataset
inference run. For the GEPA rows, the optimized prompt is selected on the
validation split and then evaluated once on the full dataset. Because a single
Overall/Test evaluation already showed a prominent degradation relative to
direct Gemini inference, repeated runs were not needed to support the paper's
captioner-reasoner observations.

## ReAct agent interface

| System | Optimization | Val (125) | Overall | Test (947) | Launcher |
| --- | --- | ---: | ---: | ---: | --- |
| Gemini-2.5-Flash direct reference | None | 77.60 | 77.03 | 75.71 | [`scripts/run_daily_gemini2_5_flash_direct_qa_baseline.sh`](scripts/run_daily_gemini2_5_flash_direct_qa_baseline.sh) |
| ReAct | None | 74.40 | 76.11 | 76.03 |  |
| ReAct | GEPA | 76.80 | 75.77 | 76.03 |  |

Each reported ReAct result likewise comes from a single full-dataset inference
run; no multi-run aggregate is reported.




