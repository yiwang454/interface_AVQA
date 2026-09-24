# textual_interface_AVQA
Public repository for paper submission _DESIGNING TEXTUAL INTERFACES TO ENHANCE LLM REASONING ON AUDIO-VISUAL QUESTION ANSWERING_.

This file records implementation and evaluation details omitted from the main paper for space.

## Dataset split

Experiments use the **Daily-Omni** benchmark with 1,197 audio-visual multiple-choice QA examples.

We use a stratified split that preserves the benchmark's task-category distribution:

- **Train:** 125 examples, used by the automatic prompt optimization (APO) algorithms during prompt search.
- **Validation:** 125 examples, used to evaluate candidate prompts and select the final prompt.
- **Test:** 947 examples, held out from prompt search and used as the main evaluation split.

The paper additionally reports full-benchmark accuracy where applicable, but held-out test accuracy is treated as the primary measure of prompt-optimization effectiveness.

## Models and system configurations

### Direct OmniLLM inference

We evaluate:

- **Qwen3-Omni-30B-A3B-Instruct**
- **Qwen3-Omni-30B-A3B-Thinking**
- **Gemini-2.5-Flash**

Qwen3-Omni is served locally. Gemini-2.5-Flash is accessed through the Google API.

Qwen3-Omni model repository:
https://github.com/QwenLM/Qwen3-Omni

### Captioner-reasoner system

The reported captioner-reasoner experiments use:

- **Captioner:** Gemini-2.5-Flash
- **Reasoner:** GPT-4.1

The captioner receives the audio-video input and produces a textual caption. The reasoner receives only the caption, question, and answer options.

We initially tested Qwen3-Omni variants as captioners, but did not use them in the reported captioner-reasoner experiments because they showed brittle instruction-following behavior in this role, including repetitive and semantically uninformative captions. We therefore use Gemini-2.5-Flash as the OmniLLM perception model for the reported interface study.

### Agent-subagent system

The reported ReAct-style agent experiments use:

- **Primary text agent / reasoner:** GPT-4.1
- **Audio-visual subagents:** Gemini-2.5-Flash

The system begins with an audio-visual captioning call. The text agent can then issue targeted multimodal evidence-collection queries to an OmniLLM query-response subagent before producing the final answer.

As with the captioner-reasoner system, Qwen3-Omni variants were not used for the reported agentic experiments because their captioning behavior was brittle under the required prompting setup.

## Decoding and multimodal input settings

### Qwen3-Omni-Instruct

- Temperature: `0`
- Top-p: `1.0`
- Top-k: `0`
- Decoding seed: `1234`

### Qwen3-Omni-Thinking

- Temperature: `0.6`
- Top-p: `0.95`
- Top-k: `20`
- Decoding seed: `1234`

These settings follow the suggested Qwen3-Omni evaluation configuration:
https://github.com/QwenLM/Qwen3-Omni

### Gemini-2.5-Flash

- Temperature: `0`
- Top-p: `1.0`
- Top-k: `0`

### Shared OmniLLM settings

- Maximum generation length: **4096 tokens**
- Video sampling rate: **2 FPS**
- Maximum sampled frames per video: **128**

## Automatic prompt optimization

We evaluate two APO algorithms:

- **COPRO**
- **GEPA**

GPT-4.1 is used as the optimizer LLM for both methods.

### Matched optimization budget

To make the two APO approaches comparable, we use the same overall prompt-search budget:

- **COPRO:** 2,500 trials
- **GEPA:** 2,500 trials

The corresponding search settings are:

#### COPRO

- Breadth: `5`
- Depth: `4`
- Optimizer temperatures searched: `{1.0, 1.2}`
- Random seeds searched: `{2, 18}`

#### GEPA

- Reflection minibatch size: `16`
- Search budget: `2500`
- Optimization seeds searched: `{2, 18, 42}`

For each optimizer, candidate prompts generated during search are evaluated on the validation split. The final prompt is selected according to validation accuracy and is then evaluated on held-out data.

## Repetition and reporting protocol

### Qwen direct inference

Qwen evaluations use a fixed decoding seed (`1234`) in all experiments.

- Qwen3-Omni-Instruct uses deterministic temperature-0 decoding.
- The fixed seed also limits sampling variation for Qwen3-Omni-Thinking.

We therefore report **single-run Qwen evaluations**.

### Gemini direct inference

The unoptimized Gemini direct-inference baseline and the GEPA-optimized Gemini direct-inference evaluation are each repeated **three times**.

For these repeated Gemini direct-inference experiments, we report **mean and standard deviation** across the three runs.

### Captioner-reasoner and ReAct interface studies

Captioner-reasoner and ReAct experiments are reported from **single runs**. These systems are used primarily as interface case studies: their main purpose is to study how APO behaves when optimization operates through explicit intermediate textual interfaces, and subsequent analysis focuses on failure modes, reasoning traces, and changes in evidence acquisition rather than small run-to-run differences.

## Notes on reproducibility

The paper keeps only the experimental details needed to understand the comparison:

- dataset split and the role of each split,
- models used in each system,
- the reason Qwen3-Omni is not used in the reported captioner-reasoner and agentic systems,
- APO methods,
- matched optimization budget,
- validation-based prompt selection.

Lower-level decoding, video-sampling, optimizer-search, and repetition settings are documented here to keep the main paper concise and within the ICASSP page limit.

