"""avqa_instr_opt: non-ReAct direct-MLLM and caption-then-text AVQA instruction optimization.

Stage 1 (this package as it stands): MLLM/LLM clients (``clients``), prompt
config (``prompt_config``), the exact-match metric (``metrics``), I/O helpers
(``data``), model/server profile resolution (``config``), and the two
workflow programs (``programs``).

Stage 2 (not yet implemented): a ``PromptTargetProgram``-style wrapper over
these programs exposing ``direct_qwen.user_instruction`` and
``captioner.user_instruction`` to COPRO/MIPROv2, with GPT as the optimizer's
prompt/reflection LM (``clients.text_lm.build_text_lm("prompt_lm")``).

Stage 1 also includes ``inference`` and its Daily-Omni launcher for fixed-prompt
QA. Optimizer launchers will be added with their respective COPRO/MIPROv2
stages, followed by formal train/val experiments.
"""

from __future__ import annotations

from . import clients, config, data, metrics, prompt_config, programs

__all__ = ["clients", "config", "data", "metrics", "prompt_config", "programs"]
