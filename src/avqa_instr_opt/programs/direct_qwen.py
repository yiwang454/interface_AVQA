"""Direct one-shot Qwen3-Omni AVQA: fixed system prompt, optimizable user instruction.

No planner, no tools, no turn loop -- this calls Qwen3-Omni exactly once per
example. All text comes from ``prompt_config.prompt_value``/``render_prompt``,
never from a literal in this file, so Stage 2's optimizer wrapper can change
behavior purely through ``prompt_config.prompt_overrides({("direct_qwen",
"user_instruction"): candidate_text})`` around one ``forward()`` call --
this module never needs to change for that to work.

Media-layout selection mirrors ``call_qwen_perception`` in the source repo's
``dspy_avqa/tools.py``: if a ``clients.reliable`` batch profile is active
(set by a caller wrapping many examples in
``ReliableQwenExecutor.run_batch``), it overrides this program's own
``QwenModelProfile`` layout for that one call, so the same instance behaves
correctly both as a bare baseline and as a reliable-batch worker.
"""

from __future__ import annotations

from typing import Any
import json

import dspy

from ..clients.qwen_omni import call_qwen_messages, to_data_url
from ..clients.reliable import active_qwen_request_profile, qwen_request_slot
from ..config import QwenModelProfile, resolve_qwen_profile
from ..metrics import format_options, normalize_option_letter
from ..prompt_config import prompt_value, render_prompt


class DirectQwenAVQAProgram(dspy.Module):
    """One-shot direct-Qwen AVQA: ``direct_qwen.user_instruction`` is the only optimizable text."""

    task_backend = "qwen"

    def __init__(self, *, profile: QwenModelProfile | None = None) -> None:
        super().__init__()
        self.profile = profile or resolve_qwen_profile("direct_qwen")

    def _active_media_layout(self) -> tuple[bool, bool]:
        """Return (video_first, use_audio_in_video), preferring an active batch profile."""
        request_profile = active_qwen_request_profile()
        if request_profile is None:
            return self.profile.video_first, self.profile.use_audio_in_video
        return request_profile.video_first, request_profile.use_audio_in_video

    def build_messages(
        self,
        *,
        question: str,
        options_json: str,
        video_path: str,
        audio_path: str | None,
    ) -> list[dict[str, Any]]:
        """Build the Qwen chat messages for one example from the active prompt config."""
        if not video_path:
            raise ValueError("DirectQwenAVQAProgram requires a non-empty video_path")

        video_first, use_audio_in_video = self._active_media_layout()
        attach_separate_audio = bool(audio_path) and not self.profile.video_only and not use_audio_in_video
        if not self.profile.video_only and not use_audio_in_video and not audio_path:
            raise ValueError(
                "DirectQwenAVQAProgram requires audio_path unless video_only or "
                "use_audio_in_video is enabled for the direct_qwen role"
            )

        prompt_text = render_prompt(
            "direct_qwen",
            "qa_prompt_template",
            user_instruction=prompt_value("direct_qwen", "user_instruction"),
            question=question,
            formatted_options=format_options(options_json),
            options=json.loads(options_json),
        )

        video_content = {"type": "video_url", "video_url": {"url": to_data_url(
            video_path, media_url_mode=self.profile.media_url_mode
        )}}
        audio_content = (
            {"type": "audio_url", "audio_url": {"url": to_data_url(
                audio_path, media_url_mode=self.profile.media_url_mode
            )}}
            if attach_separate_audio
            else None
        )
        content: list[dict[str, Any]] = []
        if video_first:
            content.append(video_content)
            if audio_content is not None:
                content.append(audio_content)
        else:
            if audio_content is not None:
                content.append(audio_content)
            content.append(video_content)
        content.append({"type": "text", "text": prompt_text})

        messages: list[dict[str, Any]] = []
        system_prompt = prompt_value("direct_qwen", "system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _call_qwen(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], str]:
        _, use_audio_in_video = self._active_media_layout()
        with qwen_request_slot():
            return call_qwen_messages(
                messages,
                model=self.profile.model,
                api_key=self.profile.api_key,
                base_url=self.profile.base_url,
                timeout=self.profile.timeout,
                max_retries=self.profile.max_retries,
                retry_delay_s=self.profile.retry_delay_s,
                temperature=self.profile.temperature,
                qwen_seed=self.profile.seed,
                top_p=self.profile.top_p,
                top_k=self.profile.top_k,
                max_tokens=self.profile.max_tokens,
                fps=self.profile.fps,
                max_frames=self.profile.max_frames,
                use_audio_in_video=use_audio_in_video,
                repetition_penalty=self.profile.repetition_penalty,
                enable_thinking=self.profile.enable_thinking,
                include_enable_thinking=self.profile.include_enable_thinking,
                stream=self.profile.stream,
                return_thinking=True,
            )

    def forward(
        self,
        question: str,
        options_json: str,
        video_path: str,
        audio_path: str | None = None,
        video_id: str | None = None,
        question_id: str | None = None,
    ) -> dspy.Prediction:
        messages = self.build_messages(
            question=question,
            options_json=options_json,
            video_path=video_path,
            audio_path=audio_path,
        )
        response_text, token_usage, thinking_text = self._call_qwen(messages)
        return dspy.Prediction(
            answer=normalize_option_letter(response_text),
            response_text=response_text,
            thinking=thinking_text,
            token_usage=token_usage,
            video_id=video_id,
            question_id=question_id,
        )
