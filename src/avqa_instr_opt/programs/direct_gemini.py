"""Direct one-call Gemini AVQA with an optimizer-visible user instruction."""

from __future__ import annotations

import json
from typing import Any

import dspy

from ..clients.gemini import call_gemini_messages, media_data_from_path
from ..config import GeminiModelProfile, resolve_gemini_profile
from ..metrics import format_options, normalize_option_letter
from ..prompt_config import prompt_value, render_prompt


class DirectGeminiAVQAProgram(dspy.Module):
    """Send the MP4 and complete multiple-choice prompt to Gemini once."""

    task_backend = "gemini"

    def __init__(self, *, profile: GeminiModelProfile | None = None) -> None:
        super().__init__()
        self.profile = profile or resolve_gemini_profile("direct_gemini")

    def build_contents(
        self,
        *,
        question: str,
        options_json: str,
        video_path: str,
        audio_path: str | None,
    ) -> list[dict[str, Any]]:
        if not video_path:
            raise ValueError("DirectGeminiAVQAProgram requires a non-empty video_path")
        if not self.profile.video_only and not audio_path:
            raise ValueError(
                "DirectGeminiAVQAProgram requires audio_path unless video_only is enabled"
            )
        prompt_text = render_prompt(
            "direct_qwen",
            "qa_prompt_template",
            user_instruction=prompt_value("direct_qwen", "user_instruction"),
            question=question,
            formatted_options=format_options(options_json),
            options=json.loads(options_json),
        )
        def media_part(path: str) -> dict[str, Any]:
            return media_data_from_path(
                path,
                media_mode=self.profile.media_mode,
                local_data_root=self.profile.local_data_root,
                gcs_data_root=self.profile.gcs_data_root,
            )

        parts: list[dict[str, Any]] = []
        if audio_path and not self.profile.video_only:
            parts.append(media_part(audio_path))
        # The DailyOmni MP4 contains its audio track, so video_only suppresses
        # only the redundant standalone WAV attachment.
        parts.extend([media_part(video_path), {"text": prompt_text}])
        return [{"role": "user", "parts": parts}]

    def forward(
        self,
        question: str,
        options_json: str,
        video_path: str,
        audio_path: str | None = None,
        video_id: str | None = None,
        question_id: str | None = None,
    ) -> dspy.Prediction:
        response_text, token_usage, thinking_text = call_gemini_messages(
            self.build_contents(
                question=question,
                options_json=options_json,
                video_path=video_path,
                audio_path=audio_path,
            ),
            system_prompt=prompt_value("direct_qwen", "system_prompt"),
            model=self.profile.model,
            api_key=self.profile.api_key or "",
            base_url=self.profile.base_url or "",
            timeout=self.profile.timeout,
            max_retries=self.profile.max_retries,
            retry_delay_s=self.profile.retry_delay_s,
            include_thoughts=self.profile.include_thoughts,
            thinking_budget=self.profile.thinking_budget,
            return_thinking=True,
            temperature=self.profile.temperature,
            gemini_seed=self.profile.seed,
            top_p=self.profile.top_p,
            top_k=self.profile.top_k,
            max_tokens=self.profile.max_tokens,
            retry_degenerate_response=True,
            print_first_prompt=self.profile.print_first_prompt,
        )
        return dspy.Prediction(
            answer=normalize_option_letter(response_text),
            response_text=response_text,
            thinking=thinking_text,
            token_usage=token_usage,
            video_id=video_id,
            question_id=question_id,
        )
