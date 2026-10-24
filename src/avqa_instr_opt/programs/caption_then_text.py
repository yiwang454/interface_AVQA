"""Fixed two-stage AVQA: live captioner -> fixed text reasoner.

Unlike the source repo's ReAct captioner (which is one tool among
``ask_caption``/``ask_perception``/``video_clip_qa`` inside a planner loop
that may call it, skip it, or call other tools first), this pipeline is
unconditionally caption-then-answer for every example: exactly one live
caption call, then exactly one text-only reasoning call over
``{caption, question, options}``. ``captioner.user_instruction`` and
``text_reasoner.user_instruction`` are separate optimization targets; the text
reasoner never sees the video.

The captioner does not receive the question/options (see
``build_caption_messages``), matching the direct_qwen_repository_selection
plan's requirement that the captioner target generic caption quality, not
question-conditioned answers.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from threading import Event, Lock
from typing import Any
import json

import dspy

from ..clients.gemini import call_gemini_messages, media_data_from_path
from ..clients.qwen_omni import call_qwen_messages, to_data_url
from ..clients.reliable import active_qwen_request_profile, qwen_request_slot
from ..clients.text_lm import build_text_lm
from ..config import (
    CaptionerModelProfile,
    GeminiModelProfile,
    QwenModelProfile,
    captioner_backend,
    resolve_captioner_profile,
)
from ..metrics import format_options, normalize_option_letter
from ..prompt_config import prompt_value, render_prompt


class CaptionResponseCache:
    """Thread-safe, process-local caption cache shared across optimizer candidates."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], tuple[str, dict[str, Any], str]] = {}
        self._inflight: dict[tuple[str, str], Event] = {}
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def __deepcopy__(self, memo):
        # Candidate programs must share one run-level cache.
        return self

    def get_or_compute(
        self,
        key: tuple[str, str],
        compute: Callable[[], tuple[str, dict[str, Any], str]],
    ) -> tuple[str, dict[str, Any], str]:
        while True:
            with self._lock:
                if key in self._values:
                    self.hits += 1
                    return self._values[key]
                event = self._inflight.get(key)
                if event is None:
                    event = Event()
                    self._inflight[key] = event
                    self.misses += 1
                    owner = True
                else:
                    owner = False
            if owner:
                try:
                    value = compute()
                    if str(value[0] or "").strip():
                        with self._lock:
                            self._values[key] = value
                    return value
                finally:
                    with self._lock:
                        self._inflight.pop(key, None)
                        event.set()
            event.wait()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._values), "hits": self.hits, "misses": self.misses}


class CaptionThenTextQAProgram(dspy.Module):
    """Fixed two-stage pipeline with independently optimizable user instructions."""

    def __init__(
        self,
        *,
        captioner_profile: CaptionerModelProfile | None = None,
        text_lm: dspy.LM | None = None,
        caption_cache: CaptionResponseCache | None = None,
    ) -> None:
        super().__init__()
        self.captioner_profile = captioner_profile or resolve_captioner_profile()
        self.captioner_backend = captioner_backend(self.captioner_profile)
        self.text_lm = text_lm or build_text_lm("text_reasoner")
        self.caption_cache = caption_cache

    def _active_media_layout(self) -> tuple[bool, bool]:
        """Return (video_first, use_audio_in_video), preferring an active batch profile."""
        if not isinstance(self.captioner_profile, QwenModelProfile):
            return True, False
        request_profile = active_qwen_request_profile()
        if request_profile is None:
            return self.captioner_profile.video_first, self.captioner_profile.use_audio_in_video
        return request_profile.video_first, request_profile.use_audio_in_video

    def build_caption_messages(
        self, *, video_path: str, audio_path: str | None
    ) -> list[dict[str, Any]]:
        """Build provider-native caption messages; deliberately question/option-free."""
        if not video_path:
            raise ValueError("CaptionThenTextQAProgram requires a non-empty video_path")

        if isinstance(self.captioner_profile, GeminiModelProfile):
            if not self.captioner_profile.video_only and not audio_path:
                raise ValueError(
                    "Gemini captioner requires audio_path unless video_only is enabled"
                )
            caption_prompt = render_prompt(
                "captioner",
                "caption_prompt_template",
                user_instruction=prompt_value("captioner", "user_instruction"),
            )
            def media_part(path: str) -> dict[str, Any]:
                return media_data_from_path(
                    path,
                    media_mode=self.captioner_profile.media_mode,
                    local_data_root=self.captioner_profile.local_data_root,
                    gcs_data_root=self.captioner_profile.gcs_data_root,
                )

            parts: list[dict[str, Any]] = []
            if audio_path and not self.captioner_profile.video_only:
                parts.append(media_part(audio_path))
            # video_only means no separate audio file; Gemini still receives the
            # MP4's embedded audio track, matching the sibling baseline.
            parts.extend([media_part(video_path), {"text": caption_prompt}])
            return [{"role": "user", "parts": parts}]

        video_first, use_audio_in_video = self._active_media_layout()
        attach_separate_audio = (
            bool(audio_path) and not self.captioner_profile.video_only and not use_audio_in_video
        )
        if not self.captioner_profile.video_only and not use_audio_in_video and not audio_path:
            raise ValueError(
                "CaptionThenTextQAProgram requires audio_path unless video_only or "
                "use_audio_in_video is enabled for the captioner_qwen role"
            )

        caption_prompt = render_prompt(
            "captioner",
            "caption_prompt_template",
            user_instruction=prompt_value("captioner", "user_instruction"),
        )

        video_content = {"type": "video_url", "video_url": {"url": to_data_url(
            video_path, media_url_mode=self.captioner_profile.media_url_mode
        )}}
        audio_content = (
            {"type": "audio_url", "audio_url": {"url": to_data_url(
                audio_path, media_url_mode=self.captioner_profile.media_url_mode
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
        content.append({"type": "text", "text": caption_prompt})

        messages: list[dict[str, Any]] = []
        system_prompt = prompt_value("captioner", "system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _call_captioner(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any], str]:
        if isinstance(self.captioner_profile, GeminiModelProfile):
            return call_gemini_messages(
                messages,
                system_prompt=prompt_value("captioner", "system_prompt"),
                model=self.captioner_profile.model,
                api_key=self.captioner_profile.api_key or "",
                base_url=self.captioner_profile.base_url or "",
                timeout=self.captioner_profile.timeout,
                max_retries=self.captioner_profile.max_retries,
                retry_delay_s=self.captioner_profile.retry_delay_s,
                include_thoughts=self.captioner_profile.include_thoughts,
                thinking_budget=self.captioner_profile.thinking_budget,
                return_thinking=True,
                temperature=self.captioner_profile.temperature,
                gemini_seed=self.captioner_profile.seed,
                top_p=self.captioner_profile.top_p,
                top_k=self.captioner_profile.top_k,
                max_tokens=self.captioner_profile.max_tokens,
                retry_degenerate_response=True,
                print_first_prompt=self.captioner_profile.print_first_prompt,
            )

        assert isinstance(self.captioner_profile, QwenModelProfile)
        _, use_audio_in_video = self._active_media_layout()
        with qwen_request_slot():
            return call_qwen_messages(
                messages,
                model=self.captioner_profile.model,
                api_key=self.captioner_profile.api_key,
                base_url=self.captioner_profile.base_url,
                timeout=self.captioner_profile.timeout,
                max_retries=self.captioner_profile.max_retries,
                retry_delay_s=self.captioner_profile.retry_delay_s,
                temperature=self.captioner_profile.temperature,
                qwen_seed=self.captioner_profile.seed,
                top_p=self.captioner_profile.top_p,
                top_k=self.captioner_profile.top_k,
                max_tokens=self.captioner_profile.max_tokens,
                fps=self.captioner_profile.fps,
                max_frames=self.captioner_profile.max_frames,
                use_audio_in_video=use_audio_in_video,
                repetition_penalty=self.captioner_profile.repetition_penalty,
                enable_thinking=self.captioner_profile.enable_thinking,
                include_enable_thinking=self.captioner_profile.include_enable_thinking,
                stream=self.captioner_profile.stream,
                return_thinking=True,
            )

    def _call_reasoner(self, *, caption: str, question: str, options_json: str) -> tuple[str, dict, str]:
        """Run the fixed text reasoner; DSPy currently exposes text only here."""
        prompt_text = render_prompt(
            "text_reasoner",
            "prompt_template",
            user_instruction=prompt_value("text_reasoner", "user_instruction"),
            caption=caption,
            question=question,
            formatted_options=format_options(options_json),
            options=json.loads(options_json), audio_caption="", video_caption=caption,
        )
        messages: list[dict[str, Any]] = []
        system_prompt = prompt_value("text_reasoner", "system_prompt")
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt_text})
        outputs = self.text_lm(messages=messages)
        return (str(outputs[0]) if outputs else ""), {}, ""

    def answer_from_caption(self, *, caption: str, question: str, options_json: str) -> str:
        """Public common reasoner path used by both live and cached captions."""
        return self._call_reasoner(
            caption=caption, question=question, options_json=options_json
        )[0]

    def forward_from_caption(
        self,
        *,
        caption: str,
        question: str,
        options_json: str,
        video_id: str | None = None,
        question_id: str | None = None,
    ) -> dspy.Prediction:
        """Run only the common reasoner stage for an externally cached caption."""
        response_text = self.answer_from_caption(
            caption=caption, question=question, options_json=options_json)
        return dspy.Prediction(
            answer=normalize_option_letter(response_text), response_text=response_text,
            caption=caption, caption_thinking="", caption_token_usage={},
            reasoner_thinking="", reasoner_token_usage={},
            video_id=video_id, question_id=question_id,
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
        caption_messages = self.build_caption_messages(video_path=video_path, audio_path=audio_path)
        if self.caption_cache is None:
            caption_text, caption_token_usage, caption_thinking = self._call_captioner(
                caption_messages
            )
        else:
            instruction_hash = sha256(
                prompt_value("captioner", "user_instruction").encode("utf-8")
            ).hexdigest()
            sample_key = str(question_id or video_path)
            caption_text, caption_token_usage, caption_thinking = self.caption_cache.get_or_compute(
                (instruction_hash, sample_key), lambda: self._call_captioner(caption_messages)
            )
        if not caption_text.strip():
            return dspy.Prediction(
                answer="", response_text="", caption=caption_text,
                caption_thinking=caption_thinking, caption_token_usage=caption_token_usage,
                reasoner_thinking="", reasoner_token_usage={},
                video_id=video_id, question_id=question_id,
            )
        response_text = self.answer_from_caption(
            caption=caption_text, question=question, options_json=options_json)
        return dspy.Prediction(
            answer=normalize_option_letter(response_text),
            response_text=response_text,
            caption=caption_text,
            caption_thinking=caption_thinking,
            caption_token_usage=caption_token_usage,
            reasoner_thinking="",
            reasoner_token_usage={},
            video_id=video_id,
            question_id=question_id,
        )
