"""Batch-scoped empty-response recovery for one-shot Qwen calls.

Adapted from react-agent-avqa-optimize's ``dspy_avqa/reliable_qwen.py``. Kept
verbatim: the primary/fallback ``QwenRequestProfile`` policy (separate
audio+video with video first, then embedded audio), the process-local
profile context managers DSPy worker threads pick up
(``qwen_request_profile``/``qwen_request_slot``), and
``ReliableQwenExecutor.run_batch``'s concurrent-primary-then-serial-fallback
shape.

Dropped relative to the source module: everything keyed to a ReAct
``turn_trace`` (``prediction_has_empty_qwen_observation``,
``mark_empty_qwen_prediction``, ``QwenValidationScore``'s GEPA
validation-list-position trick). Direct-Qwen and caption-then-text rollouts
are one response per call, so ``response_is_empty`` below is the single
predicate ``run_batch(..., has_empty_qwen_response=response_is_empty)``
needs; GEPA's positional-score exclusion can be reintroduced in Stage 2 if
GEPA support is added there.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterator, Sequence, TypeVar


@dataclass(frozen=True)
class QwenRequestProfile:
    """A fixed Qwen AV media layout used for one evaluation pass."""

    name: str
    use_audio_in_video: bool
    video_first: bool
    serial: bool
    defer_empty_response_retries: bool = True


PRIMARY_QWEN_PROFILE = QwenRequestProfile(
    name="primary_separate_audio_first",
    use_audio_in_video=False,
    video_first=False,
    serial=False,
)
PRIMARY_EMBEDDED_AUDIO_QWEN_PROFILE = QwenRequestProfile(
    name="primary_embedded_audio",
    use_audio_in_video=True,
    # Only the video item is sent; the processor reads its audio track.
    video_first=True,
    serial=False,
)
RETRY_VIDEO_FIRST_QWEN_PROFILE = QwenRequestProfile(
    name="retry_separate_video_first",
    use_audio_in_video=False,
    video_first=True,
    serial=True,
)
RETRY_EMBEDDED_AUDIO_QWEN_PROFILE = QwenRequestProfile(
    name="retry_embedded_audio",
    use_audio_in_video=True,
    # There is no standalone audio item to order in this layout. Keeping the
    # value true makes the profile explicit and avoids consulting env flags.
    video_first=True,
    serial=True,
)


def response_is_empty(response_text: str) -> bool:
    """Return whether a one-shot Qwen response has no visible content."""
    return not str(response_text or "").strip()


_ACTIVE_PROFILE_LOCK = threading.RLock()
_ACTIVE_PROFILE: QwenRequestProfile | None = None
_SERIAL_QWEN_REQUEST_GATE = threading.BoundedSemaphore(value=1)


def _read_nonnegative_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer, got {raw_value!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def _read_positive_int_env(name: str, default: int) -> int:
    value = _read_nonnegative_int_env(name, default)
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def active_qwen_request_profile() -> QwenRequestProfile | None:
    """Return the batch profile without changing process environment variables."""
    with _ACTIVE_PROFILE_LOCK:
        return _ACTIVE_PROFILE


@contextmanager
def qwen_request_profile(profile: QwenRequestProfile) -> Iterator[None]:
    """Make a profile visible to DSPy's worker threads for one batch.

    ``contextvars`` do not automatically propagate into DSPy's worker threads,
    so this deliberately uses a small process-local state protected by a lock.
    Run only one optimization/evaluation batch per process at a time.
    """
    global _ACTIVE_PROFILE
    with _ACTIVE_PROFILE_LOCK:
        if _ACTIVE_PROFILE is not None:
            raise RuntimeError(
                "A ReliableQwenExecutor profile is already active. "
                "Run only one batch per process."
            )
        _ACTIVE_PROFILE = profile
    try:
        yield
    finally:
        with _ACTIVE_PROFILE_LOCK:
            _ACTIVE_PROFILE = None


@contextmanager
def qwen_request_slot() -> Iterator[None]:
    """Enforce max_inflight=1 for a serial fallback Qwen inference request."""
    profile = active_qwen_request_profile()
    if profile is None or not profile.serial:
        yield
        return
    with _SERIAL_QWEN_REQUEST_GATE:
        yield


TItem = TypeVar("TItem")
TResult = TypeVar("TResult")


@dataclass
class ReliableQwenBatchResult:
    results: list[Any]
    exhausted_indices: list[int]
    attempted_profiles: list[str]


class ReliableQwenExecutor:
    """Own the fixed primary/fallback policy for one batch of Qwen calls.

    ``max_batch_retries`` counts retries *after* the concurrent primary pass.
    With the default separate-audio primary, retries use separate audio/video
    with video first and then embedded audio. With an embedded-audio primary,
    every retry remains embedded audio.
    """

    def __init__(
        self,
        max_batch_retries: int | None = None,
        fallback_max_workers: int | None = None,
        primary_profile: QwenRequestProfile | None = None,
    ) -> None:
        self.max_batch_retries = (
            _read_nonnegative_int_env("QWEN_RELIABLE_MAX_BATCH_RETRIES", 2)
            if max_batch_retries is None
            else max_batch_retries
        )
        if self.max_batch_retries < 0:
            raise ValueError("max_batch_retries must be non-negative")
        self.fallback_max_workers = (
            _read_positive_int_env("QWEN_RELIABLE_FALLBACK_NUM_THREADS", 1)
            if fallback_max_workers is None
            else fallback_max_workers
        )
        if self.fallback_max_workers < 1:
            raise ValueError("fallback_max_workers must be >= 1")
        self._primary_profile = primary_profile or PRIMARY_QWEN_PROFILE

    @property
    def primary_profile(self) -> QwenRequestProfile:
        return self._primary_profile

    def retry_profile(self, retry_number: int) -> QwenRequestProfile:
        if retry_number < 1:
            raise ValueError("retry_number must start at 1")
        if self.primary_profile.use_audio_in_video:
            return RETRY_EMBEDDED_AUDIO_QWEN_PROFILE
        if retry_number == 1:
            return RETRY_VIDEO_FIRST_QWEN_PROFILE
        return RETRY_EMBEDDED_AUDIO_QWEN_PROFILE

    @contextmanager
    def primary_batch(self) -> Iterator[None]:
        with qwen_request_profile(self.primary_profile):
            yield

    @contextmanager
    def retry_batch(
        self,
        retry_number: int,
        *,
        allow_parallel_requests: bool = False,
    ) -> Iterator[QwenRequestProfile]:
        profile = self.retry_profile(retry_number)
        # The executor bounds fallback workers itself. Disable the per-request
        # serial gate only for a bounded parallel fallback.
        if allow_parallel_requests:
            profile = replace(profile, serial=False)
        with qwen_request_profile(profile):
            yield profile

    def run_batch(
        self,
        items: Sequence[TItem],
        *,
        run_item: Callable[[TItem], TResult],
        has_empty_qwen_response: Callable[[TResult], bool],
        max_workers: int,
    ) -> ReliableQwenBatchResult:
        """Run a concurrent primary batch and bounded recovery for empty responses."""
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        results: list[TResult | None] = [None] * len(items)
        with self.primary_batch():
            if max_workers == 1:
                for index, item in enumerate(items):
                    results[index] = run_item(item)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(run_item, item): index
                        for index, item in enumerate(items)
                    }
                    for future in as_completed(futures):
                        results[futures[future]] = future.result()

        # Every primary worker has completed before any fallback begins.
        completed_results = [result for result in results]
        if any(result is None for result in completed_results):
            raise RuntimeError("ReliableQwenExecutor primary batch returned an incomplete result set")
        invalid_indices = [
            index
            for index, result in enumerate(completed_results)
            if has_empty_qwen_response(result)
        ]
        attempted_profiles = [self.primary_profile.name]
        for retry_number in range(1, self.max_batch_retries + 1):
            if not invalid_indices:
                break
            fallback_workers = min(self.fallback_max_workers, len(invalid_indices))
            with self.retry_batch(
                retry_number,
                allow_parallel_requests=fallback_workers > 1,
            ) as profile:
                attempted_profiles.append(profile.name)
                if fallback_workers == 1:
                    for index in invalid_indices:
                        completed_results[index] = run_item(items[index])
                else:
                    with ThreadPoolExecutor(max_workers=fallback_workers) as executor:
                        futures = {
                            executor.submit(run_item, items[index]): index
                            for index in invalid_indices
                        }
                        for future in as_completed(futures):
                            completed_results[futures[future]] = future.result()
            invalid_indices = [
                index
                for index in invalid_indices
                if has_empty_qwen_response(completed_results[index])
            ]

        return ReliableQwenBatchResult(
            results=completed_results,
            exhausted_indices=invalid_indices,
            attempted_profiles=attempted_profiles,
        )
