"""Shared wall-clock reporting for command-line entry points."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, TypeVar


T = TypeVar("T")


def _format_duration(seconds: float) -> str:
    hours, remainder = divmod(max(0.0, seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def run_with_completion_timer(label: str, operation: Callable[[], T]) -> T:
    """Run an operation and report elapsed wall time even when it raises."""
    started = perf_counter()
    try:
        return operation()
    finally:
        elapsed = perf_counter() - started
        finished_at = datetime.now(timezone.utc).isoformat()
        print(
            f"{label} finished at {finished_at}; "
            f"elapsed wall time: {_format_duration(elapsed)} ({elapsed:.2f} seconds)",
            flush=True,
        )
