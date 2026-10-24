#!/usr/bin/env python3
"""Convert Daily-Omni metadata to the JSONL format used by this repository."""

from __future__ import annotations

import argparse
import gzip
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from lhotse import AudioSource, CutSet, MonoCut, Recording, SupervisionSegment


DEFAULT_SAMPLING_RATE = 44_100


@contextmanager
def open_text_file(path: Path, mode: str) -> Iterator[TextIO]:
    """Open plain or gzip-compressed text based on the filename suffix."""
    if path.suffix == ".gz":
        with gzip.open(path, f"{mode}t", encoding="utf-8") as handle:
            yield handle
    else:
        with path.open(mode, encoding="utf-8") as handle:
            yield handle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-json",
        type=Path,
        required=True,
        help="Official Daily-Omni qa.json.",
    )
    parser.add_argument(
        "--video-root",
        "--root-path",
        dest="video_root",
        type=Path,
        required=True,
        help="Directory containing <video_id>/<video_id>_video.mp4 files.",
    )
    parser.add_argument(
        "--output-jsonl",
        "--output",
        dest="output_jsonl",
        type=Path,
        required=True,
        help="Destination JSONL or JSONL.GZ.",
    )
    return parser


def load_qa_items(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected a list of QA objects in {path}")
    return data


def to_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().lower()
    if text.endswith("s"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return 0.0


def build_custom_block(item: dict[str, Any]) -> dict[str, Any]:
    custom = {
        "options": item.get("Choice", []),
        "answer": item.get("Answer"),
        "video_id": item.get("video_id"),
        "task_type": item.get("Type"),
    }
    for key, value in item.items():
        custom.setdefault(key, value)
    return custom


def iter_cut_entries(
    qa_items: Iterable[dict[str, Any]], video_root: Path
) -> Iterable[MonoCut]:
    per_video_counts: dict[str, int] = {}
    recording_cache: dict[str, Recording] = {}

    for global_idx, qa_item in enumerate(qa_items):
        video_id = qa_item.get("video_id")
        if not video_id:
            raise ValueError(f"Missing 'video_id' in QA entry at index {global_idx}")

        video_id = str(video_id)
        video_path = (video_root / video_id / f"{video_id}_video.mp4").resolve()
        if not video_path.exists():
            print(f"Skipping QA entry {global_idx}: missing video file {video_path}")
            continue

        duration = to_seconds(qa_item.get("video_duration"))
        recording = recording_cache.get(video_id)
        if recording is None:
            recording = Recording(
                id=video_id,
                sampling_rate=DEFAULT_SAMPLING_RATE,
                num_samples=(
                    int(round(duration * DEFAULT_SAMPLING_RATE)) if duration > 0 else 0
                ),
                duration=duration,
                sources=[
                    AudioSource(type="file", channels=[0], source=str(video_path))
                ],
            )
            recording_cache[video_id] = recording

        per_video_counts[video_id] = per_video_counts.get(video_id, 0) + 1
        per_video_idx = per_video_counts[video_id]
        custom = build_custom_block(qa_item)
        custom.setdefault("video_path", str(video_path))
        supervision_id = qa_item.get("question_id") or f"{video_id}-{per_video_idx}"
        supervision = SupervisionSegment(
            id=str(supervision_id),
            recording_id=recording.id,
            start=0.0,
            duration=duration,
            channel=0,
            text=qa_item.get("Question", ""),
            custom=custom,
        )
        yield MonoCut(
            id=f"{video_id}-{per_video_idx}",
            start=0.0,
            duration=duration,
            channel=0,
            recording=recording,
            supervisions=[supervision],
        )


def derive_audio_path(video_path: str) -> Path:
    suffix = "_video.mp4"
    if not video_path.endswith(suffix):
        raise ValueError(f"video_path does not end with {suffix!r}: {video_path}")
    return Path(f"{video_path[:-len(suffix)]}_audio.wav")


def add_audio_paths(input_jsonl: Path, output_jsonl: Path) -> int:
    num_written = 0
    with open_text_file(input_jsonl, "r") as source, open_text_file(
        output_jsonl, "w"
    ) as destination:
        for line_idx, line in enumerate(source, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            try:
                custom = data["supervisions"][0]["custom"]
                video_path = custom["video_path"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(
                    f"Line {line_idx}: missing supervisions[0].custom.video_path"
                ) from exc

            audio_path = derive_audio_path(video_path)
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"Line {line_idx}: missing audio file {audio_path}"
                )
            custom["audio_path"] = str(audio_path)
            destination.write(json.dumps(data, ensure_ascii=False))
            destination.write("\n")
            num_written += 1
    return num_written


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    qa_json = args.qa_json.expanduser().resolve()
    video_root = args.video_root.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    qa_items = load_qa_items(qa_json)
    cuts = list(iter_cut_entries(qa_items, video_root))
    with tempfile.TemporaryDirectory(
        prefix=".daily_omni_prepare_", dir=output_jsonl.parent
    ) as temp_dir:
        base_jsonl = Path(temp_dir) / "daily_omni_cuts.jsonl"
        CutSet.from_cuts(cuts).to_file(base_jsonl)
        num_written = add_audio_paths(base_jsonl, output_jsonl)

    print(f"Wrote {num_written} cuts to {output_jsonl}")


if __name__ == "__main__":
    main()
