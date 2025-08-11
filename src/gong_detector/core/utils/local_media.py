"""Local media cache utilities.

Provides a simple on-disk cache for preprocessed audio files and a small
metadata index to avoid re-downloading and re-processing YouTube audio.

Folder layout under data/local_media/:
- preprocessed/: 16kHz mono WAV ready for YAMNet (VIDEOID_16k_mono.wav)
- raw/ (optional): original downloads if ever needed (not used yet)
- index.json: metadata keyed by video_id
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from gong_detector.core.utils.youtube_utils import (
    download_and_trim_youtube_audio,
)


def _find_project_root() -> Path:
    """Find project root by walking up from this file."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


_PROJECT_ROOT = _find_project_root()
LOCAL_MEDIA_BASE = _PROJECT_ROOT / "data/local_media"
LOCAL_MEDIA_PREPROCESSED = LOCAL_MEDIA_BASE / "preprocessed"
LOCAL_MEDIA_INDEX = LOCAL_MEDIA_BASE / "index.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def video_id_from_url(url: str) -> str:
    """Extract YouTube video ID from a URL.

    Supports standard and short URLs. Returns empty string if not found.
    """
    # youtu.be/<id>
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    # youtube.com/watch?v=<id>
    match = re.search(r"v=([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    # youtube.com/embed/<id>
    match = re.search(r"/embed/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)

    return ""


@dataclass
class LocalMediaEntry:
    video_id: str
    source_url: str
    video_title: str = ""
    upload_date: str = ""
    preprocessed_path: str = ""
    raw_path: str = ""
    created_at: str = ""
    last_used_at: str = ""


class LocalMediaIndex:
    """Minimal index for cached local media."""

    def __init__(self, base_dir: Path | str = LOCAL_MEDIA_BASE) -> None:
        self.base_dir = Path(base_dir)
        self.preprocessed_dir = self.base_dir / "preprocessed"
        self.index_path = self.base_dir / "index.json"
        self._index: dict[str, dict[str, Any]] = {}

        # Ensure directories exist
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "raw").mkdir(parents=True, exist_ok=True)

        self.load()

    def load(self) -> None:
        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._index = data
                    else:
                        self._index = {}
            except json.JSONDecodeError:
                # corrupt index – start fresh but keep file on next save
                self._index = {}
        else:
            self._index = {}

    def save(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        payload = json.dumps(self._index, indent=2, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, self.index_path)

    def get(self, video_id: str) -> Optional[dict[str, Any]]:
        return self._index.get(video_id)

    def upsert(self, entry: LocalMediaEntry) -> None:
        now = _utc_now_iso()
        existing = self._index.get(entry.video_id)
        record = asdict(entry)
        if existing is None:
            if not record.get("created_at"):
                record["created_at"] = now
        else:
            # preserve created_at
            record["created_at"] = existing.get("created_at", now)

        record["last_used_at"] = now
        self._index[entry.video_id] = record
        self.save()

    def touch_last_used(self, video_id: str) -> None:
        if video_id in self._index:
            self._index[video_id]["last_used_at"] = _utc_now_iso()
            self.save()


def _preprocessed_wav_path(preprocessed_dir: Path, video_id: str) -> Path:
    return preprocessed_dir / f"{video_id}_16k_mono.wav"


def ensure_preprocessed_audio(
    video_id: str,
    url: str,
    start: Optional[int] = None,
    duration: Optional[int] = None,
    local_only: bool = False,
    index: Optional[LocalMediaIndex] = None,
) -> tuple[str, dict[str, Any]]:
    """Ensure a preprocessed 16kHz mono WAV exists for the given video.

    Returns a tuple of (path, metadata_dict). The metadata contains keys that
    align with LocalMediaEntry fields. If metadata is unavailable (e.g.,
    running strictly offline), some fields may be empty strings.
    """
    idx = index or LocalMediaIndex()

    # Clean up stray tmp file if present
    target_path = _preprocessed_wav_path(idx.preprocessed_dir, video_id)
    # Use a tmp path that still has .wav extension for ffmpeg container selection
    tmp_path = target_path.with_name(target_path.stem + ".tmp" + target_path.suffix)
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass

    # If already exists, update last_used and return
    if target_path.exists():
        meta = idx.get(video_id) or {}
        idx.touch_last_used(video_id)
        if not meta:
            # Create minimal record to keep last_used tracking consistent
            idx.upsert(
                LocalMediaEntry(
                    video_id=video_id,
                    source_url=url,
                    preprocessed_path=str(target_path),
                )
            )
            meta = idx.get(video_id) or {}
        return str(target_path), meta

    # If local only, fail fast
    if local_only:
        raise RuntimeError(
            "Preprocessed audio not found locally and local_only is enabled"
        )

    # Otherwise, download and preprocess into a tmp, then move atomically
    idx.preprocessed_dir.mkdir(parents=True, exist_ok=True)
    # Write to tmp file first
    output_tmp = str(tmp_path)
    audio_path, video_title, upload_date = download_and_trim_youtube_audio(
        url=url,
        output_path=output_tmp,
        start_time=start,
        duration=duration,
    )

    # Atomically move to final
    os.replace(output_tmp, target_path)

    # Update index
    entry = LocalMediaEntry(
        video_id=video_id,
        source_url=url,
        video_title=video_title or "",
        upload_date=upload_date or "",
        preprocessed_path=str(target_path),
        raw_path="",
    )
    idx.upsert(entry)
    return str(target_path), asdict(entry)


