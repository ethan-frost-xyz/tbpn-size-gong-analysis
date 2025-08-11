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
from dataclasses import asdict, dataclass
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





@dataclass
class LocalMediaEntry:
    """Record describing a locally cached media item."""

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
        """Initialize the index, ensuring directories and loading existing data.

        Args:
            base_dir: Base directory for local media cache.
        """
        self.base_dir = Path(base_dir)
        self.preprocessed_dir = self.base_dir / "preprocessed"
        self.index_path = self.base_dir / "index.json"
        self._index: dict[str, dict[str, Any]] = {}

        # Ensure directories exist
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        (self.base_dir / "raw").mkdir(parents=True, exist_ok=True)

        self.load()

    def load(self) -> None:
        """Load the index from disk if present; otherwise start with empty."""
        if self.index_path.exists():
            try:
                with open(self.index_path, encoding="utf-8") as f:
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
        """Persist the current index atomically to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        payload = json.dumps(self._index, indent=2, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, self.index_path)

    def get(self, video_id: str) -> Optional[dict[str, Any]]:
        """Return the metadata record for a given video id, if present."""
        return self._index.get(video_id)

    def upsert(self, entry: LocalMediaEntry) -> None:
        """Insert or update an entry and set timestamps appropriately."""
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
        """Update the `last_used_at` timestamp for an existing record."""
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

    This function implements dual-cache: it ensures both raw and preprocessed
    audio are cached, then provides the requested audio segment.

    Returns a tuple of (path, metadata_dict). The metadata contains keys that
    align with LocalMediaEntry fields. If metadata is unavailable (e.g.,
    running strictly offline), some fields may be empty strings.
    """
    idx = index or LocalMediaIndex()

    # Check if full preprocessed file exists
    full_preprocessed_path = _preprocessed_wav_path(idx.preprocessed_dir, video_id)

    # Check if raw file exists (for index population)
    raw_cache_dir = idx.base_dir / "raw"
    raw_files = list(raw_cache_dir.glob(f"{video_id}.*"))
    raw_path = str(raw_files[0]) if raw_files else ""

    # If trimming is requested, we need to create a temporary trimmed file
    if start is not None or duration is not None:
        # Create a temporary output path for the trimmed audio
        import tempfile
        temp_output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_output.close()
        output_path = temp_output.name

        # If full preprocessed exists, trim from it
        if full_preprocessed_path.exists():
            meta = idx.get(video_id) or {}
            idx.touch_last_used(video_id)

            # Ensure raw_path is populated in index if we found it
            if raw_path and (not meta or not meta.get("raw_path")):
                meta["raw_path"] = raw_path
                idx.upsert(
                    LocalMediaEntry(
                        video_id=video_id,
                        source_url=url,
                        preprocessed_path=str(full_preprocessed_path),
                        raw_path=raw_path,
                        video_title=meta.get("video_title", ""),
                        upload_date=meta.get("upload_date", ""),
                    )
                )
                meta = idx.get(video_id) or {}

            # Import trim function
            from gong_detector.core.utils.youtube_utils import trim_from_preprocessed
            trim_from_preprocessed(str(full_preprocessed_path), output_path, start, duration)

            return output_path, meta
        else:
            # If local only, fail fast
            if local_only:
                raise RuntimeError(
                    "Preprocessed audio not found locally and local_only is enabled"
                )

            # Download and create both raw and preprocessed, then trim
            audio_path, video_title, upload_date = download_and_trim_youtube_audio(
                url=url,
                output_path=output_path,
                start_time=start,
                duration=duration,
            )

            # Get updated metadata from index (should now include raw_path)
            meta = idx.get(video_id) or {}
            return output_path, meta
    else:
        # No trimming requested - return the full preprocessed path
        if full_preprocessed_path.exists():
            meta = idx.get(video_id) or {}
            idx.touch_last_used(video_id)

            # Ensure raw_path is populated in index if we found it
            if raw_path and (not meta or not meta.get("raw_path")):
                meta["raw_path"] = raw_path
                idx.upsert(
                    LocalMediaEntry(
                        video_id=video_id,
                        source_url=url,
                        preprocessed_path=str(full_preprocessed_path),
                        raw_path=raw_path,
                        video_title=meta.get("video_title", ""),
                        upload_date=meta.get("upload_date", ""),
                    )
                )
                meta = idx.get(video_id) or {}

            if not meta:
                # Create minimal record to keep last_used tracking consistent
                idx.upsert(
                    LocalMediaEntry(
                        video_id=video_id,
                        source_url=url,
                        preprocessed_path=str(full_preprocessed_path),
                        raw_path=raw_path,
                    )
                )
                meta = idx.get(video_id) or {}
            return str(full_preprocessed_path), meta

        # If local only, fail fast
        if local_only:
            raise RuntimeError(
                "Preprocessed audio not found locally and local_only is enabled"
            )

        # Download and create both raw and preprocessed
        audio_path, video_title, upload_date = download_and_trim_youtube_audio(
            url=url,
            output_path=str(full_preprocessed_path),
            start_time=None,
            duration=None,
        )

        # Get updated metadata from index (should now include raw_path)
        meta = idx.get(video_id) or {}
        return str(full_preprocessed_path), meta


