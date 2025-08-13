"""Dual-cache management for raw and preprocessed audio files.

This module handles saving raw downloaded audio to cache and ensuring
preprocessed WAV files exist, implementing the dual-cache strategy.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def save_raw_to_cache(temp_path: str, video_id: str) -> str:
    """Save downloaded raw audio to the local cache.

    Args:
        temp_path: Path to temporary downloaded audio file
        video_id: YouTube video ID

    Returns:
        Path to the cached raw audio file

    Raises:
        RuntimeError: If file operations fail
    """
    # Find project root and create raw cache directory
    current = Path(__file__).resolve().parent
    project_root = None
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            project_root = parent
            break

    if not project_root:
        project_root = Path.cwd()

    raw_cache_dir = project_root / "data/local_media/raw"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    # Determine the original file extension
    temp_file = Path(temp_path)
    if not temp_file.exists():
        raise RuntimeError(f"Temporary file not found: {temp_path}")

    # Get the original extension from the downloaded file
    original_ext = temp_file.suffix
    if not original_ext:
        # Fallback to common audio extensions
        original_ext = ".webm"

    # Create the target path in raw cache
    raw_cache_path = raw_cache_dir / f"{video_id}{original_ext}"
    raw_cache_tmp = raw_cache_path.with_suffix(raw_cache_path.suffix + ".tmp")

    try:
        # Copy to temporary location first, then atomically move
        shutil.copy2(temp_path, raw_cache_tmp)
        os.replace(raw_cache_tmp, raw_cache_path)
        logger.info(f"Saved raw audio to cache: {raw_cache_path}")
        return str(raw_cache_path)
    except Exception as e:
        # Clean up temp file if it exists
        if raw_cache_tmp.exists():
            try:
                raw_cache_tmp.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to save raw audio to cache: {e}") from e


def ensure_full_preprocessed_from_raw(raw_path: str, video_id: str) -> str:
    """Ensure full preprocessed WAV exists from raw audio.

    Args:
        raw_path: Path to raw audio file
        video_id: YouTube video ID

    Returns:
        Path to the preprocessed WAV file

    Raises:
        RuntimeError: If conversion fails
    """
    # Find project root and create preprocessed cache directory
    current = Path(__file__).resolve().parent
    project_root = None
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            project_root = parent
            break

    if not project_root:
        project_root = Path.cwd()

    preprocessed_cache_dir = project_root / "data/local_media/preprocessed"
    preprocessed_cache_dir.mkdir(parents=True, exist_ok=True)

    # Create the target path in preprocessed cache
    preprocessed_path = preprocessed_cache_dir / f"{video_id}_16k_mono.wav"
    preprocessed_tmp = preprocessed_cache_dir / f"{video_id}_16k_mono.tmp.wav"

    # Skip if already exists
    if preprocessed_path.exists():
        logger.info(f"Preprocessed audio already exists: {preprocessed_path}")
        return str(preprocessed_path)

    try:
        # Convert raw to 16kHz mono WAV with high quality settings
        cmd = [
            "ffmpeg",
            "-i", raw_path,
            "-map", "a:0",  # Use first audio stream
            "-ac", "1",  # mono
            "-ar", "16000",  # 16kHz
            "-sample_fmt", "s16",  # 16-bit signed
            "-vn",  # no video
            "-sn",  # no subtitles
            "-dn",  # no data
            "-y",  # overwrite
            "-nostdin",  # non-interactive
            "-loglevel", "error",  # minimal output
            str(preprocessed_tmp),
        ]

        logger.info(f"Converting raw audio to preprocessed: {raw_path}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Atomically move to final location
        os.replace(preprocessed_tmp, preprocessed_path)
        logger.info(f"Created preprocessed audio: {preprocessed_path}")
        return str(preprocessed_path)

    except subprocess.CalledProcessError as e:
        # Clean up temp file if it exists
        if preprocessed_tmp.exists():
            try:
                preprocessed_tmp.unlink()
            except OSError:
                pass
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr}") from e
    except Exception as e:
        # Clean up temp file if it exists
        if preprocessed_tmp.exists():
            try:
                preprocessed_tmp.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to create preprocessed audio: {e}") from e
