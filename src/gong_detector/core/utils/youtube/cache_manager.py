"""Dual-cache management for raw and preprocessed audio files.

This module handles saving raw downloaded audio to cache and ensuring
preprocessed WAV files exist, implementing the dual-cache strategy.
"""

import logging
import os
import subprocess
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


def save_raw_to_cache(temp_path: str, video_id: str) -> str:
    """Save downloaded raw audio to the local cache as high-quality WAV.

    Converts the downloaded audio (typically WebM) to uncompressed WAV format
    for optimal compatibility with loudness analysis tools and to avoid
    deprecated librosa fallback methods.

    Parameters
    ----------
    temp_path : str
        Temporary file produced by yt-dlp.
    video_id : str
        YouTube video identifier used to name the cached asset.

    Returns
    -------
    str
        Path to the cached raw audio file (WAV format).

    Raises
    ------
    RuntimeError
        Raised when conversion or file operations fail.
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

    # Validate input file
    temp_file = Path(temp_path)
    if not temp_file.exists():
        raise RuntimeError(f"Temporary file not found: {temp_path}")

    # Always save raw cache as WAV for maximum compatibility
    # This eliminates soundfile compatibility issues and deprecated librosa warnings
    raw_cache_path = raw_cache_dir / f"{video_id}.wav"
    raw_cache_tmp = raw_cache_dir / f"{video_id}.tmp.wav"

    try:
        # Convert to storage-optimized WAV for LUFS analysis
        # 16kHz 16-bit PCM provides sufficient quality for loudness measurements while minimizing storage
        cmd = [
            "ffmpeg",
            "-i",
            str(temp_path),
            "-map",
            "a:0",  # Use first audio stream
            "-acodec",
            "pcm_s16le",  # 16-bit PCM (sufficient dynamic range for LUFS)
            "-ar",
            "16000",  # 16kHz sample rate (adequate for loudness analysis, saves storage)
            "-af",
            "aresample=resampler=soxr:precision=28:cheby=1",  # High-quality resampling
            "-vn",  # no video
            "-sn",  # no subtitles
            "-dn",  # no data
            "-y",  # overwrite
            "-nostdin",  # non-interactive
            "-loglevel",
            "error",  # minimal output
            str(raw_cache_tmp),
        ]

        logger.info(
            f"Converting downloaded audio to storage-optimized 16kHz WAV: {temp_path}"
        )
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Atomically move to final location
        os.replace(raw_cache_tmp, raw_cache_path)
        logger.info(f"Saved raw audio to cache as WAV: {raw_cache_path}")

        # Clean up original downloaded file to prevent duplication
        try:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug(f"Cleaned up original file: {temp_path}")
        except OSError as e:
            logger.warning(f"Could not clean up original file {temp_path}: {e}")

        return str(raw_cache_path)

    except subprocess.CalledProcessError as e:
        # Clean up temp files if they exist
        if raw_cache_tmp.exists():
            try:
                raw_cache_tmp.unlink()
            except OSError:
                pass
        # Also clean up original file if conversion failed
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
        raise RuntimeError(f"FFmpeg conversion to WAV failed: {e.stderr}") from e
    except Exception as e:
        # Clean up temp files if they exist
        if raw_cache_tmp.exists():
            try:
                raw_cache_tmp.unlink()
            except OSError:
                pass
        # Also clean up original file if conversion failed
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
        raise RuntimeError(f"Failed to save raw audio to cache: {e}") from e


def ensure_full_preprocessed_from_raw(raw_path: str, video_id: str) -> str:
    """Ensure full preprocessed WAV exists from raw audio.

    Parameters
    ----------
    raw_path : str
        Path to the cached raw audio file.
    video_id : str
        YouTube video identifier used to name the preprocessed asset.

    Returns
    -------
    str
        Path to the preprocessed WAV file.

    Raises
    ------
    RuntimeError
        Raised when conversion fails.
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
            "-i",
            raw_path,
            "-map",
            "a:0",  # Use first audio stream
            "-ac",
            "1",  # mono
            "-ar",
            "16000",  # 16kHz
            "-sample_fmt",
            "s16",  # 16-bit signed
            "-vn",  # no video
            "-sn",  # no subtitles
            "-dn",  # no data
            "-y",  # overwrite
            "-nostdin",  # non-interactive
            "-loglevel",
            "error",  # minimal output
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
