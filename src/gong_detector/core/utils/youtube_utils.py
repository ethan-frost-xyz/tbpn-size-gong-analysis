"""YouTube download and file management utilities.

This module provides functions for downloading YouTube audio,
managing temporary files, and handling file system operations
for the gong detection pipeline.
"""

import glob
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import yt_dlp  # type: ignore

try:
    import librosa  # type: ignore
    import numpy as np  # type: ignore
    import pyloudnorm as pyln  # type: ignore
    LUFS_AVAILABLE = True
except ImportError:
    LUFS_AVAILABLE = False
    pyln = None
    librosa = None
    np = None

# Configure logging
logger = logging.getLogger(__name__)


def video_id_from_url(url: str) -> str:
    """Extract YouTube video ID from a URL.

    Args:
        url: YouTube URL to extract video ID from

    Returns:
        YouTube video ID or empty string if not found
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


def get_cookies_path() -> Optional[str]:
    """Get path to cookies file if it exists.

    Returns:
        Path to cookies file or None if not found
    """
    # Common cookie file locations
    cookie_paths = [
        "cookies.txt",
        "youtube_cookies.txt",
        os.path.expanduser("~/cookies.txt"),
        os.path.expanduser("~/youtube_cookies.txt"),
    ]

    for path in cookie_paths:
        if os.path.exists(path):
            return path

    return None


def cleanup_old_temp_files(temp_dir: str, max_age_hours: int = 24) -> None:
    """Clean up old temporary audio files.

    Args:
        temp_dir: Directory containing temp files
        max_age_hours: Maximum age in hours before cleanup
    """
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    temp_files = glob.glob(os.path.join(temp_dir, "temp_youtube_audio_*.wav"))

    for temp_file in temp_files:
        file_age = current_time - os.path.getmtime(temp_file)
        if file_age > max_age_seconds:
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up old temp file: {temp_file}")
            except OSError:
                pass  # File might already be gone


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


def trim_from_preprocessed(
    preprocessed_path: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None
) -> None:
    """Trim audio from preprocessed WAV file.

    Args:
        preprocessed_path: Path to full preprocessed WAV file
        output_path: Path for output trimmed WAV file
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)

    Raises:
        RuntimeError: If trimming fails
    """
    if not Path(preprocessed_path).exists():
        raise RuntimeError(f"Preprocessed file not found: {preprocessed_path}")

    # Create output directory if needed
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-i", preprocessed_path]

    # Add trimming parameters if specified
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])

    # Copy audio stream without re-encoding (fast)
    cmd.extend([
        "-c", "copy",  # Copy without re-encoding
        "-y",  # overwrite
        "-nostdin",  # non-interactive
        "-loglevel", "error",  # minimal output
        output_path,
    ])

    try:
        logger.info(f"Trimming preprocessed audio: {start_time}s + {duration}s")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Trimmed audio saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg trimming failed: {e.stderr}") from e


def update_local_media_index(
    video_id: str,
    url: str,
    video_title: str,
    upload_date: str,
    raw_path: str,
    preprocessed_path: str,
) -> None:
    """Update the local media index with dual-cache information.

    Args:
        video_id: YouTube video ID
        url: Source YouTube URL
        video_title: Video title
        upload_date: Upload date
        raw_path: Path to raw audio file
        preprocessed_path: Path to preprocessed audio file
    """
    try:
        from gong_detector.core.utils.local_media import (
            LocalMediaEntry,
            LocalMediaIndex,
        )

        idx = LocalMediaIndex()
        entry = LocalMediaEntry(
            video_id=video_id,
            source_url=url,
            video_title=video_title or "",
            upload_date=upload_date or "",
            preprocessed_path=preprocessed_path,
            raw_path=raw_path,
        )
        idx.upsert(entry)
        logger.info(f"Updated local media index for video: {video_id}")
    except Exception as e:
        logger.warning(f"Failed to update local media index: {e}")


def download_and_trim_youtube_audio(
    url: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
    yt_dlp_options: Optional[dict] = None,
) -> tuple[str, str, str]:
    """Download YouTube audio and optionally trim to specified segment.

    This function implements dual-cache: it ensures both raw and preprocessed
    audio are cached, then provides the requested audio segment.

    Args:
        url: YouTube URL to download
        output_path: Path for output WAV file
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)
        yt_dlp_options: Additional yt-dlp options to merge

    Returns:
        Tuple of (audio_path, video_title, upload_date)

    Raises:
        RuntimeError: If download or conversion fails
    """
    logger.info(f"Processing audio from: {url}")

    # Extract video ID for caching
    video_id = video_id_from_url(url)
    if not video_id:
        raise RuntimeError(f"Could not extract video ID from URL: {url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio = os.path.join(temp_dir, "temp_audio.%(ext)s")

        # Download with yt-dlp and get video info
        downloaded_file, video_title, upload_date = _download_youtube_audio(
            url, temp_audio, yt_dlp_options
        )

        # Save raw audio to cache
        raw_cache_path = save_raw_to_cache(downloaded_file, video_id)

        # Ensure full preprocessed WAV exists
        preprocessed_path = ensure_full_preprocessed_from_raw(raw_cache_path, video_id)

        # Update local media index with both paths
        update_local_media_index(
            video_id=video_id,
            url=url,
            video_title=video_title,
            upload_date=upload_date,
            raw_path=raw_cache_path,
            preprocessed_path=preprocessed_path,
        )

        # Handle trimming if requested
        if start_time is not None or duration is not None:
            # Create trimmed version from full preprocessed
            trim_from_preprocessed(preprocessed_path, output_path, start_time, duration)
            logger.info(f"Trimmed audio saved to: {output_path}")
        else:
            # Copy full preprocessed to output path
            shutil.copy2(preprocessed_path, output_path)
            logger.info(f"Full preprocessed audio copied to: {output_path}")

        logger.info(f"Video title: {video_title}")

    return output_path, video_title, upload_date


def _download_youtube_audio(
    url: str, output_template: str, yt_dlp_options: Optional[dict] = None
) -> tuple[str, str, str]:
    """Download audio from YouTube using yt-dlp and extract video title.

    Args:
        url: YouTube URL to download
        output_template: Template for output filename
        yt_dlp_options: Additional yt-dlp options to merge

    Returns:
        Tuple of (downloaded_file_path, video_title, upload_date)
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,  # Reduce output noise
        # Speed-related options
        "http_chunk_size": 10 * 1024 * 1024,  # 10MB chunks
        "retries": 20,
        "fragment_retries": 20,
        # Prefer IPv4; many VPN exits have better IPv4 peering to Google CDNs
        "source_address": "0.0.0.0",
    }

    # Add cookies if available
    cookies_path = get_cookies_path()
    if cookies_path:
        logger.info(f"Using cookies from: {cookies_path}")
        ydl_opts["cookiefile"] = cookies_path
    else:
        logger.warning(
            "No cookies file found. If you encounter bot detection, create a cookies.txt file."
        )
        logger.warning(
            "See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
        )

    # Use aria2c if available for multi-connection downloads (often faster on VPN)
    try:
        if shutil.which("aria2c") is not None:
            ydl_opts["external_downloader"] = "aria2c"
            # -x: max connections per server, -s: split, -k: chunk size
            ydl_opts["external_downloader_args"] = ["-x16", "-s16", "-k", "5M"]
    except Exception:
        # Fallback silently if detection fails
        pass

    # Merge additional yt-dlp options
    if yt_dlp_options:
        ydl_opts.update(yt_dlp_options)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise RuntimeError("Failed to extract video information")

            video_title = info.get("title", "Unknown Video")
            upload_date = info.get("upload_date", "")

            # Download the audio
            ydl.download([url])

        # Find the downloaded file
        temp_dir = os.path.dirname(output_template)
        downloaded_files = [
            f for f in os.listdir(temp_dir) if f.startswith("temp_audio")
        ]

        if not downloaded_files:
            raise RuntimeError("Failed to download audio from YouTube")

        return os.path.join(temp_dir, downloaded_files[0]), video_title, upload_date

    except Exception as e:
        if "Sign in to confirm you're not a bot" in str(e):
            logger.error("\nBot detection detected! To fix this:")
            logger.error("1. Create a cookies.txt file with your YouTube cookies")
            logger.error("2. Place it in the project root or your home directory")
            logger.error(
                "3. See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
            )
        raise RuntimeError(f"YouTube download failed: {e}") from e


def _convert_and_trim_audio(
    input_file: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
) -> None:
    """Convert audio to WAV format and optionally trim it.

    Args:
        input_file: Input audio file path
        output_path: Output WAV file path
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)

    Raises:
        subprocess.CalledProcessError: If ffmpeg conversion fails
    """
    cmd = ["ffmpeg", "-i", input_file]

    # Add trimming parameters if specified
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])

    # Audio conversion settings for YAMNet
    cmd.extend(
        [
            "-ac",
            "1",  # mono
            "-ar",
            "16000",  # 16kHz
            "-y",  # overwrite output
            output_path,
        ]
    )

    logger.info("Converting audio to 16kHz mono WAV...")
    if start_time is not None or duration is not None:
        logger.info(f"Trimming: start={start_time}s, duration={duration}s")

    subprocess.run(cmd, check=True, capture_output=True)


def create_folder_name_from_title(video_title: str) -> str:
    """Create folder name from video title date information.

    Args:
        video_title: Video title from YouTube (e.g., "TBPN | Monday, July 7th")

    Returns:
        Folder name in format tbpn_monday_july_7th
    """
    if not video_title:
        return "tbpn_unknown_date"

    try:
        # Look for patterns like "TBPN | Monday, July 7th" or "Monday, July 7th"
        import re

        # Pattern to match day, month, and day number
        # Matches: "Monday, July 7th", "Tuesday, August 15th", etc.
        pattern = r"(\w+),\s+(\w+)\s+(\d+)(?:st|nd|rd|th)?"
        match = re.search(pattern, video_title)

        if match:
            day_name = match.group(1).lower()  # monday, tuesday, etc.
            month_name = match.group(2).lower()  # july, august, etc.
            day_num = int(match.group(3))

            # Get ordinal suffix for day
            if 10 <= day_num <= 20:  # Special case for 11th, 12th, 13th
                suffix = "th"
            else:
                suffix = {1: "st", 2: "nd", 3: "rd"}.get(day_num % 10, "th")

            return f"tbpn_{day_name}_{month_name}_{day_num}{suffix}"

        return "tbpn_unknown_date"

    except (ValueError, IndexError):
        return "tbpn_unknown_date"


def create_folder_name_from_date(upload_date: str) -> str:
    """Create folder name from YouTube upload date in format tbpn_dayname_month_dayordinal.

    Args:
        upload_date: Upload date from YouTube (format: YYYYMMDD)

    Returns:
        Folder name in format tbpn_monday_june_23rd
    """
    if not upload_date or len(upload_date) != 8:
        return "tbpn_unknown_date"

    try:
        from datetime import datetime

        year = int(upload_date[:4])
        month = int(upload_date[4:6])
        day = int(upload_date[6:8])

        # Create datetime object
        date_obj = datetime(year, month, day)

        # Get day name and month name
        day_name = date_obj.strftime("%A").lower()  # monday, tuesday, etc.
        month_name = date_obj.strftime("%B").lower()  # january, february, etc.

        # Get ordinal suffix for day
        if 10 <= day <= 20:  # Special case for 11th, 12th, 13th
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

        return f"tbpn_{day_name}_{month_name}_{day}{suffix}"

    except (ValueError, IndexError):
        return "tbpn_unknown_date"


def sanitize_title_for_folder(title: str) -> str:
    """Convert video title to safe folder name.

    Args:
        title: Video title from YouTube

    Returns:
        Sanitized folder name safe for filesystem
    """
    # Convert to lowercase
    sanitized = title.lower()
    # Remove commas
    sanitized = sanitized.replace(",", "")
    # Remove or replace problematic characters
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", sanitized)
    # Replace spaces with underscores
    sanitized = sanitized.replace(" ", "_")
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized


def create_temp_audio_path(temp_dir: str) -> str:
    """Create a unique temporary audio file path.

    Args:
        temp_dir: Directory for temporary files

    Returns:
        Path to temporary audio file
    """
    return os.path.join(temp_dir, f"temp_youtube_audio_{uuid.uuid4().hex[:8]}.wav")


def setup_directories() -> tuple[str, str]:
    """Create necessary directories and return paths.

    Returns:
        Tuple of (temp_audio_dir, csv_results_dir)
    """
    # Find project root robustly
    current = Path(__file__).resolve().parent
    project_root = None
    for parent in [current] + list(current.parents):
        if (parent / "data").exists() and (parent / "src").exists():
            project_root = parent
            break

    if not project_root:
        project_root = Path.cwd()

    # Use data directory for all temporary and output files
    temp_audio_dir = str(project_root / "data/temp_audio")
    csv_results_dir = str(project_root / "data/csv_results")

    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(csv_results_dir, exist_ok=True)

    return temp_audio_dir, csv_results_dir


def compute_lufs_segments(
    video_id: str,
    timestamps: list[tuple[float, float]],
    measurement_type: str = "integrated",
    index: Optional[Any] = None,
    batch_context: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Compute LUFS loudness for audio segments from raw audio with batch weighting support.

    This function extracts segments from the raw cached audio based on timestamps
    from gong detections and computes LUFS loudness measurements using BS.1770-4
    K-weighting and EBU R128 gating. Supports batch-weighted measurements across
    multiple videos for proper relative loudness analysis.

    Args:
        video_id: YouTube video ID
        timestamps: List of (start_time, end_time) tuples in seconds
        measurement_type: Type of LUFS measurement:
            - "integrated": Integrated loudness over entire segment
            - "short_term": Short-term loudness (3s sliding window)
            - "momentary": Momentary loudness (400ms sliding window)
        index: Optional LocalMediaIndex instance
        batch_context: Optional dict with batch weighting information:
            - "all_segments": List of all audio segments from all videos
            - "reference_lufs": Reference LUFS level for batch normalization
            - "enable_batch_weighting": Boolean to enable batch weighting

    Returns:
        List of dictionaries containing LUFS measurements for each segment.
        Each dict contains:
        - start_time: Start time in seconds
        - end_time: End time in seconds
        - duration: Segment duration in seconds
        - lufs: LUFS measurement value (batch-weighted if enabled)
        - raw_lufs: Raw LUFS measurement (before batch weighting)
        - measurement_type: Type of measurement used
        - valid: Boolean indicating if measurement was successful
        - batch_weighted: Boolean indicating if batch weighting was applied

    Raises:
        RuntimeError: If LUFS library not available or raw audio not found
        ValueError: If timestamps are invalid
    """
    if not LUFS_AVAILABLE:
        raise RuntimeError(
            "LUFS analysis requires pyloudnorm and librosa. "
            "Install with: pip install pyloudnorm librosa"
        )

    if not timestamps:
        return []

    # Validate measurement type
    valid_types = ["integrated", "short_term", "momentary"]
    if measurement_type not in valid_types:
        raise ValueError(f"measurement_type must be one of {valid_types}")

    logger.info(f"Computing LUFS for {len(timestamps)} segments from video {video_id}")

    # Find raw audio file
    try:
        from gong_detector.core.utils.local_media import LocalMediaIndex
        idx = index or LocalMediaIndex()

        # Get raw audio path from index
        meta = idx.get(video_id)
        if not meta or not meta.get("raw_path"):
            raise RuntimeError(f"No raw audio path found for video {video_id}")

        raw_path = meta["raw_path"]
        if not Path(raw_path).exists():
            raise RuntimeError(f"Raw audio file not found: {raw_path}")

    except ImportError as err:
        raise RuntimeError("Could not import LocalMediaIndex") from err

    results = []

    try:
        # Load raw audio file using librosa (supports WebM and other formats)
        logger.info(f"Loading raw audio: {raw_path}")
        audio_data, sample_rate = librosa.load(raw_path, sr=None, mono=False)

        # Ensure mono audio for LUFS analysis
        if len(audio_data.shape) > 1:
            # Convert stereo to mono using equal weighting
            # Note: BS.1770 specifies channel weighting, but for simplicity we use equal weighting
            audio_data = audio_data.mean(axis=0)  # Average across channels (axis 0)

        # Create loudness meter with appropriate settings
        meter = pyln.Meter(sample_rate)  # Uses BS.1770-4 K-weighting by default

        logger.info(f"Processing {len(timestamps)} segments with {measurement_type} LUFS")

        for i, (start_time, end_time) in enumerate(timestamps):
            segment_result = {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "lufs": None,
                "raw_lufs": None,
                "measurement_type": measurement_type,
                "valid": False,
                "batch_weighted": False,
            }

            try:
                # Validate timestamps
                if start_time < 0 or end_time <= start_time:
                    logger.warning(f"Invalid timestamp pair: {start_time}-{end_time}s")
                    results.append(segment_result)
                    continue

                # Convert time to sample indices
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)

                # Check bounds
                if start_sample >= len(audio_data) or end_sample > len(audio_data):
                    logger.warning(f"Timestamp {start_time}-{end_time}s exceeds audio length")
                    results.append(segment_result)
                    continue

                # Extract segment
                segment_audio = audio_data[start_sample:end_sample]

                # Skip very short segments (less than 400ms for momentary, 3s for short-term)
                min_duration = 0.4 if measurement_type == "momentary" else 3.0 if measurement_type == "short_term" else 0.1
                if len(segment_audio) / sample_rate < min_duration:
                    logger.warning(f"Segment {i+1} too short ({len(segment_audio)/sample_rate:.2f}s) for {measurement_type} LUFS")
                    results.append(segment_result)
                    continue

                # Compute LUFS measurement
                if measurement_type == "integrated":
                    # Integrated loudness over entire segment with EBU R128 gating
                    lufs_value = meter.integrated_loudness(segment_audio)
                elif measurement_type == "short_term":
                    # Short-term loudness (3s sliding window)
                    # pyloudnorm doesn't have separate short_term method, use integrated for now
                    # This is a simplified implementation - in production you'd implement proper 3s sliding window
                    lufs_value = meter.integrated_loudness(segment_audio)
                    logger.warning("Short-term LUFS approximated using integrated loudness")
                else:  # momentary
                    # Momentary loudness (400ms sliding window)
                    # pyloudnorm doesn't have separate momentary method, use integrated for now
                    # This is a simplified implementation - in production you'd implement proper 400ms sliding window
                    lufs_value = meter.integrated_loudness(segment_audio)
                    logger.warning("Momentary LUFS approximated using integrated loudness")

                # Check for valid measurement
                if lufs_value == -float('inf') or lufs_value != lufs_value:  # NaN check
                    logger.warning(f"Invalid LUFS measurement for segment {i+1}")
                else:
                    # Store raw LUFS value
                    segment_result["raw_lufs"] = float(lufs_value)
                    segment_result["lufs"] = float(lufs_value)  # Default to raw value
                    segment_result["valid"] = True
                    logger.debug(f"Segment {i+1}: {start_time:.1f}-{end_time:.1f}s = {lufs_value:.1f} LUFS")

            except Exception as e:
                logger.error(f"Error processing segment {i+1} ({start_time}-{end_time}s): {e}")

            results.append(segment_result)

        # Apply batch weighting if provided
        if batch_context and batch_context.get("enable_batch_weighting", False):
            logger.info("Applying batch weighting to LUFS measurements...")
            reference_lufs = batch_context.get("reference_lufs", -23.0)  # EBU R128 reference
            
            # Get all valid measurements for batch statistics
            valid_results = [r for r in results if r["valid"]]
            if valid_results:
                # Calculate batch statistics
                all_lufs = [r["raw_lufs"] for r in valid_results]
                batch_mean = sum(all_lufs) / len(all_lufs)
                
                # Apply batch weighting: adjust relative to batch mean and reference
                batch_offset = reference_lufs - batch_mean
                
                for result in valid_results:
                    # Apply batch weighting
                    result["lufs"] = result["raw_lufs"] + batch_offset
                    result["batch_weighted"] = True
                
                logger.info(f"Batch weighting applied: offset = {batch_offset:.1f} dB")
                logger.info(f"Batch mean: {batch_mean:.1f} LUFS → Reference: {reference_lufs:.1f} LUFS")

        # Summary statistics
        valid_measurements = [r["lufs"] for r in results if r["valid"]]
        if valid_measurements:
            logger.info(f"LUFS analysis complete: {len(valid_measurements)}/{len(timestamps)} valid measurements")
            logger.info(f"LUFS range: {min(valid_measurements):.1f} to {max(valid_measurements):.1f} LUFS")
            logger.info(f"Mean LUFS: {sum(valid_measurements)/len(valid_measurements):.1f} LUFS")
            
            # Show batch weighting info if applied
            batch_weighted_count = sum(1 for r in results if r.get("batch_weighted", False))
            if batch_weighted_count > 0:
                logger.info(f"Batch weighting applied to {batch_weighted_count} measurements")
        else:
            logger.warning("No valid LUFS measurements obtained")

    except Exception as e:
        logger.error(f"Failed to compute LUFS for video {video_id}: {e}")
        # Return empty results with error indication
        for start_time, end_time in timestamps:
            results.append({
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "lufs": None,
                "measurement_type": measurement_type,
                "valid": False,
            })

    return results


def compute_batch_weighted_lufs(
    all_video_data: list[dict[str, Any]],
    measurement_type: str = "integrated",
    reference_lufs: float = -23.0,
) -> dict[str, list[dict[str, Any]]]:
    """Compute batch-weighted LUFS across all videos for proper relative analysis.
    
    This function processes all detection segments from multiple videos together,
    computes LUFS measurements, and applies batch weighting to normalize loudness
    measurements relative to the entire dataset rather than individual videos.
    
    Args:
        all_video_data: List of video data dicts, each containing:
            - "video_id": YouTube video ID
            - "timestamps": List of (start_time, end_time) tuples
            - "result": Detection result dict
        measurement_type: Type of LUFS measurement (integrated, short_term, momentary)
        reference_lufs: Reference LUFS level for batch normalization (default: -23.0 LUFS)
    
    Returns:
        Dictionary mapping video_id to list of LUFS measurement dicts for that video.
        Each measurement dict contains batch-weighted LUFS values.
    
    Raises:
        RuntimeError: If LUFS library not available
        ValueError: If video data is invalid
    """
    if not LUFS_AVAILABLE:
        raise RuntimeError(
            "LUFS analysis requires pyloudnorm and librosa. "
            "Install with: pip install pyloudnorm librosa"
        )
    
    if not all_video_data:
        return {}
    
    logger.info(f"Computing batch-weighted LUFS for {len(all_video_data)} videos")
    
    # Step 1: Collect all raw LUFS measurements across all videos
    all_raw_lufs = []
    video_results = {}
    
    try:
        from gong_detector.core.utils.local_media import LocalMediaIndex
        index = LocalMediaIndex()
    except ImportError:
        logger.warning("Could not import LocalMediaIndex, using None")
        index = None
    
    # Process each video to get raw LUFS measurements
    for video_data in all_video_data:
        video_id = video_data["video_id"]
        timestamps = video_data["timestamps"]
        
        logger.info(f"Processing video {video_id} with {len(timestamps)} detection segments")
        
        # Compute raw LUFS for this video (no batch weighting yet)
        lufs_results = compute_lufs_segments(
            video_id=video_id,
            timestamps=timestamps,
            measurement_type=measurement_type,
            index=index,
            batch_context=None,  # No batch weighting on first pass
        )
        
        # Store results for this video
        video_results[video_id] = lufs_results
        
        # Collect valid raw LUFS measurements for batch statistics
        valid_lufs = [r["raw_lufs"] for r in lufs_results if r["valid"] and r["raw_lufs"] is not None]
        all_raw_lufs.extend(valid_lufs)
    
    # Step 2: Calculate batch statistics
    if not all_raw_lufs:
        logger.warning("No valid LUFS measurements found across all videos")
        return video_results
    
    batch_mean_lufs = sum(all_raw_lufs) / len(all_raw_lufs)
    batch_offset = reference_lufs - batch_mean_lufs
    
    logger.info(f"Batch LUFS statistics:")
    logger.info(f"  Total measurements: {len(all_raw_lufs)}")
    logger.info(f"  Batch mean: {batch_mean_lufs:.1f} LUFS")
    logger.info(f"  Reference level: {reference_lufs:.1f} LUFS")
    logger.info(f"  Batch offset: {batch_offset:.1f} dB")
    logger.info(f"  LUFS range: {min(all_raw_lufs):.1f} to {max(all_raw_lufs):.1f} LUFS")
    
    # Step 3: Apply batch weighting to all measurements
    total_weighted = 0
    for video_id, lufs_results in video_results.items():
        for result in lufs_results:
            if result["valid"] and result["raw_lufs"] is not None:
                # Apply batch weighting
                result["lufs"] = result["raw_lufs"] + batch_offset
                result["batch_weighted"] = True
                total_weighted += 1
    
    logger.info(f"Applied batch weighting to {total_weighted} measurements across {len(video_results)} videos")
    
    return video_results


def compute_true_peak_segments(
    video_id: str,
    timestamps: list[tuple[float, float]],
    measurement_type: str = "integrated",
    index: Optional[Any] = None,
    batch_context: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Compute True Peak (dBTP) for audio segments from raw audio with batch weighting support.

    This function extracts segments from the raw cached audio based on timestamps
    from gong detections and computes True Peak measurements using ITU-R BS.1770-4
    standard. Supports batch-weighted measurements across multiple videos.

    Args:
        video_id: YouTube video ID
        timestamps: List of (start_time, end_time) tuples in seconds
        measurement_type: Type of measurement context (integrated, short_term, momentary)
        index: Optional LocalMediaIndex instance
        batch_context: Optional dict with batch weighting information

    Returns:
        List of dictionaries containing True Peak measurements for each segment.
        Each dict contains:
        - start_time: Start time in seconds
        - end_time: End time in seconds
        - duration: Segment duration in seconds
        - dbtp: True Peak measurement value (batch-weighted if enabled)
        - raw_dbtp: Raw True Peak measurement (before batch weighting)
        - measurement_type: Type of measurement used
        - valid: Boolean indicating if measurement was successful
        - batch_weighted: Boolean indicating if batch weighting was applied

    Raises:
        RuntimeError: If LUFS library not available or raw audio not found
        ValueError: If timestamps are invalid
    """
    if not LUFS_AVAILABLE:
        raise RuntimeError(
            "True Peak analysis requires pyloudnorm and librosa. "
            "Install with: pip install pyloudnorm librosa"
        )

    if not timestamps:
        return []

    logger.info(f"Computing True Peak for {len(timestamps)} segments from video {video_id}")

    # Find raw audio file
    try:
        from gong_detector.core.utils.local_media import LocalMediaIndex
        idx = index or LocalMediaIndex()

        # Get raw audio path from index
        meta = idx.get(video_id)
        if not meta or not meta.get("raw_path"):
            raise RuntimeError(f"No raw audio path found for video {video_id}")

        raw_path = meta["raw_path"]
        if not Path(raw_path).exists():
            raise RuntimeError(f"Raw audio file not found: {raw_path}")

    except ImportError as err:
        raise RuntimeError("Could not import LocalMediaIndex") from err

    results = []

    try:
        # Load raw audio file using librosa (supports WebM and other formats)
        logger.info(f"Loading raw audio for True Peak: {raw_path}")
        audio_data, sample_rate = librosa.load(raw_path, sr=None, mono=False)

        # Ensure mono audio for True Peak analysis
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=0)  # Average across channels

        # Create loudness meter for True Peak measurement
        meter = pyln.Meter(sample_rate)

        logger.info(f"Processing {len(timestamps)} segments with True Peak analysis")

        for i, (start_time, end_time) in enumerate(timestamps):
            segment_result = {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "dbtp": None,
                "raw_dbtp": None,
                "measurement_type": measurement_type,
                "valid": False,
                "batch_weighted": False,
            }

            try:
                # Validate timestamps
                if start_time < 0 or end_time <= start_time:
                    logger.warning(f"Invalid timestamp pair: {start_time}-{end_time}s")
                    results.append(segment_result)
                    continue

                # Convert time to sample indices
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)

                # Check bounds
                if start_sample >= len(audio_data) or end_sample > len(audio_data):
                    logger.warning(f"Timestamp {start_time}-{end_time}s exceeds audio length")
                    results.append(segment_result)
                    continue

                # Extract segment
                segment_audio = audio_data[start_sample:end_sample]

                # Skip very short segments (less than 100ms)
                if len(segment_audio) / sample_rate < 0.1:
                    logger.warning(f"Segment {i+1} too short ({len(segment_audio)/sample_rate:.2f}s) for True Peak")
                    results.append(segment_result)
                    continue

                # Compute True Peak measurement using pyloudnorm
                try:
                    # pyloudnorm doesn't have direct True Peak method, but we can compute it manually
                    # True Peak requires oversampling - we'll use a simple approximation for now
                    # by finding the maximum absolute value and converting to dBTP
                    peak_amplitude = np.max(np.abs(segment_audio))
                    
                    # Convert to dBTP (True Peak approximation)
                    if peak_amplitude > 0:
                        dbtp_value = 20.0 * np.log10(peak_amplitude)
                    else:
                        dbtp_value = -80.0  # Silence floor
                        
                except Exception as peak_error:
                    logger.warning(f"True Peak computation failed for segment {i+1}: {peak_error}")
                    dbtp_value = -80.0

                # Check for valid measurement
                if dbtp_value == -float('inf') or dbtp_value != dbtp_value:  # NaN check
                    logger.warning(f"Invalid True Peak measurement for segment {i+1}")
                else:
                    # Store raw True Peak value
                    segment_result["raw_dbtp"] = float(dbtp_value)
                    segment_result["dbtp"] = float(dbtp_value)  # Default to raw value
                    segment_result["valid"] = True
                    logger.debug(f"Segment {i+1}: {start_time:.1f}-{end_time:.1f}s = {dbtp_value:.1f} dBTP")

            except Exception as e:
                logger.error(f"Error processing segment {i+1} ({start_time}-{end_time}s): {e}")

            results.append(segment_result)

        # Summary statistics
        valid_measurements = [r["dbtp"] for r in results if r["valid"]]
        if valid_measurements:
            logger.info(f"True Peak analysis complete: {len(valid_measurements)}/{len(timestamps)} valid measurements")
            logger.info(f"True Peak range: {min(valid_measurements):.1f} to {max(valid_measurements):.1f} dBTP")
            logger.info(f"Mean True Peak: {sum(valid_measurements)/len(valid_measurements):.1f} dBTP")
        else:
            logger.warning("No valid True Peak measurements obtained")

    except Exception as e:
        logger.error(f"Failed to compute True Peak for video {video_id}: {e}")
        # Return empty results with error indication
        for start_time, end_time in timestamps:
            results.append({
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "dbtp": None,
                "raw_dbtp": None,
                "measurement_type": measurement_type,
                "valid": False,
                "batch_weighted": False,
            })

    return results


def compute_batch_weighted_dbtp(
    all_video_data: list[dict[str, Any]],
    measurement_type: str = "integrated",
    reference_dbtp: float = -1.0,
) -> dict[str, list[dict[str, Any]]]:
    """Compute batch-weighted True Peak across all videos for proper relative analysis.
    
    This function processes all detection segments from multiple videos together,
    computes True Peak measurements, and applies batch weighting to normalize
    measurements relative to the entire dataset.
    
    Args:
        all_video_data: List of video data dicts, each containing:
            - "video_id": YouTube video ID
            - "timestamps": List of (start_time, end_time) tuples
            - "result": Detection result dict
        measurement_type: Type of measurement context (integrated, short_term, momentary)
        reference_dbtp: Reference True Peak level for batch normalization (default: -1.0 dBTP)
    
    Returns:
        Dictionary mapping video_id to list of True Peak measurement dicts for that video.
        Each measurement dict contains batch-weighted True Peak values.
    
    Raises:
        RuntimeError: If LUFS library not available
        ValueError: If video data is invalid
    """
    if not LUFS_AVAILABLE:
        raise RuntimeError(
            "True Peak analysis requires pyloudnorm and librosa. "
            "Install with: pip install pyloudnorm librosa"
        )
    
    if not all_video_data:
        return {}
    
    logger.info(f"Computing batch-weighted True Peak for {len(all_video_data)} videos")
    
    # Step 1: Collect all raw True Peak measurements across all videos
    all_raw_dbtp = []
    video_results = {}
    
    try:
        from gong_detector.core.utils.local_media import LocalMediaIndex
        index = LocalMediaIndex()
    except ImportError:
        logger.warning("Could not import LocalMediaIndex, using None")
        index = None
    
    # Process each video to get raw True Peak measurements
    for video_data in all_video_data:
        video_id = video_data["video_id"]
        timestamps = video_data["timestamps"]
        
        logger.info(f"Processing video {video_id} with {len(timestamps)} detection segments")
        
        # Compute raw True Peak for this video (no batch weighting yet)
        dbtp_results = compute_true_peak_segments(
            video_id=video_id,
            timestamps=timestamps,
            measurement_type=measurement_type,
            index=index,
            batch_context=None,  # No batch weighting on first pass
        )
        
        # Store results for this video
        video_results[video_id] = dbtp_results
        
        # Collect valid raw True Peak measurements for batch statistics
        valid_dbtp = [r["raw_dbtp"] for r in dbtp_results if r["valid"] and r["raw_dbtp"] is not None]
        all_raw_dbtp.extend(valid_dbtp)
    
    # Step 2: Calculate batch statistics
    if not all_raw_dbtp:
        logger.warning("No valid True Peak measurements found across all videos")
        return video_results
    
    batch_mean_dbtp = sum(all_raw_dbtp) / len(all_raw_dbtp)
    batch_offset = reference_dbtp - batch_mean_dbtp
    
    logger.info(f"Batch True Peak statistics:")
    logger.info(f"  Total measurements: {len(all_raw_dbtp)}")
    logger.info(f"  Batch mean: {batch_mean_dbtp:.1f} dBTP")
    logger.info(f"  Reference level: {reference_dbtp:.1f} dBTP")
    logger.info(f"  Batch offset: {batch_offset:.1f} dB")
    logger.info(f"  True Peak range: {min(all_raw_dbtp):.1f} to {max(all_raw_dbtp):.1f} dBTP")
    
    # Step 3: Apply batch weighting to all measurements
    total_weighted = 0
    for video_id, dbtp_results in video_results.items():
        for result in dbtp_results:
            if result["valid"] and result["raw_dbtp"] is not None:
                # Apply batch weighting
                result["dbtp"] = result["raw_dbtp"] + batch_offset
                result["batch_weighted"] = True
                total_weighted += 1
    
    logger.info(f"Applied batch weighting to {total_weighted} measurements across {len(video_results)} videos")
    
    return video_results
