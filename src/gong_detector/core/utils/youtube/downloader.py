"""YouTube audio downloading functionality.

This module handles downloading audio from YouTube URLs using yt-dlp,
with support for cookies, retries, and various optimization settings.
"""

import logging
import os
import shutil
import tempfile
from typing import Optional

import yt_dlp  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)


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


def download_youtube_audio(
    url: str, output_template: str, yt_dlp_options: Optional[dict] = None
) -> tuple[str, str, str]:
    """Download audio from YouTube using yt-dlp and extract video metadata.

    Args:
        url: YouTube URL to download
        output_template: Template for output filename
        yt_dlp_options: Additional yt-dlp options to merge

    Returns:
        Tuple of (downloaded_file_path, video_title, upload_date)

    Raises:
        RuntimeError: If download fails
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


def download_and_process_youtube_audio(
    url: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
    yt_dlp_options: Optional[dict] = None,
) -> tuple[str, str, str]:
    """Download YouTube audio and prepare for processing.

    This is a high-level function that coordinates downloading with the
    cache management and audio processing modules.

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

    # Import here to avoid circular imports
    from ..local_media import LocalMediaEntry, LocalMediaIndex
    from .audio_processor import trim_from_preprocessed
    from .cache_manager import ensure_full_preprocessed_from_raw, save_raw_to_cache
    from .metadata_utils import video_id_from_url

    # Extract video ID for caching
    video_id = video_id_from_url(url)
    if not video_id:
        raise RuntimeError(f"Could not extract video ID from URL: {url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio = os.path.join(temp_dir, "temp_audio.%(ext)s")

        # Download with yt-dlp and get video info
        downloaded_file, video_title, upload_date = download_youtube_audio(
            url, temp_audio, yt_dlp_options
        )

        # Save raw audio to cache
        raw_cache_path = save_raw_to_cache(downloaded_file, video_id)

        # Ensure full preprocessed WAV exists
        preprocessed_path = ensure_full_preprocessed_from_raw(raw_cache_path, video_id)

        # Update local media index with both paths
        try:
            idx = LocalMediaIndex()
            entry = LocalMediaEntry(
                video_id=video_id,
                source_url=url,
                video_title=video_title or "",
                upload_date=upload_date or "",
                preprocessed_path=preprocessed_path,
                raw_path=raw_cache_path,
            )
            idx.upsert(entry)
            logger.info(f"Updated local media index for video: {video_id}")
        except Exception as e:
            logger.warning(f"Failed to update local media index: {e}")

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
