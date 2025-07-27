"""YouTube download and file management utilities.

This module provides functions for downloading YouTube audio,
managing temporary files, and handling file system operations
for the gong detection pipeline.
"""

import glob
import os
import re
import subprocess
import tempfile
import time
import uuid
from typing import Optional

import yt_dlp  # type: ignore


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
                print(f"Cleaned up old temp file: {temp_file}")
            except OSError:
                pass  # File might already be gone


def download_and_trim_youtube_audio(
    url: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
    yt_dlp_options: Optional[dict] = None,
) -> tuple[str, str, str]:
    """Download YouTube audio and optionally trim to specified segment.

    Args:
        url: YouTube URL to download
        output_path: Path for output WAV file
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)

    Returns:
        Tuple of (audio_path, video_title, upload_date)

    Raises:
        RuntimeError: If download or conversion fails
    """
    print(f"Downloading audio from: {url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio = os.path.join(temp_dir, "temp_audio.%(ext)s")

        # Download with yt-dlp and get video info
        downloaded_file, video_title, upload_date = _download_youtube_audio(
            url, temp_audio, yt_dlp_options
        )

        # Convert and trim with ffmpeg
        _convert_and_trim_audio(downloaded_file, output_path, start_time, duration)

        print(f"Audio saved to: {output_path}")
        print(f"Video title: {video_title}")

    return output_path, video_title, upload_date


def _download_youtube_audio(url: str, output_template: str, yt_dlp_options: Optional[dict] = None) -> tuple[str, str, str]:
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
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": output_template,
        "quiet": True,  # Reduce output noise
    }

    # Add cookies if available
    cookies_path = get_cookies_path()
    if cookies_path:
        print(f"Using cookies from: {cookies_path}")
        ydl_opts["cookiefile"] = cookies_path
    else:
        print("No cookies file found. If you encounter bot detection, create a cookies.txt file.")
        print("See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp")

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
        downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith("temp_audio")]

        if not downloaded_files:
            raise RuntimeError("Failed to download audio from YouTube")

        return os.path.join(temp_dir, downloaded_files[0]), video_title, upload_date

    except Exception as e:
        if "Sign in to confirm you're not a bot" in str(e):
            print("\nBot detection detected! To fix this:")
            print("1. Create a cookies.txt file with your YouTube cookies")
            print("2. Place it in the project root or your home directory")
            print("3. See: https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp")
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

    print("Converting audio to 16kHz mono WAV...")
    if start_time is not None or duration is not None:
        print(f"Trimming: start={start_time}s, duration={duration}s")

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
        pattern = r'(\w+),\s+(\w+)\s+(\d+)(?:st|nd|rd|th)?'
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
    temp_audio_dir = "temp_audio"
    csv_results_dir = "csv_results"

    os.makedirs(temp_audio_dir, exist_ok=True)
    os.makedirs(csv_results_dir, exist_ok=True)

    return temp_audio_dir, csv_results_dir
