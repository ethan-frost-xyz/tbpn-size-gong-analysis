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
) -> tuple[str, str]:
    """Download YouTube audio and optionally trim to specified segment.

    Args:
        url: YouTube URL to download
        output_path: Path for output WAV file
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)

    Returns:
        Tuple of (audio_path, video_title)

    Raises:
        RuntimeError: If download or conversion fails
    """
    print(f"Downloading audio from: {url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_audio = os.path.join(temp_dir, "temp_audio.%(ext)s")

        # Download with yt-dlp and get video info
        downloaded_file, video_title = _download_youtube_audio(url, temp_audio)

        # Convert and trim with ffmpeg
        _convert_and_trim_audio(downloaded_file, output_path, start_time, duration)

        print(f"Audio saved to: {output_path}")
        print(f"Video title: {video_title}")

    return output_path, video_title


def _download_youtube_audio(url: str, output_template: str) -> tuple[str, str]:
    """Download audio from YouTube using yt-dlp and extract video title.

    Args:
        url: YouTube URL to download
        output_template: Template for output filename

    Returns:
        Tuple of (downloaded_file_path, video_title)
    """
    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": output_template,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Get video info first
        info = ydl.extract_info(url, download=False)
        video_title = info.get("title", "Unknown Video")

        # Download the audio
        ydl.download([url])

    # Find the downloaded file
    temp_dir = os.path.dirname(output_template)
    downloaded_files = [f for f in os.listdir(temp_dir) if f.startswith("temp_audio")]

    if not downloaded_files:
        raise RuntimeError("Failed to download audio from YouTube")

    return os.path.join(temp_dir, downloaded_files[0]), video_title


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
    # Replace multiple spaces/underscores with single underscore
    sanitized = re.sub(r"[_\s]+", "_", sanitized)
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