"""Audio processing and trimming utilities.

This module handles audio conversion, trimming operations, and
format standardization for the gong detection pipeline.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)


def trim_from_preprocessed(
    preprocessed_path: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
) -> None:
    """Trim audio from preprocessed WAV file.

    Parameters
    ----------
    preprocessed_path : str
        Path to the cached 16 kHz mono WAV file.
    output_path : str
        Destination for the trimmed output.
    start_time : int, optional
        Start time in seconds for the trimmed segment.
    duration : int, optional
        Duration in seconds for the trimmed segment.

    Raises
    ------
    RuntimeError
        Raised when ffmpeg fails to trim the audio.
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
    cmd.extend(
        [
            "-c",
            "copy",  # Copy without re-encoding
            "-y",  # overwrite
            "-nostdin",  # non-interactive
            "-loglevel",
            "error",  # minimal output
            output_path,
        ]
    )

    try:
        logger.info(f"Trimming preprocessed audio: {start_time}s + {duration}s")
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Trimmed audio saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg trimming failed: {e.stderr}") from e


def convert_and_trim_audio(
    input_file: str,
    output_path: str,
    start_time: Optional[int] = None,
    duration: Optional[int] = None,
) -> None:
    """Convert audio to WAV format and optionally trim it.

    Parameters
    ----------
    input_file : str
        Source audio or video path.
    output_path : str
        Destination for the converted WAV file.
    start_time : int, optional
        Start time in seconds for trimming.
    duration : int, optional
        Duration in seconds for trimming.

    Raises
    ------
    subprocess.CalledProcessError
        Raised when ffmpeg fails to convert or trim the audio.
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
