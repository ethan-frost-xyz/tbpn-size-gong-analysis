"""File system utilities for directory setup and temporary file management.

This module provides functions for setting up project directories,
cleaning up temporary files, and managing file system operations.
"""

import glob
import logging
import os
import time
import uuid
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


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


def create_temp_audio_path(temp_dir: str) -> str:
    """Create a unique temporary audio file path.

    Args:
        temp_dir: Directory for temporary files

    Returns:
        Path to temporary audio file
    """
    return os.path.join(temp_dir, f"temp_youtube_audio_{uuid.uuid4().hex[:8]}.wav")
