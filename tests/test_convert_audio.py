"""Simple tests for convert_audio module."""

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest  # type: ignore

from gong_detector.convert_audio import convert_youtube_audio

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture  # type: ignore


def test_convert_local_file(tmp_path: Path, mocker: "MockerFixture") -> None:
    """Test converting a local file."""
    # Create test file
    input_file = tmp_path / "test.mp3"
    input_file.write_text("dummy")
    output_file = tmp_path / "output.wav"
    
    # Mock ffmpeg
    mock_run = mocker.patch("subprocess.run")
    
    # Test conversion
    result = convert_youtube_audio(str(input_file), str(output_file))
    
    assert result == str(output_file)
    mock_run.assert_called_once()


def test_convert_youtube_url(tmp_path: Path, mocker: "MockerFixture") -> None:
    """Test converting YouTube URL."""
    output_file = tmp_path / "output.wav"
    temp_file = "temp_video.mp3"
    
    # Mock YouTube download
    mock_ydl = mocker.patch("yt_dlp.YoutubeDL")
    mock_ydl.return_value.__enter__.return_value.extract_info.return_value = {}
    mock_ydl.return_value.__enter__.return_value.prepare_filename.return_value = temp_file
    
    # Mock ffmpeg
    mock_run = mocker.patch("subprocess.run")
    
    # Mock file operations
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.remove")
    
    # Test conversion
    result = convert_youtube_audio("https://youtube.com/watch?v=test", str(output_file))
    
    assert result == str(output_file)
    mock_run.assert_called_once() 