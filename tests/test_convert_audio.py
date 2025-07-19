"""Tests for the convert_audio module."""

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest  # type: ignore

from gong_detector.convert_audio import convert_youtube_audio, _convert_to_wav

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # type: ignore
    from _pytest.fixtures import FixtureRequest  # type: ignore
    from _pytest.logging import LogCaptureFixture  # type: ignore
    from _pytest.monkeypatch import MonkeyPatch  # type: ignore
    from pytest_mock.plugin import MockerFixture  # type: ignore


class TestConvertYoutubeAudio:
    """Test cases for convert_youtube_audio function."""

    def test_convert_local_file_success(
        self, tmp_path: Path, mocker: "MockerFixture"
    ) -> None:
        """Test successful conversion of a local audio file."""
        # Create a temporary input file
        input_file = tmp_path / "test_input.mp3"
        input_file.write_text("dummy audio content")
        
        output_file = tmp_path / "test_output.wav"
        
        # Mock the ffmpeg conversion
        mock_convert = mocker.patch(
            "gong_detector.convert_audio._convert_to_wav"
        )
        
        # Call the function
        result = convert_youtube_audio(str(input_file), str(output_file))
        
        # Verify the result
        assert result == str(output_file)
        mock_convert.assert_called_once_with(str(input_file), str(output_file))

    def test_convert_youtube_url_success(
        self, tmp_path: Path, mocker: "MockerFixture"
    ) -> None:
        """Test successful conversion of a YouTube URL."""
        output_file = tmp_path / "test_output.wav"
        temp_mp3 = tmp_path / "temp_download.mp3"
        temp_mp3.write_text("dummy audio content")
        
        # Mock the YouTube download
        mock_download = mocker.patch(
            "gong_detector.convert_audio._download_youtube_audio",
            return_value=str(temp_mp3)
        )
        
        # Mock the ffmpeg conversion
        mock_convert = mocker.patch(
            "gong_detector.convert_audio._convert_to_wav"
        )
        
        # Mock os.remove to prevent actual file deletion
        mock_remove = mocker.patch("os.remove")
        
        # Call the function with a YouTube URL
        youtube_url = "https://www.youtube.com/watch?v=test123"
        result = convert_youtube_audio(youtube_url, str(output_file))
        
        # Verify the result
        assert result == str(output_file)
        mock_download.assert_called_once_with(youtube_url)
        mock_convert.assert_called_once_with(str(temp_mp3), str(output_file))
        mock_remove.assert_called_once_with(str(temp_mp3))

    def test_convert_nonexistent_local_file_error(self, tmp_path: Path) -> None:
        """Test error handling when local file doesn't exist."""
        nonexistent_file = tmp_path / "nonexistent.mp3"
        output_file = tmp_path / "test_output.wav"
        
        # Mock _download_youtube_audio to raise an error for invalid URL
        with patch(
            "gong_detector.convert_audio._download_youtube_audio"
        ) as mock_download:
            mock_download.side_effect = RuntimeError("Invalid URL or download failed")
            
            with pytest.raises(RuntimeError, match="Conversion failed"):
                convert_youtube_audio(str(nonexistent_file), str(output_file))

    def test_convert_default_output_path(
        self, tmp_path: Path, mocker: "MockerFixture", monkeypatch: "MonkeyPatch"
    ) -> None:
        """Test conversion with default output path."""
        # Change to the temporary directory
        monkeypatch.chdir(tmp_path)
        
        # Create a temporary input file
        input_file = tmp_path / "test_input.mp3"
        input_file.write_text("dummy audio content")
        
        # Mock the ffmpeg conversion
        mock_convert = mocker.patch(
            "gong_detector.convert_audio._convert_to_wav"
        )
        
        # Call the function with default output path
        result = convert_youtube_audio(str(input_file))
        
        # Verify the result uses default filename
        expected_path = str(tmp_path / "audio.wav")
        assert result == expected_path
        mock_convert.assert_called_once_with(str(input_file), expected_path)


class TestConvertToWav:
    """Test cases for _convert_to_wav function."""

    def test_convert_to_wav_success(self, tmp_path: Path, mocker: "MockerFixture") -> None:
        """Test successful WAV conversion using ffmpeg."""
        input_file = tmp_path / "input.mp3"
        output_file = tmp_path / "output.wav"
        
        # Mock ffmpeg chain
        mock_ffmpeg = mocker.patch("gong_detector.convert_audio.ffmpeg")
        mock_input = Mock()
        mock_output = Mock()
        mock_run = Mock()
        
        mock_ffmpeg.input.return_value = mock_input
        mock_input.output.return_value = mock_output
        mock_output.run = mock_run
        
        # Call the function
        _convert_to_wav(str(input_file), str(output_file))
        
        # Verify ffmpeg was called correctly
        mock_ffmpeg.input.assert_called_once_with(str(input_file))
        mock_input.output.assert_called_once_with(
            str(output_file),
            acodec="pcm_s16le",
            ac=1,
            ar=16000,
            y=None
        )
        mock_run.assert_called_once_with(quiet=True, overwrite_output=True)

    def test_convert_to_wav_ffmpeg_error(
        self, tmp_path: Path, mocker: "MockerFixture"
    ) -> None:
        """Test error handling when ffmpeg conversion fails."""
        input_file = tmp_path / "input.mp3"
        output_file = tmp_path / "output.wav"
        
        # Mock ffmpeg to raise an error
        mock_ffmpeg = mocker.patch("gong_detector.convert_audio.ffmpeg")
        mock_ffmpeg.Error = Exception  # Mock the ffmpeg.Error class
        
        mock_input = Mock()
        mock_output = Mock()
        mock_run = Mock(side_effect=Exception("FFmpeg failed"))
        
        mock_ffmpeg.input.return_value = mock_input
        mock_input.output.return_value = mock_output
        mock_output.run = mock_run
        
        # Call the function and expect an error
        with pytest.raises(RuntimeError, match="FFmpeg conversion failed"):
            _convert_to_wav(str(input_file), str(output_file)) 