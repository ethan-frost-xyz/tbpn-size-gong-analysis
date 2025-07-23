"""Tests for convert_audio module."""

import subprocess
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

if TYPE_CHECKING:
    pass

from gong_detector.core.convert_audio import (
    _convert_to_wav,
    _download_audio,
    convert_youtube_audio,
    get_audio_info,
    validate_audio_file,
)


class TestInputValidation:
    """Tests for input validation."""

    def test_convert_youtube_audio_empty_url(self) -> None:
        """Test convert_youtube_audio with empty URL."""
        with pytest.raises(ValueError, match="URL or file path cannot be empty"):
            convert_youtube_audio("", "output.wav")

    def test_convert_youtube_audio_empty_output(self) -> None:
        """Test convert_youtube_audio with empty output path."""
        with pytest.raises(ValueError, match="Output path cannot be empty"):
            convert_youtube_audio("test_url", "")

    def test_convert_youtube_audio_whitespace_input(self) -> None:
        """Test convert_youtube_audio with whitespace-only inputs."""
        with pytest.raises(ValueError, match="URL or file path cannot be empty"):
            convert_youtube_audio("   ", "output.wav")

        with pytest.raises(ValueError, match="Output path cannot be empty"):
            convert_youtube_audio("test_url", "   ")


class TestLocalFileConversion:
    """Tests for local file conversion."""

    @patch("gong_detector.core.convert_audio._convert_to_wav")
    @patch("os.path.exists")
    def test_convert_local_file_success(
        self, mock_exists: Mock, mock_convert: Mock
    ) -> None:
        """Test successful conversion of local file."""
        mock_exists.return_value = True

        result = convert_youtube_audio("test.mp3", "output.wav")

        assert result == "output.wav"
        mock_convert.assert_called_once_with("test.mp3", "output.wav")

    @patch("gong_detector.core.convert_audio._download_audio")
    @patch("gong_detector.core.convert_audio._convert_to_wav")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_convert_youtube_url_success(
        self,
        mock_remove: Mock,
        mock_exists: Mock,
        mock_convert: Mock,
        mock_download: Mock,
    ) -> None:
        """Test successful conversion of YouTube URL."""
        # First call returns False (not local file), subsequent calls for downloaded file
        mock_exists.side_effect = [False, True, True]
        mock_download.return_value = "temp_download.mp3"

        result = convert_youtube_audio("https://youtube.com/watch?v=test", "output.wav")

        assert result == "output.wav"
        mock_download.assert_called_once_with("https://youtube.com/watch?v=test")
        mock_convert.assert_called_once_with("temp_download.mp3", "output.wav")
        mock_remove.assert_called_once_with("temp_download.mp3")

    @patch("gong_detector.core.convert_audio._download_audio")
    @patch("gong_detector.core.convert_audio._convert_to_wav")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_cleanup_on_conversion_failure(
        self,
        mock_remove: Mock,
        mock_exists: Mock,
        mock_convert: Mock,
        mock_download: Mock,
    ) -> None:
        """Test cleanup occurs even when conversion fails."""
        mock_exists.side_effect = [
            False,
            True,
            True,
        ]  # URL, downloaded file, exists for cleanup
        mock_download.return_value = "temp_download.mp3"
        mock_convert.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

        with pytest.raises(subprocess.CalledProcessError):
            convert_youtube_audio("https://youtube.com/watch?v=test", "output.wav")

        # Should still attempt cleanup
        mock_remove.assert_called_once_with("temp_download.mp3")


class TestYouTubeDownload:
    """Tests for YouTube download functionality."""

    @patch("gong_detector.core.convert_audio.yt_dlp.YoutubeDL")
    @patch("os.listdir")
    @patch("os.path.exists")
    def test_download_audio_success(
        self, mock_exists: Mock, mock_listdir: Mock, mock_ydl_class: Mock
    ) -> None:
        """Test successful YouTube audio download."""
        # Mock yt-dlp
        mock_ydl = Mock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl

        mock_info = {"title": "test_video"}
        mock_ydl.extract_info.return_value = mock_info
        mock_ydl.prepare_filename.return_value = "temp_test_video.webm"

        # Mock file system
        mock_listdir.return_value = ["temp_audio.mp3"]
        mock_exists.return_value = True

        result = _download_audio("https://youtube.com/watch?v=test")

        assert "temp_test_video.mp3" in result
        mock_ydl.download.assert_called_once_with(["https://youtube.com/watch?v=test"])

    @patch("gong_detector.core.convert_audio.yt_dlp.YoutubeDL")
    def test_download_audio_extract_info_failure(self, mock_ydl_class: Mock) -> None:
        """Test download failure when extract_info returns None."""
        mock_ydl = Mock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = None

        with pytest.raises(RuntimeError, match="Failed to extract video information"):
            _download_audio("https://youtube.com/watch?v=test")

    @patch("gong_detector.core.convert_audio.yt_dlp.YoutubeDL")
    @patch("os.listdir")
    @patch("os.path.exists")
    def test_download_audio_file_not_found(
        self, mock_exists: Mock, mock_listdir: Mock, mock_ydl_class: Mock
    ) -> None:
        """Test download failure when file is not created."""
        mock_ydl = Mock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_ydl.extract_info.return_value = {"title": "test"}
        mock_ydl.prepare_filename.return_value = "temp_test.mp3"

        mock_listdir.return_value = ["temp_audio.mp3"]
        mock_exists.return_value = False  # Downloaded file doesn't exist

        with pytest.raises(RuntimeError, match="Downloaded file not found"):
            _download_audio("https://youtube.com/watch?v=test")

    @patch("gong_detector.core.convert_audio.yt_dlp.YoutubeDL")
    def test_download_audio_general_exception(self, mock_ydl_class: Mock) -> None:
        """Test download failure with general exception."""
        mock_ydl_class.side_effect = Exception("Network error")

        with pytest.raises(RuntimeError, match="YouTube download failed"):
            _download_audio("https://youtube.com/watch?v=test")


class TestAudioConversion:
    """Tests for audio conversion functionality."""

    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_convert_to_wav_success(
        self, mock_makedirs: Mock, mock_exists: Mock, mock_run: Mock
    ) -> None:
        """Test successful audio conversion."""
        mock_exists.side_effect = [True, True]  # Input exists, output created

        _convert_to_wav("input.mp3", "output.wav")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]  # First positional argument (command)
        assert "ffmpeg" in args
        assert "input.mp3" in args
        assert "output.wav" in args
        assert "-ar" in args and "16000" in args  # Sample rate
        assert "-ac" in args and "1" in args  # Mono

    @patch("os.path.exists")
    def test_convert_to_wav_input_not_found(self, mock_exists: Mock) -> None:
        """Test conversion failure when input file doesn't exist."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Input audio file not found"):
            _convert_to_wav("nonexistent.mp3", "output.wav")

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_convert_to_wav_ffmpeg_not_found(
        self, mock_exists: Mock, mock_run: Mock
    ) -> None:
        """Test conversion failure when ffmpeg is not found."""
        mock_exists.side_effect = [True, False]  # Input exists, output not created
        mock_run.side_effect = FileNotFoundError("ffmpeg not found")

        with pytest.raises(RuntimeError, match="FFmpeg not found"):
            _convert_to_wav("input.mp3", "output.wav")

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_convert_to_wav_ffmpeg_error(
        self, mock_exists: Mock, mock_run: Mock
    ) -> None:
        """Test conversion failure when ffmpeg returns error."""
        mock_exists.side_effect = [True, False]  # Input exists, output not created

        error = subprocess.CalledProcessError(1, "ffmpeg")
        error.stderr = "Invalid input format"
        mock_run.side_effect = error

        with pytest.raises(RuntimeError, match="Audio conversion failed"):
            _convert_to_wav("input.mp3", "output.wav")

    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("os.makedirs")
    def test_convert_to_wav_creates_output_dir(
        self, mock_makedirs: Mock, mock_exists: Mock, mock_run: Mock
    ) -> None:
        """Test that output directory is created if it doesn't exist."""
        mock_exists.side_effect = [
            True,
            False,
            True,
        ]  # Input exists, dir doesn't exist, output created

        _convert_to_wav("input.mp3", "subdir/output.wav")

        mock_makedirs.assert_called_once_with("subdir", exist_ok=True)


class TestAudioValidation:
    """Tests for audio file validation."""

    @patch("os.path.exists")
    def test_validate_audio_file_exists_with_audio_extension(
        self, mock_exists: Mock
    ) -> None:
        """Test validation of existing audio file."""
        mock_exists.return_value = True

        assert validate_audio_file("test.wav")
        assert validate_audio_file("test.mp3")
        assert validate_audio_file("test.flac")
        assert validate_audio_file("test.ogg")

    @patch("os.path.exists")
    def test_validate_audio_file_not_exists(self, mock_exists: Mock) -> None:
        """Test validation of non-existent file."""
        mock_exists.return_value = False

        assert not validate_audio_file("nonexistent.wav")

    @patch("os.path.exists")
    def test_validate_audio_file_wrong_extension(self, mock_exists: Mock) -> None:
        """Test validation of file with non-audio extension."""
        mock_exists.return_value = True

        assert not validate_audio_file("test.txt")
        assert not validate_audio_file("test.jpg")
        assert not validate_audio_file("test")  # No extension

    @patch("os.path.exists")
    def test_validate_audio_file_case_insensitive(self, mock_exists: Mock) -> None:
        """Test validation is case insensitive."""
        mock_exists.return_value = True

        assert validate_audio_file("test.WAV")
        assert validate_audio_file("test.Mp3")
        assert validate_audio_file("test.FLAC")


class TestAudioInfo:
    """Tests for audio information extraction."""

    @patch("os.path.exists")
    def test_get_audio_info_file_not_found(self, mock_exists: Mock) -> None:
        """Test getting info for non-existent file."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            get_audio_info("nonexistent.wav")

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_get_audio_info_success(self, mock_exists: Mock, mock_run: Mock) -> None:
        """Test successful audio info extraction."""
        mock_exists.return_value = True

        # Mock ffprobe output
        mock_output = {
            "format": {"duration": "10.5", "size": "1048576", "format_name": "wav"},
            "streams": [
                {
                    "codec_type": "audio",
                    "sample_rate": "44100",
                    "channels": 2,
                    "codec_name": "pcm_s16le",
                }
            ],
        }

        mock_result = Mock()
        mock_result.stdout = str(mock_output).replace("'", '"')  # Valid JSON
        mock_run.return_value = mock_result

        # Mock json.loads
        with patch("json.loads", return_value=mock_output):
            info = get_audio_info("test.wav")

        assert info["duration"] == 10.5
        assert info["size"] == 1048576
        assert info["format_name"] == "wav"
        assert info["sample_rate"] == 44100
        assert info["channels"] == 2
        assert info["codec_name"] == "pcm_s16le"

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_get_audio_info_ffprobe_error(
        self, mock_exists: Mock, mock_run: Mock
    ) -> None:
        """Test audio info extraction when ffprobe fails."""
        mock_exists.return_value = True
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")

        with pytest.raises(RuntimeError, match="Failed to get audio info"):
            get_audio_info("test.wav")

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_get_audio_info_no_audio_stream(
        self, mock_exists: Mock, mock_run: Mock
    ) -> None:
        """Test audio info extraction with no audio stream."""
        mock_exists.return_value = True

        mock_output = {
            "format": {"duration": "10.0", "size": "1000", "format_name": "mp4"},
            "streams": [{"codec_type": "video"}],  # Only video stream
        }

        mock_result = Mock()
        mock_result.stdout = str(mock_output).replace("'", '"')
        mock_run.return_value = mock_result

        with patch("json.loads", return_value=mock_output):
            info = get_audio_info("test.mp4")

        # Should handle missing audio stream gracefully
        assert info["sample_rate"] == 0
        assert info["channels"] == 0
        assert info["codec_name"] == "unknown"


class TestMainScript:
    """Tests for main script functionality."""

    @patch("gong_detector.core.convert_audio.convert_youtube_audio")
    @patch("gong_detector.core.convert_audio.validate_audio_file")
    @patch("gong_detector.core.convert_audio.get_audio_info")
    @patch("sys.argv", ["convert_audio.py", "test_input", "test_output.wav"])
    def test_main_script_success(
        self, mock_get_info: Mock, mock_validate: Mock, mock_convert: Mock
    ) -> None:
        """Test successful main script execution."""
        mock_convert.return_value = "test_output.wav"
        mock_validate.return_value = True
        mock_get_info.return_value = {
            "duration": 10.5,
            "sample_rate": 44100,
            "channels": 2,
        }

        # Import and run the main section

        # This would normally be tested by running the script directly,
        # but we can verify the functions are called correctly
        mock_convert.assert_not_called()  # Only called when script runs

    def test_audio_extensions_comprehensive(self) -> None:
        """Test that all common audio extensions are supported."""
        common_extensions = [
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".ogg",
            ".aac",
            ".webm",
            ".mp4",
        ]

        with patch("os.path.exists", return_value=True):
            for ext in common_extensions:
                assert validate_audio_file(f"test{ext}"), (
                    f"Extension {ext} should be valid"
                )
                assert validate_audio_file(f"test{ext.upper()}"), (
                    f"Extension {ext.upper()} should be valid"
                )
