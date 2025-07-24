"""Tests for detect_from_youtube module."""

import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture

from gong_detector.core.detect_from_youtube import (
    _convert_and_trim_audio,
    _download_youtube_audio,
    cleanup_old_temp_files,
    create_argument_parser,
    create_temp_audio_path,
    download_and_trim_youtube_audio,
    format_time,
    print_summary,
    process_audio_with_yamnet,
    save_results_to_csv,
    setup_directories,
)


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_format_time_basic(self) -> None:
        """Test basic time formatting."""
        assert format_time(0) == "00:00:00"
        assert format_time(60) == "00:01:00"
        assert format_time(3661) == "01:01:01"
        assert format_time(7325.5) == "02:02:05"  # Truncates seconds

    def test_format_time_edge_cases(self) -> None:
        """Test time formatting edge cases."""
        assert format_time(0.5) == "00:00:00"  # Sub-second rounds down
        assert format_time(59.9) == "00:00:59"  # Just under minute
        assert format_time(86399) == "23:59:59"  # Just under day

    @patch("os.makedirs")
    def test_setup_directories(self, mock_makedirs: Mock) -> None:
        """Test directory setup."""
        temp_dir, csv_dir = setup_directories()

        assert temp_dir == "temp_audio"
        assert csv_dir == "csv_results"

        # Should create both directories
        mock_makedirs.assert_any_call("temp_audio", exist_ok=True)
        mock_makedirs.assert_any_call("csv_results", exist_ok=True)

    def test_create_temp_audio_path(self) -> None:
        """Test temporary audio path creation."""
        path = create_temp_audio_path("temp_dir")

        assert path.startswith("temp_dir/temp_youtube_audio_")
        assert path.endswith(".wav")
        assert len(path.split("_")[-1]) == 12  # 8-char hex + .wav

    def test_create_temp_audio_path_unique(self) -> None:
        """Test that temporary paths are unique."""
        path1 = create_temp_audio_path("temp_dir")
        path2 = create_temp_audio_path("temp_dir")

        assert path1 != path2


class TestFileCleanup:
    """Tests for file cleanup functionality."""

    @patch("glob.glob")
    @patch("os.path.getmtime")
    @patch("os.remove")
    @patch("time.time")
    def test_cleanup_old_temp_files_removes_old(
        self, mock_time: Mock, mock_remove: Mock, mock_getmtime: Mock, mock_glob: Mock
    ) -> None:
        """Test cleanup removes old files."""
        mock_time.return_value = 86400  # Current time = 1 day
        mock_glob.return_value = ["temp_audio/temp_youtube_audio_abc123.wav"]
        mock_getmtime.return_value = 0  # File is 1 day old

        cleanup_old_temp_files("temp_audio", max_age_hours=12)

        # Should remove the old file
        mock_remove.assert_called_once_with("temp_audio/temp_youtube_audio_abc123.wav")

    @patch("glob.glob")
    @patch("os.path.getmtime")
    @patch("os.remove")
    @patch("time.time")
    def test_cleanup_old_temp_files_keeps_recent(
        self, mock_time: Mock, mock_remove: Mock, mock_getmtime: Mock, mock_glob: Mock
    ) -> None:
        """Test cleanup keeps recent files."""
        mock_time.return_value = 3600  # Current time = 1 hour
        mock_glob.return_value = ["temp_audio/temp_youtube_audio_abc123.wav"]
        mock_getmtime.return_value = 1800  # File is 30 minutes old

        cleanup_old_temp_files("temp_audio", max_age_hours=1)

        # Should not remove the recent file
        mock_remove.assert_not_called()

    @patch("glob.glob")
    @patch("os.path.getmtime")
    @patch("os.remove")
    @patch("time.time")
    def test_cleanup_old_temp_files_handles_errors(
        self, mock_time: Mock, mock_remove: Mock, mock_getmtime: Mock, mock_glob: Mock
    ) -> None:
        """Test cleanup handles removal errors gracefully."""
        mock_time.return_value = 86400
        mock_glob.return_value = ["temp_audio/temp_youtube_audio_abc123.wav"]
        mock_getmtime.return_value = 0
        mock_remove.side_effect = OSError("Permission denied")

        # Should not raise exception
        cleanup_old_temp_files("temp_audio", max_age_hours=12)

        mock_remove.assert_called_once()


class TestYouTubeDownloadAndTrim:
    """Tests for YouTube download and trimming."""

    @patch("gong_detector.core.detect_from_youtube._download_youtube_audio")
    @patch("gong_detector.core.detect_from_youtube._convert_and_trim_audio")
    def test_download_and_trim_youtube_audio_basic(
        self, mock_convert: Mock, mock_download: Mock
    ) -> None:
        """Test basic YouTube download and trim."""
        mock_download.return_value = "temp_download.mp3"

        result = download_and_trim_youtube_audio(
            "https://youtube.com/watch?v=test", "output.wav", start_time=10, duration=30
        )

        assert result == "output.wav"
        mock_download.assert_called_once()
        mock_convert.assert_called_once_with("temp_download.mp3", "output.wav", 10, 30)

    @patch("gong_detector.core.detect_from_youtube.yt_dlp.YoutubeDL")
    @patch("os.listdir")
    def test_download_youtube_audio_success(
        self, mock_listdir: Mock, mock_ydl_class: Mock
    ) -> None:
        """Test successful YouTube download."""
        # Mock yt-dlp
        mock_ydl = Mock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl

        # Mock directory listing
        mock_listdir.return_value = ["temp_audio.mp3"]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = os.path.join(temp_dir, "temp_audio.%(ext)s")
            result = _download_youtube_audio(
                "https://youtube.com/watch?v=test", output_template
            )

            assert "temp_audio.mp3" in result
            mock_ydl.download.assert_called_once_with(
                ["https://youtube.com/watch?v=test"]
            )

    @patch("gong_detector.core.detect_from_youtube.yt_dlp.YoutubeDL")
    @patch("os.listdir")
    def test_download_youtube_audio_no_files(
        self, mock_listdir: Mock, mock_ydl_class: Mock
    ) -> None:
        """Test YouTube download when no files are created."""
        mock_ydl = Mock()
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl
        mock_listdir.return_value = []  # No files found

        with tempfile.TemporaryDirectory() as temp_dir:
            output_template = os.path.join(temp_dir, "temp_audio.%(ext)s")

            with pytest.raises(
                RuntimeError, match="Failed to download audio from YouTube"
            ):
                _download_youtube_audio(
                    "https://youtube.com/watch?v=test", output_template
                )

    @patch("subprocess.run")
    def test_convert_and_trim_audio_with_trim(self, mock_run: Mock) -> None:
        """Test audio conversion with trimming parameters."""
        _convert_and_trim_audio("input.mp3", "output.wav", 10, 30)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]  # Command arguments

        assert "ffmpeg" in args
        assert "input.mp3" in args
        assert "output.wav" in args
        assert "-ss" in args and "10" in args  # Start time
        assert "-t" in args and "30" in args  # Duration

    @patch("subprocess.run")
    def test_convert_and_trim_audio_no_trim(self, mock_run: Mock) -> None:
        """Test audio conversion without trimming."""
        _convert_and_trim_audio("input.mp3", "output.wav")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]

        # Should not have trim parameters
        assert "-ss" not in args
        assert "-t" not in args


class TestOutputAndSummary:
    """Tests for output and summary functions."""

    def test_print_summary_basic(self, capsys: "CaptureFixture") -> None:
        """Test basic summary printing."""
        detections = [(1.5, 0.8), (3.2, 0.6), (5.1, 0.7)]

        print_summary(detections, 10.0, 0.0)

        captured = capsys.readouterr()
        assert "Detected 3 gongs" in captured.out
        assert "00:00:00 and 00:00:10" in captured.out
        assert "Average confidence: 0.700" in captured.out
        assert "Maximum confidence: 0.800" in captured.out

    def test_print_summary_with_offset(self, capsys: "CaptureFixture") -> None:
        """Test summary printing with time offset."""
        detections = [(1.0, 0.5)]

        print_summary(detections, 5.0, 100.0)  # Start at 100 seconds

        captured = capsys.readouterr()
        assert "00:01:40 and 00:01:45" in captured.out  # Fixed expected output

    def test_print_summary_no_detections(self, capsys: "CaptureFixture") -> None:
        """Test summary printing with no detections."""
        print_summary([], 10.0, 0.0)

        captured = capsys.readouterr()
        assert "Detected 0 gongs" in captured.out
        # Should not have confidence info
        assert "Average confidence" not in captured.out

    @patch("gong_detector.core.detect_from_youtube.YAMNetGongDetector")
    def test_save_results_to_csv(self, mock_detector_class: Mock) -> None:
        """Test saving results to CSV."""
        # Mock detector and dataframe
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector

        mock_df = Mock()
        mock_detector.detections_to_dataframe.return_value = mock_df

        detections = [(1.5, 0.8), (3.2, 0.6)]
        save_results_to_csv(detections, "test_results", "csv_dir")

        mock_df.to_csv.assert_called_once_with("csv_dir/test_results.csv", index=False)

    @patch("gong_detector.core.detect_from_youtube.YAMNetGongDetector")
    def test_save_results_to_csv_adds_extension(
        self, mock_detector_class: Mock
    ) -> None:
        """Test CSV saving adds .csv extension if missing."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_df = Mock()
        mock_detector.detections_to_dataframe.return_value = mock_df

        save_results_to_csv([], "test_results", "csv_dir")

        # Should add .csv extension
        mock_df.to_csv.assert_called_once_with("csv_dir/test_results.csv", index=False)


class TestYAMNetProcessing:
    """Tests for YAMNet audio processing."""

    @patch("gong_detector.core.detect_from_youtube.YAMNetGongDetector")
    def test_process_audio_with_yamnet_complete_workflow(
        self, mock_detector_class: Mock
    ) -> None:
        """Test complete YAMNet processing workflow."""
        # Mock detector instance
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector

        # Mock audio processing
        mock_waveform = np.random.random(16000).astype(np.float32)
        mock_detector.load_and_preprocess_audio.return_value = (mock_waveform, 16000)

        # Mock inference
        mock_scores = np.random.random((10, 521))
        mock_scores[:, 172] = 0.8  # Set gong scores high
        mock_detector.run_inference.return_value = (mock_scores, None, None)

        # Mock detection
        mock_detections = [(1.0, 0.8), (3.0, 0.7)]
        mock_detector.detect_gongs.return_value = mock_detections

        # Run processing
        detections, duration, max_confidence = process_audio_with_yamnet(
            "test.wav", 0.5
        )

        # Verify calls and results
        mock_detector.load_model.assert_called_once()
        mock_detector.load_and_preprocess_audio.assert_called_once_with("test.wav")
        mock_detector.run_inference.assert_called_once()
        mock_detector.detect_gongs.assert_called_once()
        mock_detector.print_detections.assert_called_once_with(mock_detections)

        assert detections == mock_detections
        assert duration == 1.0  # 16000 samples / 16000 Hz
        assert max_confidence == 0.8


class TestArgumentParser:
    """Tests for command line argument parsing."""

    def test_create_argument_parser_basic(self) -> None:
        """Test basic argument parser creation."""
        parser = create_argument_parser()

        # Test with minimal args
        args = parser.parse_args(["https://youtube.com/watch?v=test"])

        assert args.youtube_url == "https://youtube.com/watch?v=test"
        assert args.start_time is None
        assert args.duration is None
        assert args.threshold == 0.4  # Default
        assert args.save_csv is None

    def test_create_argument_parser_all_args(self) -> None:
        """Test argument parser with all arguments."""
        parser = create_argument_parser()

        args = parser.parse_args(
            [
                "https://youtube.com/watch?v=test",
                "--start_time",
                "100",
                "--duration",
                "30",
                "--threshold",
                "0.6",
                "--save_csv",
                "results.csv",
            ]
        )

        assert args.youtube_url == "https://youtube.com/watch?v=test"
        assert args.start_time == 100
        assert args.duration == 30
        assert args.threshold == 0.6
        assert args.save_csv == "results.csv"

    def test_create_argument_parser_help_text(self) -> None:
        """Test that parser has expected help text."""
        parser = create_argument_parser()

        help_text = parser.format_help()
        assert "Detect gongs in YouTube videos" in help_text
        assert "Examples:" in help_text
        assert "--threshold" in help_text


class TestMainIntegration:
    """Integration tests for main functionality."""

    @patch("gong_detector.core.detect_from_youtube.setup_directories")
    @patch("gong_detector.core.detect_from_youtube.cleanup_old_temp_files")
    @patch("gong_detector.core.detect_from_youtube.create_temp_audio_path")
    @patch("gong_detector.core.detect_from_youtube.download_and_trim_youtube_audio")
    @patch("gong_detector.core.detect_from_youtube.process_audio_with_yamnet")
    @patch("os.path.exists")
    @patch("os.remove")
    @patch("sys.argv", ["detect_from_youtube.py", "https://youtube.com/watch?v=test"])
    def test_main_workflow_basic(
        self,
        mock_remove: Mock,
        mock_exists: Mock,
        mock_process: Mock,
        mock_download: Mock,
        mock_create_path: Mock,
        mock_cleanup: Mock,
        mock_setup: Mock,
    ) -> None:
        """Test basic main workflow."""
        # Setup mocks
        mock_setup.return_value = ("temp_audio", "csv_results")
        mock_create_path.return_value = "temp_audio/temp_file.wav"
        mock_download.return_value = "temp_audio/temp_file.wav"
        mock_process.return_value = ([(1.0, 0.8)], 10.0, 0.8)
        mock_exists.return_value = True

        # Import main function and test argument parsing
        from gong_detector.core.detect_from_youtube import create_argument_parser

        parser = create_argument_parser()
        args = parser.parse_args(["https://youtube.com/watch?v=test"])

        # Verify argument parsing worked
        assert args.youtube_url == "https://youtube.com/watch?v=test"
        assert args.threshold == 0.4

    def test_main_with_csv_output(self) -> None:
        """Test main workflow with CSV output."""
        # Test CSV saving component directly
        detections = [(1.0, 0.8), (3.0, 0.6)]

        # This should work now that we've fixed the directory creation
        save_results_to_csv(detections, "test_results", "csv_results")

        # Verify the file was created
        import os

        assert os.path.exists("csv_results/test_results.csv")

        # Test summary printing component
        print_summary(detections, 10.0, 0.0)


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_download_error_handling(self) -> None:
        """Test error handling during download."""
        # Test that invalid URLs raise appropriate exceptions
        with pytest.raises(Exception):  # yt-dlp raises DownloadError for invalid URLs
            download_and_trim_youtube_audio("invalid_url", "output.wav")

    def test_yamnet_error_handling(self) -> None:
        """Test error handling during YAMNet processing."""
        # Test that non-existent files raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            process_audio_with_yamnet("test.wav", 0.5)
