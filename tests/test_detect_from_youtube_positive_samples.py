"""Tests for positive samples saving functionality."""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture

from gong_detector.core.detect_from_youtube import save_positive_samples


class TestSavePositiveSamples:
    """Tests for save_positive_samples function."""

    @patch("gong_detector.core.detect_from_youtube.YAMNetGongDetector")
    @patch("soundfile.write")
    def test_save_positive_samples_basic(
        self, mock_sf_write: Mock, mock_detector_class: Mock
    ) -> None:
        """Test basic positive samples saving."""
        # Mock detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector

        # Mock audio loading
        mock_waveform = np.random.random(16000).astype(np.float32)  # 1 second at 16kHz
        mock_detector.load_and_preprocess_audio.return_value = (mock_waveform, 16000)

        # Test data
        detections = [(1.0, 0.8), (3.0, 0.7)]
        audio_path = "test_audio.wav"

        with tempfile.TemporaryDirectory() as temp_dir:
            positive_dir = Path(temp_dir) / "positive"

            # Call function
            save_positive_samples(detections, audio_path, positive_dir)

            # Verify calls
            mock_detector.load_and_preprocess_audio.assert_called_once_with(audio_path)
            assert mock_sf_write.call_count == 2  # Two detections

            # Verify directory was created
            assert positive_dir.exists()

    def test_save_positive_samples_no_detections(
        self, capsys: "CaptureFixture"
    ) -> None:
        """Test handling of empty detections list."""
        detections: list[tuple[float, float]] = []
        audio_path = "test_audio.wav"

        with tempfile.TemporaryDirectory() as temp_dir:
            positive_dir = Path(temp_dir) / "positive"

            save_positive_samples(detections, audio_path, positive_dir)

            captured = capsys.readouterr()
            assert "No gong detections to save" in captured.out

    @patch("gong_detector.core.detect_from_youtube.YAMNetGongDetector")
    @patch("soundfile.write")
    def test_save_positive_samples_error_handling(
        self, mock_sf_write: Mock, mock_detector_class: Mock
    ) -> None:
        """Test error handling during sample saving."""
        # Mock detector
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector

        # Mock audio loading
        mock_waveform = np.random.random(16000).astype(np.float32)
        mock_detector.load_and_preprocess_audio.return_value = (mock_waveform, 16000)

        # Mock soundfile to raise error
        mock_sf_write.side_effect = Exception("Write failed")

        detections = [(1.0, 0.8)]
        audio_path = "test_audio.wav"

        with tempfile.TemporaryDirectory() as temp_dir:
            positive_dir = Path(temp_dir) / "positive"

            # Should not raise exception
            save_positive_samples(detections, audio_path, positive_dir)

            # Verify error was handled gracefully
            mock_sf_write.assert_called_once()
