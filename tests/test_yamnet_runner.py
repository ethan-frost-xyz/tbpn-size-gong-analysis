"""Tests for yamnet_runner module."""

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np
import pytest

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture

from gong_detector.core.yamnet_runner import YAMNetGongDetector


class TestYAMNetGongDetectorInit:
    """Tests for YAMNetGongDetector initialization."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        detector = YAMNetGongDetector()

        assert detector.model is None
        assert detector.class_names is None
        assert detector.gong_class_index == 172
        assert detector.target_sample_rate == 16000

    def test_init_attributes_types(self) -> None:
        """Test initialization attribute types."""
        detector = YAMNetGongDetector()

        assert isinstance(detector.gong_class_index, int)
        assert isinstance(detector.target_sample_rate, int)


class TestModelLoading:
    """Tests for model loading functionality."""

    @patch("gong_detector.core.yamnet_runner.hub.load")
    @patch("gong_detector.core.yamnet_runner.pd.read_csv")
    def test_load_model_success(self, mock_read_csv: Mock, mock_hub_load: Mock) -> None:
        """Test successful model loading."""
        # Mock the model
        mock_model = Mock()
        mock_model.class_map_path.return_value.numpy.return_value.decode.return_value = "test_path"
        mock_hub_load.return_value = mock_model

        # Mock the CSV reading
        mock_df = Mock()
        mock_df.__getitem__ = Mock(return_value=[
            "class1",
            "class2",
            "gong",
            "class4",
        ] * 50)  # 200+ classes
        mock_read_csv.return_value = mock_df

        detector = YAMNetGongDetector()
        detector.load_model()

        assert detector.model is not None
        assert detector.class_names is not None
        assert len(detector.class_names) >= 100  # Should have many classes

        # Verify calls
        mock_hub_load.assert_called_once_with("https://tfhub.dev/google/yamnet/1")
        mock_read_csv.assert_called_once_with("test_path")

    @patch("gong_detector.core.yamnet_runner.hub.load")
    def test_load_model_failure(self, mock_hub_load: Mock) -> None:
        """Test model loading failure."""
        mock_hub_load.side_effect = Exception("Network error")

        detector = YAMNetGongDetector()

        with pytest.raises(RuntimeError, match="Model loading failed"):
            detector.load_model()

    def test_get_gong_class_name_no_classes(self) -> None:
        """Test getting gong class name when no classes loaded."""
        detector = YAMNetGongDetector()
        result = detector._get_gong_class_name()

        assert result == "unknown"

    def test_get_gong_class_name_with_classes(self) -> None:
        """Test getting gong class name with classes loaded."""
        detector = YAMNetGongDetector()
        detector.class_names = ["class1"] * 200  # Fill with dummy classes
        detector.class_names[172] = "Gong"  # Set the gong class

        result = detector._get_gong_class_name()
        assert result == "Gong"


class TestAudioPreprocessing:
    """Tests for audio preprocessing functionality."""

    @patch("gong_detector.core.yamnet_runner.tf.io.read_file")
    @patch("gong_detector.core.yamnet_runner.tf.audio.decode_wav")
    def test_load_audio_file_success(self, mock_decode: Mock, mock_read: Mock) -> None:
        """Test successful audio file loading."""
        # Mock file reading
        mock_read.return_value = b"fake_audio_data"

        # Mock audio decoding
        mock_waveform = Mock()
        mock_waveform.numpy.return_value.flatten.return_value = np.array(
            [0.1, 0.2, 0.3], dtype=np.float32
        )
        mock_sample_rate = Mock()
        mock_sample_rate.numpy.return_value = 16000

        mock_decode.return_value = (mock_waveform, mock_sample_rate)

        detector = YAMNetGongDetector()

        with patch("os.path.exists", return_value=True):
            waveform, sample_rate = detector._load_audio_file("test.wav")

        assert isinstance(waveform, np.ndarray)
        assert sample_rate == 16000
        assert len(waveform) == 3

    def test_load_and_preprocess_audio_file_not_found(self) -> None:
        """Test audio preprocessing with non-existent file."""
        detector = YAMNetGongDetector()

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            detector.load_and_preprocess_audio("nonexistent.wav")

    @patch("gong_detector.core.yamnet_runner.tf.signal.resample", create=True)
    def test_resample_if_needed_different_rate(self, mock_resample: Mock) -> None:
        """Test resampling when sample rates differ."""
        detector = YAMNetGongDetector()
        original_waveform = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        # Mock resampling result
        mock_result = Mock()
        mock_result.numpy.return_value.flatten.return_value = np.array(
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float32
        )
        mock_resample.return_value = mock_result

        result = detector._resample_if_needed(original_waveform, 8000)

        assert len(result) == 7  # Upsampled
        mock_resample.assert_called_once()

    def test_resample_if_needed_same_rate(self) -> None:
        """Test that no resampling occurs when rates match."""
        detector = YAMNetGongDetector()
        original_waveform = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = detector._resample_if_needed(original_waveform, 16000)

        assert np.array_equal(result, original_waveform)

    def test_normalize_audio_basic(self) -> None:
        """Test basic audio normalization."""
        detector = YAMNetGongDetector()
        audio = np.array([2.0, -1.5, 1.0, -2.0], dtype=np.float32)

        result = detector._normalize_audio(audio)

        # Should be normalized to [-1, 1] range
        assert np.max(np.abs(result)) <= 1.0
        assert abs(np.max(np.abs(result)) - 1.0) < 1e-6

    def test_normalize_audio_zero(self) -> None:
        """Test normalization of zero audio."""
        detector = YAMNetGongDetector()
        audio = np.zeros(10, dtype=np.float32)

        result = detector._normalize_audio(audio)

        assert np.array_equal(result, audio)


class TestInference:
    """Tests for YAMNet inference functionality."""

    def test_run_inference_model_not_loaded(self) -> None:
        """Test inference when model is not loaded."""
        detector = YAMNetGongDetector()
        waveform = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            detector.run_inference(waveform)

    @patch("gong_detector.core.yamnet_runner.tf.constant")
    def test_run_inference_success(self, mock_constant: Mock) -> None:
        """Test successful inference."""
        detector = YAMNetGongDetector()

        # Mock model
        mock_model = Mock()
        mock_scores = Mock()
        mock_scores_array = np.random.random((100, 521))  # 100 time steps, 521 classes
        mock_scores.numpy.return_value = mock_scores_array
        mock_scores.shape = mock_scores_array.shape  # Add shape attribute
        mock_embeddings = Mock()
        mock_embeddings.numpy.return_value = np.random.random((100, 1024))
        mock_spectrogram = Mock()
        mock_spectrogram.numpy.return_value = np.random.random((64, 100))

        mock_model.return_value = (mock_scores, mock_embeddings, mock_spectrogram)
        detector.model = mock_model

        waveform = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        scores, embeddings, spectrogram = detector.run_inference(waveform)

        assert isinstance(scores, np.ndarray)
        assert isinstance(embeddings, np.ndarray)
        assert isinstance(spectrogram, np.ndarray)
        assert scores.shape[1] == 521  # YAMNet has 521 classes


class TestGongDetection:
    """Tests for gong detection functionality."""

    def test_detect_gongs_basic(self) -> None:
        """Test basic gong detection."""
        detector = YAMNetGongDetector()

        # Create mock scores with some gong detections
        scores = np.zeros((10, 521))  # Start with all zeros
        scores[2, 172] = 0.8  # High confidence gong at position 2
        scores[7, 172] = 0.6  # Medium confidence gong at position 7
        scores[5, 172] = 0.3  # Low confidence gong (below threshold)

        detections = detector.detect_gongs(scores, confidence_threshold=0.5)

        assert len(detections) == 2  # Only 2 above threshold
        assert detections[0][1] == 0.8  # First detection confidence
        assert detections[1][1] == 0.6  # Second detection confidence

    def test_detect_gongs_no_detections(self) -> None:
        """Test gong detection with no valid detections."""
        detector = YAMNetGongDetector()

        # Create scores with low gong confidence
        scores = np.random.random((10, 521)) * 0.3  # All below 0.5 threshold

        detections = detector.detect_gongs(scores, confidence_threshold=0.5)

        assert len(detections) == 0

    def test_calculate_hop_length_with_duration(self) -> None:
        """Test hop length calculation with known duration."""
        detector = YAMNetGongDetector()

        hop_length = detector._calculate_hop_length(10.0, 20)

        assert hop_length == 0.5  # 10 seconds / 20 predictions

    def test_calculate_hop_length_without_duration(self) -> None:
        """Test hop length calculation without duration."""
        detector = YAMNetGongDetector()

        hop_length = detector._calculate_hop_length(None, 20)

        assert hop_length == 0.48  # Default YAMNet hop length


class TestOutputFormatting:
    """Tests for detection output formatting."""

    def test_print_detections_empty(self, capsys: "CaptureFixture") -> None:
        """Test printing empty detections."""
        detector = YAMNetGongDetector()
        detector.print_detections([])

        captured = capsys.readouterr()
        assert "No gong detections found" in captured.out

    def test_print_detections_with_data(self, capsys: "CaptureFixture") -> None:
        """Test printing detections with data."""
        detector = YAMNetGongDetector()
        detections = [(1.5, 0.8), (3.2, 0.6)]

        detector.print_detections(detections)

        captured = capsys.readouterr()
        assert "GONG DETECTIONS" in captured.out
        assert "1.50" in captured.out
        assert "0.8000" in captured.out

    def test_detections_to_dataframe_empty(self) -> None:
        """Test converting empty detections to DataFrame."""
        detector = YAMNetGongDetector()
        df = detector.detections_to_dataframe([])

        assert len(df) == 0
        assert "timestamp_seconds" in df.columns
        assert "confidence" in df.columns

    def test_detections_to_dataframe_with_data(self) -> None:
        """Test converting detections to DataFrame."""
        detector = YAMNetGongDetector()
        detections = [(1.5, 0.8), (3.2, 0.6), (5.1, 0.7)]

        df = detector.detections_to_dataframe(detections)

        assert len(df) == 3
        assert df["timestamp_seconds"].tolist() == [1.5, 3.2, 5.1]
        assert df["confidence"].tolist() == [0.8, 0.6, 0.7]


class TestIntegration:
    """Integration tests for the full detector workflow."""

    @patch("gong_detector.core.yamnet_runner.hub.load")
    @patch("gong_detector.core.yamnet_runner.pd.read_csv")
    @patch("os.path.exists")
    @patch("gong_detector.core.yamnet_runner.tf.io.read_file")
    @patch("gong_detector.core.yamnet_runner.tf.audio.decode_wav")
    def test_full_workflow_mock(
        self,
        mock_decode: Mock,
        mock_read: Mock,
        mock_exists: Mock,
        mock_read_csv: Mock,
        mock_hub_load: Mock,
    ) -> None:
        """Test the full workflow with mocked dependencies."""
        # Setup mocks
        mock_exists.return_value = True

        # Mock model loading
        mock_model = Mock()
        mock_model.class_map_path.return_value.numpy.return_value.decode.return_value = "test"
        mock_hub_load.return_value = mock_model

        mock_df = Mock()
        mock_df.__getitem__ = Mock(return_value=["class"] * 521)
        mock_read_csv.return_value = mock_df

        # Mock audio loading
        mock_read.return_value = b"fake"
        mock_waveform = Mock()
        mock_waveform.numpy.return_value.flatten.return_value = np.random.random(
            16000
        ).astype(np.float32)
        mock_sample_rate = Mock()
        mock_sample_rate.numpy.return_value = 16000
        mock_decode.return_value = (mock_waveform, mock_sample_rate)

        # Mock inference
        mock_scores = Mock()
        mock_scores.numpy.return_value = np.random.random((10, 521))
        mock_embeddings = Mock()
        mock_embeddings.numpy.return_value = np.random.random((10, 1024))
        mock_spectrogram = Mock()
        mock_spectrogram.numpy.return_value = np.random.random((64, 10))
        mock_model.return_value = (mock_scores, mock_embeddings, mock_spectrogram)

        # Run workflow
        detector = YAMNetGongDetector()
        detector.load_model()

        waveform, sample_rate = detector.load_and_preprocess_audio("test.wav")
        scores, _, _ = detector.run_inference(waveform)
        detections = detector.detect_gongs(scores)

        # Basic assertions
        assert isinstance(waveform, np.ndarray)
        assert sample_rate == 16000
        assert isinstance(scores, np.ndarray)
        assert isinstance(detections, list)
