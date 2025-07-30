"""YAMNet runner module for gong detection.

This module provides the core YAMNet functionality for loading the model,
processing audio, and running inference to detect gong sounds.
"""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import tensorflow as tf  # type: ignore
import tensorflow_hub as hub  # type: ignore


class YAMNetGongDetector:
    """YAMNet-based gong sound detector for audio analysis."""

    def __init__(
        self, use_trained_classifier: bool = False, batch_size: int = 1000
    ) -> None:
        """Initialize the YAMNet gong detector.

        Args:
            use_trained_classifier: Whether to use the trained classifier for enhanced detection
            batch_size: Batch size for classifier predictions (larger = faster but more memory)
        """
        # Configure TensorFlow for optimal CPU usage
        self._configure_tensorflow()

        self.model: Optional[hub.KerasLayer] = None
        self.class_names: Optional[list[str]] = None
        self.gong_class_index: int = 172  # YAMNet class index for "gong"
        self.target_sample_rate: int = 16000

        # Trained classifier support
        self.use_trained_classifier: bool = use_trained_classifier
        self.trained_classifier: Optional[object] = None
        self.classifier_config: Optional[dict] = None
        self.batch_size: int = batch_size

    def _configure_tensorflow(self) -> None:
        """Configure TensorFlow for optimal CPU performance."""
        # Enable multi-threading
        tf.config.threading.set_inter_op_parallelism_threads(
            8
        )  # Use 8 threads for inter-op
        tf.config.threading.set_intra_op_parallelism_threads(
            4
        )  # Use 4 threads for intra-op

        # Enable memory growth to prevent memory issues
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory growth setting failed: {e}")

        # Enable mixed precision for faster computation (if supported)
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass  # Mixed precision not available, continue with default

    def load_model(self) -> None:
        """Load the YAMNet model from TensorFlow Hub."""
        print("Loading YAMNet model from TensorFlow Hub...")
        try:
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            self._load_class_names()
            print(
                f"Model loaded successfully. Total classes: {len(self.class_names or [])}"
            )
            print(
                f"Gong class (index {self.gong_class_index}): {self._get_gong_class_name()}"
            )
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {e}") from e

    def load_trained_classifier(self) -> None:
        """Load the trained classifier for enhanced gong detection.

        Raises:
            RuntimeError: If classifier loading fails
        """
        if not self.use_trained_classifier:
            return

        print("Loading trained classifier...")
        try:
            # Find model files relative to the core module
            module_dir = Path(__file__).parent
            core_dir = module_dir.parent
            models_dir = core_dir / "models"

            classifier_path = models_dir / "classifier.pkl"
            config_path = models_dir / "config.json"

            if not classifier_path.exists():
                raise FileNotFoundError(
                    f"Trained classifier not found: {classifier_path}"
                )

            # Load classifier
            with open(classifier_path, "rb") as f:
                self.trained_classifier = pickle.load(f)

            # Load config
            import json

            with open(config_path) as f:
                self.classifier_config = json.load(f)

            print(
                f"✓ Loaded {self.classifier_config['model_type']} with {self.classifier_config['feature_count']} features"
            )
            print(
                f"✓ Training accuracy: {self.classifier_config['performance']['accuracy']:.3f}"
            )

        except Exception as e:
            raise RuntimeError(f"Trained classifier loading failed: {e}") from e

    def _load_class_names(self) -> None:
        """Load class names from the model."""
        if self.model is None:
            raise RuntimeError("Model must be loaded first")

        class_map_path = self.model.class_map_path().numpy().decode("utf-8")
        self.class_names = list(pd.read_csv(class_map_path)["display_name"])

    def _get_gong_class_name(self) -> str:
        """Get the name of the gong class."""
        if self.class_names is None:
            return "unknown"
        return self.class_names[self.gong_class_index]

    def load_and_preprocess_audio(self, audio_path: str) -> tuple[np.ndarray, int]:
        """Load and preprocess audio file for YAMNet inference.

        Args:
            audio_path: Path to the audio file to process

        Returns:
            Tuple of (preprocessed_waveform, sample_rate)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio processing fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Loading audio file: {audio_path}")

        try:
            waveform, sample_rate = self._load_audio_file(audio_path)
            waveform = self._resample_if_needed(waveform, sample_rate)
            waveform = self._normalize_audio(waveform)

            print(
                f"Audio loaded: {len(waveform)} samples at {self.target_sample_rate}Hz"
            )
            print(f"Duration: {len(waveform) / self.target_sample_rate:.2f} seconds")

            return waveform, self.target_sample_rate

        except Exception as e:
            raise ValueError(f"Failed to process audio file {audio_path}: {e}") from e

    def _load_audio_file(self, audio_path: str) -> tuple[np.ndarray, int]:
        """Load audio file using TensorFlow."""
        audio_binary = tf.io.read_file(audio_path)
        audio_decoded = tf.audio.decode_wav(
            audio_binary,
            desired_channels=1,  # Convert to mono
            desired_samples=-1,  # Keep original length
        )

        waveform = audio_decoded[0].numpy().flatten()  # type: ignore
        sample_rate = int(audio_decoded[1].numpy())  # type: ignore

        return waveform, sample_rate

    def _resample_if_needed(
        self, waveform: np.ndarray, current_rate: int
    ) -> np.ndarray:
        """Resample audio to target sample rate if needed."""
        if current_rate == self.target_sample_rate:
            return waveform

        print(f"Resampling from {current_rate}Hz to {self.target_sample_rate}Hz...")

        # Use TensorFlow's built-in resampling for better quality
        waveform_tensor = tf.constant(waveform, dtype=tf.float32)
        waveform_tensor = tf.expand_dims(waveform_tensor, 0)  # Add batch dimension

        resampled = tf.signal.resample(
            waveform_tensor, int(len(waveform) * self.target_sample_rate / current_rate)
        )

        return resampled.numpy().flatten()

    def _normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        waveform = waveform.astype(np.float32)
        max_val = np.max(np.abs(waveform))

        if max_val > 0:
            waveform = waveform / max_val

        return waveform

    def run_inference(
        self, waveform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run YAMNet inference on audio waveform.

        Args:
            waveform: Audio waveform as numpy array

        Returns:
            Tuple of (scores, embeddings, spectrogram)

        Raises:
            RuntimeError: If model is not loaded or inference fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print("Running YAMNet inference...")

        try:
            # Use float32 for optimal performance on CPU
            waveform_tensor = tf.constant(waveform, dtype=tf.float32)

            # Run inference with optimized settings
            with tf.device("/CPU:0"):  # Explicitly use CPU for better control
                scores, embeddings, spectrogram = self.model(waveform_tensor)

            print(f"Inference complete. Generated {scores.shape[0]} predictions")
            print("Each prediction covers ~0.96 seconds of audio")

            return scores.numpy(), embeddings.numpy(), spectrogram.numpy()

        except Exception as e:
            raise RuntimeError(f"YAMNet inference failed: {e}") from e

    def detect_gongs(
        self,
        scores: np.ndarray,
        confidence_threshold: float = 0.5,
        max_confidence_threshold: Optional[float] = None,
        audio_duration: Optional[float] = None,
    ) -> list[tuple[float, float, float]]:
        """Detect gong sounds based on YAMNet scores.

        Args:
            scores: YAMNet prediction scores array
            confidence_threshold: Minimum confidence for gong detection
            max_confidence_threshold: Maximum confidence for gong detection (optional)
            audio_duration: Total audio duration in seconds (for validation)

        Returns:
            List of (window_start, confidence, display_timestamp) tuples for detected gongs
        """
        if max_confidence_threshold is not None:
            print(
                f"Detecting gongs with confidence range: {confidence_threshold} - {max_confidence_threshold}"
            )
        else:
            print(f"Detecting gongs with confidence threshold: {confidence_threshold}")

        gong_scores = scores[:, self.gong_class_index]
        hop_length = self._calculate_hop_length(audio_duration, len(gong_scores))
        window_duration = 0.96  # YAMNet window duration

        detections: list[tuple[float, float, float]] = []
        for i, confidence in enumerate(gong_scores):
            # Check minimum threshold
            if confidence <= confidence_threshold:
                continue
            # Check maximum threshold if specified
            if (
                max_confidence_threshold is not None
                and confidence >= max_confidence_threshold
            ):
                continue

            window_start = i * hop_length
            display_timestamp = window_start + (window_duration / 2)  # Center of window
            detections.append((window_start, float(confidence), display_timestamp))

        print(f"Found {len(detections)} gong detections in threshold range")
        return detections

    def detect_gongs_with_classifier(
        self,
        embeddings: np.ndarray,
        confidence_threshold: float = 0.5,
        max_confidence_threshold: Optional[float] = None,
        audio_duration: Optional[float] = None,
    ) -> list[tuple[float, float, float]]:
        """Detect gong sounds using trained classifier on YAMNet embeddings.

        Args:
            embeddings: YAMNet embeddings array
            confidence_threshold: Minimum confidence for gong detection
            max_confidence_threshold: Maximum confidence for gong detection (optional)
            audio_duration: Total audio duration in seconds (for validation)

        Returns:
            List of (window_start, confidence, display_timestamp) tuples for detected gongs
        """
        if not self.use_trained_classifier or self.trained_classifier is None:
            raise RuntimeError(
                "Trained classifier not loaded. Call load_trained_classifier() first."
            )

        if max_confidence_threshold is not None:
            print(
                f"Detecting gongs with trained classifier - confidence range: {confidence_threshold} - {max_confidence_threshold}"
            )
        else:
            print(
                f"Detecting gongs with trained classifier - confidence threshold: {confidence_threshold}"
            )

        hop_length = self._calculate_hop_length(audio_duration, len(embeddings))
        window_duration = 0.96  # YAMNet window duration

        print(
            f"Processing {len(embeddings)} embeddings in batches of {self.batch_size}..."
        )

        detections: list[tuple[float, float, float]] = []

        # Process embeddings in batches for better performance
        for batch_start in range(0, len(embeddings), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(embeddings))
            batch_embeddings = embeddings[batch_start:batch_end]

            # Reshape all embeddings in batch for classifier prediction
            batch_embeddings_reshaped = batch_embeddings.reshape(
                -1, batch_embeddings.shape[1]
            )

            # Get predictions and confidences for entire batch
            predictions = self.trained_classifier.predict(batch_embeddings_reshaped)
            probabilities = self.trained_classifier.predict_proba(
                batch_embeddings_reshaped
            )
            confidences = probabilities[
                :, 1
            ]  # Probability for positive class (gong = 1)

            # Process each prediction in the batch
            for i, (prediction, confidence) in enumerate(zip(predictions, confidences)):
                global_index = batch_start + i

                # Only consider positive predictions (gong = 1)
                if prediction != 1:
                    continue

                # Check minimum threshold
                if confidence <= confidence_threshold:
                    continue
                # Check maximum threshold if specified
                if (
                    max_confidence_threshold is not None
                    and confidence >= max_confidence_threshold
                ):
                    continue

                window_start = global_index * hop_length
                display_timestamp = window_start + (
                    window_duration / 2
                )  # Center of window
                detections.append((window_start, float(confidence), display_timestamp))

        print(f"Found {len(detections)} gong detections with trained classifier")
        return detections

    def set_batch_size(self, batch_size: int) -> None:
        """Set the batch size for classifier predictions.

        Args:
            batch_size: New batch size (larger = faster but more memory)
        """
        self.batch_size = batch_size
        print(f"Batch size set to {batch_size}")

    def get_performance_info(self) -> dict:
        """Get information about current performance configuration.

        Returns:
            Dictionary with performance settings
        """
        return {
            "batch_size": self.batch_size,
            "use_trained_classifier": self.use_trained_classifier,
            "tensorflow_threads": {
                "inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
                "intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
            },
            "available_devices": {
                "cpu": len(tf.config.list_physical_devices("CPU")),
                "gpu": len(tf.config.list_physical_devices("GPU")),
            },
        }

    def _calculate_hop_length(
        self, audio_duration: Optional[float], num_predictions: int
    ) -> float:
        """Calculate hop length between predictions."""
        if audio_duration is not None and num_predictions > 0:
            return audio_duration / num_predictions
        return 0.48  # Default YAMNet hop length

    def print_detections(self, detections: list[tuple[float, float, float]]) -> None:
        """Print gong detections in a formatted table with YouTube timestamps.

        Args:
            detections: List of (window_start, confidence, display_timestamp) tuples to display
        """
        if not detections:
            print("No gong detections found.")
            return

        # Import format_time here to avoid circular imports
        from ..utils.results_utils import format_time

        print("\n" + "=" * 70)
        print("GONG DETECTIONS")
        print("=" * 70)
        print(f"{'Window Start':<15} {'YouTube Time':<12} {'Confidence':<12}")
        print("-" * 47)

        for window_start, confidence, display_timestamp in detections:
            youtube_time = format_time(display_timestamp)
            print(f"{window_start:<15.2f} {youtube_time:<12} {confidence:<12.4f}")

        print("=" * 70)

    def detections_to_dataframe(
        self, detections: list[tuple[float, float, float]]
    ) -> pd.DataFrame:
        """Convert detections to a pandas DataFrame.

        Args:
            detections: List of (window_start, confidence, display_timestamp) tuples

        Returns:
            DataFrame with window_start_seconds, youtube_timestamp, and confidence columns
        """
        if not detections:
            return pd.DataFrame(
                {"window_start_seconds": [], "youtube_timestamp": [], "confidence": []}
            )

        window_starts, confidences, display_timestamps = zip(*detections)

        # Format YouTube timestamps as HH:MM:SS
        from ..utils.results_utils import format_time

        formatted_timestamps = [format_time(ts) for ts in display_timestamps]

        return pd.DataFrame(
            {
                "window_start_seconds": window_starts,
                "youtube_timestamp": formatted_timestamps,
                "confidence": confidences,
            }
        )
