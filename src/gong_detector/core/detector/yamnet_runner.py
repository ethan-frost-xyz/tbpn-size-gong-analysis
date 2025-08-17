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

# Configure TensorFlow logging before import to reduce spam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=info, 2=warnings, 3=errors only
import tensorflow as tf  # type: ignore
import tensorflow_hub as hub  # type: ignore

# Set TensorFlow logging to only show errors and warnings
tf.get_logger().setLevel('WARNING')


class YAMNetGongDetector:
    """YAMNet-based gong sound detector for audio analysis."""

    def __init__(
        self, use_trained_classifier: bool = False, batch_size: int = 4000
    ) -> None:
        """Initialize the YAMNet gong detector.

        Args:
            use_trained_classifier: Whether to use the trained classifier for enhanced detection
            batch_size: Batch size for classifier predictions (larger = faster but more memory)
        """
        # Initialize basic attributes first
        self.model: Optional[hub.KerasLayer] = None
        self.class_names: Optional[list[str]] = None
        self.gong_class_index: int = 172  # YAMNet class index for "gong"
        self.target_sample_rate: int = 16000

        # Trained classifier support
        self.use_trained_classifier: bool = use_trained_classifier
        self.trained_classifier: Optional[object] = None
        self.classifier_config: Optional[dict] = None
        self.batch_size: int = batch_size

        # Configure TensorFlow for optimal GPU/CPU performance (after batch_size is set)
        self._configure_tensorflow()

    def _configure_tensorflow(self) -> None:
        """Configure TensorFlow for optimal GPU/CPU performance on Mac M4."""
        # Configure GPU if available (Mac M4 Metal)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Enable memory growth for GPU
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Optimize for Mac M4 GPU
                tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
                tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
                
                # Enable mixed precision for faster computation (supported on M4)
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print(f"[OK] GPU acceleration enabled: {len(gpus)} GPU(s) detected (Mac M4 Metal)")
                print("[OK] Mixed precision enabled (float16)")
                
            except RuntimeError as e:
                print(f"GPU setup failed, falling back to CPU: {e}")
                self._configure_cpu_fallback()
        else:
            print("No GPU detected, using CPU optimization")
            self._configure_cpu_fallback()
            
        # Optimize memory allocation
        tf.config.experimental.enable_tensor_float_32_execution(True)  # Enable TF32 on supported hardware
        
        # Set memory limits to prevent system crashes
        self._configure_memory_limits()

    def _configure_cpu_fallback(self) -> None:
        """Configure CPU-specific optimizations for M4 chip."""
        # M4 has 10 CPU cores (4 performance + 6 efficiency), optimize accordingly
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores

    def _configure_memory_limits(self) -> None:
        """Configure memory limits to prevent system crashes."""
        import psutil
        
        # Get system memory info
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        # Store memory info for runtime checks
        self._system_memory_gb = total_gb
        self._safe_memory_threshold_gb = max(4.0, total_gb * 0.25)  # Reserve 25% or 4GB minimum
        
        # Adjust batch size based on available memory
        if total_gb <= 8:
            # Low memory system - use smaller batch size
            self.batch_size = min(self.batch_size, 1000)
            print(f"[INFO] Low memory system ({total_gb:.1f}GB) - reduced batch size to {self.batch_size}")
        elif total_gb <= 16:
            # Medium memory system - moderate batch size
            self.batch_size = min(self.batch_size, 2000)
            print(f"[INFO] Medium memory system ({total_gb:.1f}GB) - batch size limited to {self.batch_size}")
        
        # Show memory warnings if there are issues
        if available_gb < 4:
            print(f"[WARNING] Low memory: {available_gb:.1f}GB available. Consider closing other applications.")
        elif available_gb < 8:
            print(f"System memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
        
        # Configure GPU memory if available (silent unless there are issues)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                print("[WARNING] GPU memory configuration failed")

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
                f"[OK] Loaded {self.classifier_config['model_type']} with {self.classifier_config['feature_count']} features"
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
        """Run YAMNet inference on audio waveform with memory protection.

        Args:
            waveform: Audio waveform as numpy array

        Returns:
            Tuple of (scores, embeddings, spectrogram)

        Raises:
            RuntimeError: If model is not loaded or inference fails
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Memory protection: chunk large audio files based on system memory
        audio_duration_seconds = len(waveform) / self.target_sample_rate
        
        # Dynamic chunk size based on system memory
        if hasattr(self, '_system_memory_gb'):
            if self._system_memory_gb <= 8:
                max_duration_seconds = 600  # 10 minutes for low memory systems
            elif self._system_memory_gb <= 16:
                max_duration_seconds = 1200  # 20 minutes for medium memory systems
            else:
                max_duration_seconds = 1800  # 30 minutes for high memory systems
        else:
            max_duration_seconds = 1200  # Conservative default
        
        max_samples = max_duration_seconds * self.target_sample_rate
        
        # Check memory before processing
        self._check_memory_before_processing(audio_duration_seconds)
        
        if len(waveform) > max_samples:
            print(f"[INFO] Large audio detected ({audio_duration_seconds:.1f}s)")
            print(f"Processing in chunks of {max_duration_seconds}s for memory safety...")
            return self._run_chunked_inference(waveform, max_samples)
        
        print("Running YAMNet inference...")
        return self._run_single_inference(waveform)

    def _run_single_inference(self, waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on a single chunk."""
        try:
            # Use float32 for optimal performance (compatible with mixed precision)
            waveform_tensor = tf.constant(waveform, dtype=tf.float32)

            # Run inference with GPU acceleration (if available)
            scores, embeddings, spectrogram = self.model(waveform_tensor)

            print(f"Inference complete. Generated {scores.shape[0]} predictions")
            print("Each prediction covers ~0.96 seconds of audio")

            return scores.numpy(), embeddings.numpy(), spectrogram.numpy()

        except tf.errors.ResourceExhaustedError as e:
            if "OOM" in str(e):
                print(f"[WARNING] GPU out of memory. Falling back to CPU for this inference...")
                print(f"Audio duration: {len(waveform) / self.target_sample_rate:.1f}s")
                # Try CPU fallback
                try:
                    with tf.device("/CPU:0"):
                        waveform_tensor = tf.constant(waveform, dtype=tf.float32)
                        scores, embeddings, spectrogram = self.model(waveform_tensor)
                        print("[OK] CPU fallback successful")
                        return scores.numpy(), embeddings.numpy(), spectrogram.numpy()
                except Exception as cpu_e:
                    raise RuntimeError(
                        f"Both GPU and CPU inference failed. Audio too large. "
                        f"Try processing shorter segments. GPU error: {e}, CPU error: {cpu_e}"
                    ) from e
            raise RuntimeError(f"YAMNet inference failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"YAMNet inference failed: {e}") from e

    def _run_chunked_inference(
        self, waveform: np.ndarray, chunk_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on large audio in chunks to prevent OOM."""
        all_scores = []
        all_embeddings = []
        all_spectrograms = []
        
        num_chunks = int(np.ceil(len(waveform) / chunk_size))
        overlap_samples = int(0.96 * self.target_sample_rate)  # YAMNet window size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size + overlap_samples, len(waveform))
            chunk = waveform[start_idx:end_idx]
            
            print(f"Processing chunk {i+1}/{num_chunks} ({len(chunk) / self.target_sample_rate:.1f}s)")
            
            try:
                scores, embeddings, spectrogram = self._run_single_inference(chunk)
                
                # Remove overlap from all but first chunk
                if i > 0:
                    overlap_frames = int(overlap_samples / (self.target_sample_rate * 0.48))  # 0.48s hop
                    scores = scores[overlap_frames:]
                    embeddings = embeddings[overlap_frames:]
                    spectrogram = spectrogram[overlap_frames:]
                
                all_scores.append(scores)
                all_embeddings.append(embeddings)
                all_spectrograms.append(spectrogram)
                
                # Force garbage collection between chunks
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"[WARNING] Chunk {i+1} failed: {e}")
                continue
        
        if not all_scores:
            raise RuntimeError("All chunks failed to process")
        
        # Concatenate all results
        final_scores = np.concatenate(all_scores, axis=0)
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_spectrograms = np.concatenate(all_spectrograms, axis=0)
        
        print(f"[OK] Chunked processing complete. Total predictions: {final_scores.shape[0]}")
        return final_scores, final_embeddings, final_spectrograms
    
    def _check_memory_before_processing(self, audio_duration_seconds: float) -> None:
        """Check system memory before processing and warn if insufficient."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            # Estimate memory usage (rough calculation)
            # Audio: 16kHz * 4 bytes * duration * 3 (waveform + scores + embeddings)
            estimated_memory_gb = (audio_duration_seconds * 16000 * 4 * 3) / (1024**3)
            
            if hasattr(self, '_safe_memory_threshold_gb'):
                if available_gb < self._safe_memory_threshold_gb:
                    print(f"[WARNING] Low available memory ({available_gb:.1f}GB)")
                    print(f"Estimated processing needs: {estimated_memory_gb:.1f}GB")
                    print("Consider closing other applications or processing shorter audio segments")
                    
                    # Auto-reduce batch size for this processing
                    if self.batch_size > 500:
                        old_batch_size = self.batch_size
                        self.batch_size = 500
                        print(f"[AUTO] Reduced batch size from {old_batch_size} to {self.batch_size} for this session")
                        
        except ImportError:
            pass  # psutil not available, skip memory check

    def detect_gongs(
        self,
        scores: np.ndarray,
        confidence_threshold: float = 0.94,
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
        confidence_threshold: float = 0.94,
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

        # Process embeddings in batches for maximum performance
        for batch_start in range(0, len(embeddings), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(embeddings))
            batch_embeddings = embeddings[batch_start:batch_end]

            # Reshape all embeddings in batch for classifier prediction (optimized)
            batch_embeddings_reshaped = batch_embeddings.reshape(
                -1, batch_embeddings.shape[1]
            )

            # Use TensorFlow ops for GPU acceleration if available
            with tf.device(''):  # Let TensorFlow choose best device (GPU if available)
                # Get predictions and confidences for entire batch
                predictions = self.trained_classifier.predict(batch_embeddings_reshaped)
                probabilities = self.trained_classifier.predict_proba(
                    batch_embeddings_reshaped
                )
            confidences = probabilities[
                :, 1
            ]  # Probability for positive class (gong = 1)

            # Vectorized processing for better performance
            # Create masks for filtering
            positive_mask = predictions == 1
            min_threshold_mask = confidences > confidence_threshold
            
            if max_confidence_threshold is not None:
                max_threshold_mask = confidences < max_confidence_threshold
                valid_mask = positive_mask & min_threshold_mask & max_threshold_mask
            else:
                valid_mask = positive_mask & min_threshold_mask
            
            # Get valid indices and confidences
            valid_indices = np.where(valid_mask)[0]
            valid_confidences = confidences[valid_mask]
            
            # Vectorized calculation of timestamps
            global_indices = batch_start + valid_indices
            window_starts = global_indices * hop_length
            display_timestamps = window_starts + (window_duration / 2)
            
            # Add all valid detections at once
            for i, (window_start, confidence, display_timestamp) in enumerate(
                zip(window_starts, valid_confidences, display_timestamps)
            ):
                detections.append((float(window_start), float(confidence), float(display_timestamp)))

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
        gpus = tf.config.list_physical_devices("GPU")
        gpu_details = []
        if gpus:
            for gpu in gpus:
                try:
                    gpu_details.append({
                        "name": gpu.name,
                        "device_type": gpu.device_type,
                        "memory_growth": True
                    })
                except:
                    gpu_details.append({"name": "GPU", "device_type": "GPU", "memory_growth": True})
        
        return {
            "batch_size": self.batch_size,
            "use_trained_classifier": self.use_trained_classifier,
            "tensorflow_threads": {
                "inter_op": tf.config.threading.get_inter_op_parallelism_threads(),
                "intra_op": tf.config.threading.get_intra_op_parallelism_threads(),
            },
            "available_devices": {
                "cpu": len(tf.config.list_physical_devices("CPU")),
                "gpu": len(gpus),
                "gpu_details": gpu_details,
            },
            "mixed_precision": tf.keras.mixed_precision.global_policy().name,
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
