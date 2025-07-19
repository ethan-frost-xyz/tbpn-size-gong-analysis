"""YAMNet-based gong detection module.

This module provides a comprehensive YAMNet-based gong detection system
for audio analysis. It handles audio preprocessing, model inference, and
confidence-based detection of gong sounds.
"""

import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


class YAMNetGongDetector:
    """YAMNet-based gong sound detector for audio analysis.
    
    This class encapsulates the functionality needed to detect gong sounds
    in audio files using Google's YAMNet model from TensorFlow Hub.
    """
    
    def __init__(self) -> None:
        """Initialize the YAMNet gong detector.
        
        Sets up the detector with default parameters and prepares it
        for model loading and inference.
        """
        self.model: Optional[Any] = None
        self.class_names: Optional[List[str]] = None
        self.gong_class_index: int = 138  # YAMNet class index for "gong"
        
    def load_model(self) -> None:
        """Load the YAMNet model from TensorFlow Hub.
        
        Downloads and initializes the YAMNet model and its associated
        class names for audio classification.
        
        Raises:
            RuntimeError: If model loading fails
        """
        print("Loading YAMNet model from TensorFlow Hub...")
        try:
            # Load YAMNet model and class names
            self.model = hub.load("https://tfhub.dev/google/yamnet/1")
            
            # Load class names
            if self.model is not None:
                class_map_path = self.model.class_map_path().numpy().decode('utf-8')
                self.class_names = list(pd.read_csv(class_map_path)['display_name'])
                
                print(f"Model loaded successfully. Total classes: {len(self.class_names)}")
                print(f"Gong class (index {self.gong_class_index}): {self.class_names[self.gong_class_index]}")
            else:
                raise RuntimeError("Failed to load model")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
            
    def load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
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
            # Load audio file using TensorFlow
            audio_binary = tf.io.read_file(audio_path)
            audio_decoded = tf.audio.decode_wav(
                audio_binary,
                desired_channels=1,  # Convert to mono
                desired_samples=-1   # Keep original length
            )
            
            # Extract waveform and sample rate (TensorFlow returns tuple-like object)
            waveform = audio_decoded[0].numpy().flatten()  # type: ignore
            sample_rate = int(audio_decoded[1].numpy())  # type: ignore
            
            # Ensure 16kHz sample rate (YAMNet requirement)
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate}Hz to 16000Hz...")
                # Simple linear interpolation resampling
                target_length = int(len(waveform) * 16000 / sample_rate)
                indices = np.linspace(0, len(waveform) - 1, target_length)
                waveform = np.interp(indices, np.arange(len(waveform)), waveform)
                sample_rate = 16000
                
            # Normalize audio to [-1, 1] range
            waveform = waveform.astype(np.float32)
            if np.max(np.abs(waveform)) > 0:
                waveform = waveform / np.max(np.abs(waveform))
                
            print(f"Audio loaded: {len(waveform)} samples at {sample_rate}Hz")
            print(f"Duration: {len(waveform) / sample_rate:.2f} seconds")
            
            return waveform, sample_rate
            
        except Exception as e:
            raise ValueError(f"Failed to process audio file {audio_path}: {e}")
            
    def run_inference(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            # Convert to TensorFlow tensor
            waveform_tensor = tf.constant(waveform, dtype=tf.float32)
            
            # Run inference
            scores, embeddings, spectrogram = self.model(waveform_tensor)
            
            print(f"Inference complete. Generated {scores.shape[0]} predictions")
            print(f"Each prediction covers ~0.96 seconds of audio")
            
            return scores.numpy(), embeddings.numpy(), spectrogram.numpy()
            
        except Exception as e:
            raise RuntimeError(f"YAMNet inference failed: {e}")
            
    def detect_gongs(
        self, 
        scores: np.ndarray, 
        confidence_threshold: float = 0.5,
        sample_rate: int = 16000
    ) -> List[Tuple[float, float]]:
        """Detect gong sounds based on YAMNet scores.
        
        Args:
            scores: YAMNet prediction scores array
            confidence_threshold: Minimum confidence for gong detection
            sample_rate: Audio sample rate in Hz (not used in calculation but kept for API consistency)
            
        Returns:
            List of (timestamp, confidence) tuples for detected gongs
        """
        print(f"Detecting gongs with confidence threshold: {confidence_threshold}")
        
        # Extract gong class scores
        gong_scores = scores[:, self.gong_class_index]
        
        # Find detections above threshold
        detections: List[Tuple[float, float]] = []
        for i, confidence in enumerate(gong_scores):
            if confidence > confidence_threshold:
                # Calculate timestamp (YAMNet produces ~1 prediction per 0.96 seconds)
                timestamp = i * 0.96  # seconds
                detections.append((timestamp, float(confidence)))
                
        print(f"Found {len(detections)} gong detections above threshold")
        
        return detections
        
    def print_detections(self, detections: List[Tuple[float, float]]) -> None:
        """Print gong detections in a formatted table.
        
        Args:
            detections: List of (timestamp, confidence) tuples to display
        """
        if not detections:
            print("No gong detections found.")
            return
            
        print("\n" + "="*50)
        print("GONG DETECTIONS")
        print("="*50)
        print(f"{'Timestamp (s)':<15} {'Confidence':<12}")
        print("-" * 27)
        
        for timestamp, confidence in detections:
            print(f"{timestamp:<15.2f} {confidence:<12.4f}")
            
        print("="*50)
        
    def detections_to_dataframe(self, detections: List[Tuple[float, float]]) -> pd.DataFrame:
        """Convert detections to a pandas DataFrame.
        
        Args:
            detections: List of (timestamp, confidence) tuples
            
        Returns:
            DataFrame with timestamp_seconds and confidence columns
        """
        if not detections:
            # Create empty DataFrame with proper columns
            df = pd.DataFrame()
            df['timestamp_seconds'] = pd.Series([], dtype='float64')
            df['confidence'] = pd.Series([], dtype='float64')
            return df
        
        # Create DataFrame from detections
        timestamps = [d[0] for d in detections]
        confidences = [d[1] for d in detections]
        df = pd.DataFrame()
        df['timestamp_seconds'] = timestamps
        df['confidence'] = confidences
        return df


def run_detection_pipeline(audio_path: str = "audio.wav") -> None:
    """Run the complete gong detection pipeline on an audio file.
    
    This is a convenience function that demonstrates the full workflow
    of loading the model, processing audio, and detecting gongs.
    
    Args:
        audio_path: Path to the audio file to analyze
        
    Raises:
        Various exceptions from the detection pipeline
    """
    print("🎵 Starting YAMNet Gong Detection Pipeline")
    print("="*50)
    
    try:
        # Initialize detector
        detector = YAMNetGongDetector()
        
        # Load model
        detector.load_model()
        
        # Verify gong class exists
        if detector.class_names is None:
            raise RuntimeError("Class names not loaded")
            
        assert detector.gong_class_index < len(detector.class_names), \
            f"Gong class index {detector.gong_class_index} out of range"
        assert "gong" in detector.class_names[detector.gong_class_index].lower(), \
            f"Class at index {detector.gong_class_index} is not gong: {detector.class_names[detector.gong_class_index]}"
        
        print(f"✅ Gong class verification passed: '{detector.class_names[detector.gong_class_index]}'")
        
        # Load and preprocess audio
        waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)
        
        # Run inference
        scores, embeddings, spectrogram = detector.run_inference(waveform)
        
        # Detect gongs
        detections = detector.detect_gongs(scores, confidence_threshold=0.5, sample_rate=sample_rate)
        
        # Print results
        detector.print_detections(detections)
        
        # Create DataFrame
        if detections:
            df = detector.detections_to_dataframe(detections)
            print(f"\n📊 Detection DataFrame shape: {df.shape}")
            print("First few detections:")
            print(df.head())
            
            # Optionally save to CSV
            output_path = "gong_detections.csv"
            df.to_csv(output_path, index=False)
            print(f"💾 Detections saved to: {output_path}")
        
        print("\n🎉 Detection pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Detection pipeline failed: {e}")
        raise 