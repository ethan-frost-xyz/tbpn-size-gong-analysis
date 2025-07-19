"""Demo script for YAMNet gong detection.

This script demonstrates how to use the gong detection functionality
with different audio files and settings.
"""

import os
import sys
from typing import List

from detector import YAMNetGongDetector


def create_sample_audio() -> None:
    """Create a simple sample audio file for testing (sine wave)."""
    try:
        import numpy as np
        import scipy.io.wavfile as wavfile  # type: ignore
        
        print("Creating sample audio file for testing...")
        
        # Generate a simple test tone (440 Hz for 5 seconds)
        sample_rate = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Create a simple tone
        frequency = 440  # A4 note
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Save as WAV file
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write("sample_audio.wav", sample_rate, audio_int16)
        
        print("✅ Sample audio file created: sample_audio.wav")
        
    except ImportError:
        print("❌ scipy not available. Please install: pip install scipy")
    except Exception as e:
        print(f"❌ Failed to create sample audio: {e}")


def demo_basic_detection(audio_path: str) -> None:
    """Demonstrate basic gong detection.
    
    Args:
        audio_path: Path to the audio file to analyze
    """
    print(f"\n🔍 Demo: Basic Gong Detection")
    print(f"Audio file: {audio_path}")
    print("-" * 50)
    
    detector = YAMNetGongDetector()
    
    try:
        # Load model
        detector.load_model()
        
        # Process audio
        waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)
        
        # Run inference
        scores, _, _ = detector.run_inference(waveform)
        
        # Detect gongs with different thresholds
        thresholds = [0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            print(f"\n📊 Testing threshold: {threshold}")
            detections = detector.detect_gongs(scores, confidence_threshold=threshold)
            
            if detections:
                print(f"Found {len(detections)} detections:")
                for timestamp, confidence in detections:
                    print(f"  {timestamp:6.2f}s - confidence: {confidence:.4f}")
            else:
                print("  No gongs detected at this threshold")
                
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def demo_class_exploration(audio_path: str) -> None:
    """Demonstrate exploration of all YAMNet classes for the audio.
    
    Args:
        audio_path: Path to the audio file to analyze
    """
    print(f"\n🔬 Demo: YAMNet Class Exploration")
    print(f"Audio file: {audio_path}")
    print("-" * 50)
    
    detector = YAMNetGongDetector()
    
    try:
        # Load model
        detector.load_model()
        
        # Process audio
        waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)
        
        # Run inference
        scores, _, _ = detector.run_inference(waveform)
        
        # Ensure class names are loaded
        if detector.class_names is None:
            print("❌ Class names not loaded")
            return
            
        # Find top classes across all time frames
        mean_scores = scores.mean(axis=0)
        top_indices = mean_scores.argsort()[-10:][::-1]  # Top 10 classes
        
        print("\n🏆 Top 10 detected classes (average confidence):")
        for i, idx in enumerate(top_indices, 1):
            class_name = detector.class_names[idx]
            confidence = mean_scores[idx]
            print(f"  {i:2d}. {class_name:<25} - {confidence:.4f}")
            
        # Show gong-specific info
        gong_confidence = mean_scores[detector.gong_class_index]
        gong_rank = len(mean_scores) - mean_scores.argsort().argsort()[detector.gong_class_index]
        
        print(f"\n🎯 Gong class analysis:")
        print(f"  Class: {detector.class_names[detector.gong_class_index]}")
        print(f"  Average confidence: {gong_confidence:.4f}")
        print(f"  Rank: {gong_rank}/{len(detector.class_names)}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main() -> None:
    """Main demo execution."""
    print("🎵 YAMNet Gong Detection Demo")
    print("=" * 50)
    
    # Check for audio file argument
    audio_files: List[str] = []
    
    if len(sys.argv) > 1:
        audio_files = sys.argv[1:]
    else:
        # Default test files to try
        default_files = ["audio.wav", "sample_audio.wav", "test.wav"]
        audio_files = [f for f in default_files if os.path.exists(f)]
        
        # If no files exist, create a sample
        if not audio_files:
            create_sample_audio()
            if os.path.exists("sample_audio.wav"):
                audio_files = ["sample_audio.wav"]
    
    if not audio_files:
        print("❌ No audio files found!")
        print("\nUsage:")
        print("  python demo.py <audio_file1> [audio_file2] ...")
        print("  python demo.py  # Will create sample_audio.wav for testing")
        return
    
    # Run demos on each audio file
    for audio_file in audio_files:
        if os.path.exists(audio_file):
            demo_basic_detection(audio_file)
            demo_class_exploration(audio_file)
        else:
            print(f"❌ Audio file not found: {audio_file}")
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main() 