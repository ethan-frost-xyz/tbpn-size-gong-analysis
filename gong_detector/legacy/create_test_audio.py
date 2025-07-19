#!/usr/bin/env python3
"""
Create test audio files for spectrogram comparison.

This script generates simple test audio files to demonstrate
the spectrogram comparison functionality.
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Tuple


def create_simple_tone(frequency: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Create a simple sine wave tone.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio data as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    return np.sin(2 * np.pi * frequency * t)


def create_gong_like_sound(duration: float = 2.0, sample_rate: int = 16000) -> np.ndarray:
    """
    Create a gong-like sound with multiple harmonics and decay.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio data as numpy array
    """
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Base frequency and harmonics
    base_freq = 250  # Hz
    harmonics = [1, 2, 3, 4, 6, 8]  # Harmonic ratios
    
    # Create gong sound with multiple harmonics
    gong = np.zeros_like(t)
    for i, harmonic in enumerate(harmonics):
        freq = base_freq * harmonic
        amplitude = 1.0 / (i + 1)  # Decreasing amplitude for higher harmonics
        gong += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Add exponential decay
    decay = np.exp(-2 * t)
    gong *= decay
    
    # Normalize
    gong = gong / np.max(np.abs(gong)) * 0.8
    
    return gong


def create_reference_gong(duration: float = 1.5, sample_rate: int = 16000) -> np.ndarray:
    """
    Create a reference gong sound (different characteristics).
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio data as numpy array
    """
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Different base frequency
    base_freq = 400  # Hz
    harmonics = [1, 1.5, 2, 3, 5]  # Different harmonic structure
    
    # Create reference gong
    gong = np.zeros_like(t)
    for i, harmonic in enumerate(harmonics):
        freq = base_freq * harmonic
        amplitude = 0.8 / (i + 1)
        gong += amplitude * np.sin(2 * np.pi * freq * t)
    
    # Different decay characteristic
    decay = np.exp(-1.5 * t)
    gong *= decay
    
    # Normalize
    gong = gong / np.max(np.abs(gong)) * 0.7
    
    return gong


def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = 16000) -> None:
    """
    Save audio data to WAV file.
    
    Args:
        audio: Audio data
        file_path: Output file path
        sample_rate: Sample rate in Hz
    """
    import soundfile as sf
    sf.write(file_path, audio, sample_rate)
    print(f"Saved: {file_path}")


def main() -> None:
    """Create test audio files."""
    # Ensure samples directory exists
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)
    
    print("Creating test audio files...")
    
    # Create TBPN-like gong
    tbpn_gong = create_gong_like_sound(duration=2.5)
    save_audio(tbpn_gong, "samples/tbpn_gong.wav")
    
    # Create reference gong
    reference_gong = create_reference_gong(duration=2.0)
    save_audio(reference_gong, "samples/reference_gong.wav")
    
    # Create a simple tone for comparison
    simple_tone = create_simple_tone(frequency=1000, duration=1.0)
    save_audio(simple_tone, "samples/simple_tone.wav")
    
    print("\nTest audio files created:")
    print("- samples/tbpn_gong.wav (2.5s gong-like sound)")
    print("- samples/reference_gong.wav (2.0s reference gong)")
    print("- samples/simple_tone.wav (1.0s 1kHz tone)")
    print("\nYou can now run:")
    print("python compare_spectrograms.py samples/tbpn_gong.wav samples/reference_gong.wav")


if __name__ == "__main__":
    main() 