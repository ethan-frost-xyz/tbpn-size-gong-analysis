#!/usr/bin/env python3
"""
Spectrogram comparison script for TBPN gong analysis.

This script visually compares the spectrograms of a real-world TBPN gong
from podcast audio against a reference YAMNet "gong" clip to help assess
frequency characteristics, duration, and harmonic structure.
"""

import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def load_audio(file_path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (default: 16000 for YAMNet compatibility)

    Returns:
        Tuple of (audio_data, sample_rate)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file cannot be loaded
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        # Ensure sample rate is an integer
        return audio, int(sr)
    except Exception as e:
        raise ValueError(f"Failed to load audio file {file_path}: {e}") from e


def compute_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute Mel spectrogram from audio data.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate

    Returns:
        Mel spectrogram in dB scale
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=1024, hop_length=512, n_mels=128
    )

    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def get_audio_stats(audio: np.ndarray, sr: int) -> dict:
    """
    Calculate basic audio statistics.

    Args:
        audio: Audio data as numpy array
        sr: Sample rate

    Returns:
        Dictionary containing audio statistics
    """
    duration = len(audio) / sr
    peak_amplitude = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    db_rms = 20 * np.log10(rms) if rms > 0 else -np.inf

    return {
        "duration": duration,
        "peak_amplitude": peak_amplitude,
        "rms": rms,
        "db_rms": db_rms,
    }


def plot_spectrograms(
    tbpn_audio: np.ndarray,
    reference_audio: np.ndarray,
    sr: int,
    tbpn_path: str,
    reference_path: str,
) -> None:
    """
    Create side-by-side spectrogram comparison plot.

    Args:
        tbpn_audio: TBPN gong audio data
        reference_audio: Reference gong audio data
        sr: Sample rate
        tbpn_path: Path to TBPN audio file (for title)
        reference_path: Path to reference audio file (for title)
    """
    # Compute spectrograms
    tbpn_spec = compute_mel_spectrogram(tbpn_audio, sr)
    reference_spec = compute_mel_spectrogram(reference_audio, sr)

    # Get statistics
    tbpn_stats = get_audio_stats(tbpn_audio, sr)
    reference_stats = get_audio_stats(reference_audio, sr)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot TBPN spectrogram
    img1 = librosa.display.specshow(
        tbpn_spec, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax1
    )
    ax1.set_title(f"TBPN Gong ({tbpn_stats['duration']:.2f}s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")

    # Plot reference spectrogram
    librosa.display.specshow(
        reference_spec, sr=sr, hop_length=512, x_axis="time", y_axis="mel", ax=ax2
    )
    ax2.set_title(f"Reference Gong ({reference_stats['duration']:.2f}s)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")

    # Add colorbar
    plt.colorbar(img1, format="%+2.0f dB")

    # Save plot
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "spectrogram_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSpectrogram comparison saved to: {output_file}")
    plt.close()

    # Print simple statistics
    print("\n=== Audio Statistics ===")
    print(
        f"TBPN Gong: {tbpn_stats['duration']:.3f}s, Peak: {tbpn_stats['peak_amplitude']:.3f}"
    )
    print(
        f"Reference Gong: {reference_stats['duration']:.3f}s, Peak: {reference_stats['peak_amplitude']:.3f}"
    )


def main() -> None:
    """Main function to run spectrogram comparison."""
    parser = argparse.ArgumentParser(
        description="Compare spectrograms of TBPN gong and reference audio"
    )
    parser.add_argument("tbpn_path", help="Path to TBPN gong audio file (.wav)")
    parser.add_argument(
        "reference_path", help="Path to reference gong audio file (.wav)"
    )

    args = parser.parse_args()

    try:
        # Load audio files
        print(f"Loading TBPN gong: {args.tbpn_path}")
        tbpn_audio, sr = load_audio(args.tbpn_path)

        print(f"Loading reference gong: {args.reference_path}")
        reference_audio, _ = load_audio(args.reference_path, target_sr=sr)

        # Create comparison plot
        print("Generating spectrogram comparison...")
        plot_spectrograms(
            tbpn_audio, reference_audio, sr, args.tbpn_path, args.reference_path
        )

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
