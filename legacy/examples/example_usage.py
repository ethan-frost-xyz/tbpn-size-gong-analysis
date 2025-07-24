#!/usr/bin/env python3
"""
Example usage of the spectrogram comparison script.

This demonstrates how to use the compare_spectrograms module programmatically.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from compare_spectrograms import get_audio_stats, load_audio, plot_spectrograms


def example_comparison(tbpn_path: str, reference_path: str) -> None:
    """
    Show how to compare two audio files.

    Args:
        tbpn_path: Path to TBPN gong audio file
        reference_path: Path to reference gong audio file
    """
    print("=== Spectrogram Comparison Example ===")

    try:
        # Load audio files
        print(f"Loading TBPN gong: {tbpn_path}")
        tbpn_audio, sr = load_audio(tbpn_path)

        print(f"Loading reference gong: {reference_path}")
        reference_audio, _ = load_audio(reference_path, target_sr=sr)

        # Get basic stats
        tbpn_stats = get_audio_stats(tbpn_audio, sr)
        reference_stats = get_audio_stats(reference_audio, sr)

        print(f"\nTBPN Gong duration: {tbpn_stats['duration']:.3f}s")
        print(f"Reference Gong duration: {reference_stats['duration']:.3f}s")

        # Create comparison plot
        print("\nGenerating spectrogram comparison...")
        plot_spectrograms(tbpn_audio, reference_audio, sr, tbpn_path, reference_path)

    except FileNotFoundError as e:
        print(f"Error: Audio file not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


def main() -> None:
    """Demonstrate usage."""
    # Example file paths (update these with your actual files)
    tbpn_file = "samples/tbpn_gong.wav"
    reference_file = "samples/reference_gong.wav"

    print("Spectrogram Comparison Example")
    print("=" * 40)
    print(f"TBPN file: {tbpn_file}")
    print(f"Reference file: {reference_file}")
    print()

    # Check if files exist
    if not Path(tbpn_file).exists():
        print(f"Warning: TBPN file not found: {tbpn_file}")
        print("Please place your TBPN gong audio file in the samples/ directory")
        return

    if not Path(reference_file).exists():
        print(f"Warning: Reference file not found: {reference_file}")
        print("Please place your reference gong audio file in the samples/ directory")
        return

    # Run comparison
    example_comparison(tbpn_file, reference_file)


if __name__ == "__main__":
    main()
