"""Extract YAMNet embeddings from training audio samples.

This script processes all audio files in the raw_samples folders,
extracts YAMNet embeddings, and saves them for training.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add core module to path
sys.path.append(str(Path(__file__).parent.parent.parent / "core"))
from core.yamnet_runner import YAMNetGongDetector


def get_audio_files(folder_path: Path) -> list[Path]:
    """Get all audio files from a folder, including subdirectories.

    Args:
        folder_path: Path to folder containing audio files

    Returns:
        List of audio file paths
    """
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac"}
    audio_files = []
    
    # If folder doesn't exist, return empty list
    if not folder_path.exists():
        return audio_files
    
    # Recursively find all audio files
    for file_path in folder_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
            audio_files.append(file_path)
    
    return audio_files


def extract_embeddings_from_file(
    detector: YAMNetGongDetector, audio_path: Path
) -> np.ndarray:
    """Extract YAMNet embeddings from a single audio file.

    Args:
        detector: Initialized YAMNet detector
        audio_path: Path to audio file

    Returns:
        Array of embeddings (mean across time)
    """
    try:
        # Load and process audio
        waveform, _ = detector.load_and_preprocess_audio(str(audio_path))

        # Run inference to get embeddings
        _, embeddings, _ = detector.run_inference(waveform)

        # Take mean across time dimension to get single embedding
        mean_embedding = np.mean(embeddings, axis=0)

        print(f"✓ Processed: {audio_path.name}")
        return mean_embedding

    except Exception as e:
        print(f"✗ Failed to process {audio_path.name}: {e}")
        return np.array([])


def main() -> None:
    """Extract embeddings from all training samples."""
    print("Starting embedding extraction...")

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    positive_dir = data_dir / "raw_samples" / "positive"
    negative_dir = data_dir / "raw_samples" / "negative"
    processed_dir = data_dir / "processed"

    # Create processed directory if it doesn't exist
    processed_dir.mkdir(exist_ok=True)

    # Initialize YAMNet detector
    detector = YAMNetGongDetector()
    detector.load_model()

    # Collect all embeddings and labels
    all_embeddings: list[np.ndarray] = []
    all_labels: list[int] = []
    all_filenames: list[str] = []

    # Process positive samples (gongs)
    print(f"\nProcessing positive samples from: {positive_dir}")
    positive_files = get_audio_files(positive_dir)

    for audio_file in positive_files:
        embedding = extract_embeddings_from_file(detector, audio_file)
        if len(embedding) > 0:
            all_embeddings.append(embedding)
            all_labels.append(1)  # 1 = gong
            all_filenames.append(audio_file.name)

    # Process negative samples (non-gongs)
    print(f"\nProcessing negative samples from: {negative_dir}")
    negative_files = get_audio_files(negative_dir)

    for audio_file in negative_files:
        embedding = extract_embeddings_from_file(detector, audio_file)
        if len(embedding) > 0:
            all_embeddings.append(embedding)
            all_labels.append(0)  # 0 = not gong
            all_filenames.append(audio_file.name)

    # Convert to arrays
    if not all_embeddings:
        print(
            "No embeddings extracted! Please add audio files to the raw_samples folders."
        )
        return

    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)

    # Save embeddings
    embeddings_df = pd.DataFrame(embeddings_array)
    embeddings_df.to_csv(processed_dir / "embeddings.csv", index=False)

    # Save labels
    labels_df = pd.DataFrame({"filename": all_filenames, "label": labels_array})
    labels_df.to_csv(processed_dir / "labels.csv", index=False)

    # Save metadata
    metadata_df = pd.DataFrame(
        {
            "total_samples": [len(all_embeddings)],
            "positive_samples": [sum(labels_array)],
            "negative_samples": [len(labels_array) - sum(labels_array)],
            "embedding_dimensions": [embeddings_array.shape[1]],
        }
    )
    metadata_df.to_csv(processed_dir / "metadata.csv", index=False)

    # Print summary
    print(f"\n{'=' * 50}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Total samples: {len(all_embeddings)}")
    print(f"Positive (gong): {sum(labels_array)}")
    print(f"Negative (non-gong): {len(labels_array) - sum(labels_array)}")
    print(f"Embedding dimensions: {embeddings_array.shape[1]}")
    print(f"\nFiles saved to: {processed_dir}")
    print("- embeddings.csv")
    print("- labels.csv")
    print("- metadata.csv")


if __name__ == "__main__":
    main()
