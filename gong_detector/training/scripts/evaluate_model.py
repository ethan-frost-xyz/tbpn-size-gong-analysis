"""Evaluate the trained gong classifier on test data.

This script loads a trained classifier and evaluates its performance
on new audio samples to verify it works correctly.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Add core module to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from gong_detector.core.yamnet_runner import YAMNetGongDetector


def load_trained_model(models_dir: Path) -> tuple[object, dict]:
    """Load the trained classifier and its configuration.

    Args:
        models_dir: Directory containing saved model files

    Returns:
        Tuple of (classifier, config_dict)
    """
    model_path = models_dir / "classifier.pkl"
    config_path = models_dir / "config.json"

    if not model_path.exists():
        raise FileNotFoundError(
            "No trained model found! Run train_classifier.py first."
        )

    # Load classifier
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print(f"✓ Loaded {config['model_type']} with {config['feature_count']} features")
    print(f"✓ Training accuracy: {config['performance']['accuracy']:.3f}")

    return classifier, config


def test_on_audio_files(
    classifier: object,
    detector: YAMNetGongDetector,
    test_files: list[Path],
    expected_label: int,
) -> list[tuple[str, int, float]]:
    """Test classifier on a list of audio files.

    Args:
        classifier: Trained classifier
        detector: YAMNet detector for feature extraction
        test_files: List of audio file paths
        expected_label: Expected label (1 for gong, 0 for non-gong)

    Returns:
        List of (filename, predicted_label, confidence) tuples
    """
    results: list[tuple[str, int, float]] = []

    for audio_file in test_files:
        try:
            # Extract embedding from audio
            waveform, _ = detector.load_and_preprocess_audio(str(audio_file))
            _, embeddings, _ = detector.run_inference(waveform)
            mean_embedding = np.mean(embeddings, axis=0).reshape(1, -1)

            # Make prediction
            prediction = classifier.predict(mean_embedding)[0]
            confidence = classifier.predict_proba(mean_embedding)[0].max()

            results.append((audio_file.name, prediction, confidence))

            # Show result
            status = "✓" if prediction == expected_label else "✗"
            label_name = "Gong" if prediction == 1 else "Non-Gong"
            print(f"{status} {audio_file.name}: {label_name} ({confidence:.3f})")

        except Exception as e:
            print(f"✗ Failed to process {audio_file.name}: {e}")

    return results


def calculate_test_metrics(
    results: list[tuple[str, int, float]], expected_label: int
) -> dict[str, float]:
    """Calculate performance metrics from test results.

    Args:
        results: List of (filename, prediction, confidence) tuples
        expected_label: Expected label for all files

    Returns:
        Dictionary of performance metrics
    """
    if not results:
        return {}

    predictions = [result[1] for result in results]
    correct = sum(1 for pred in predictions if pred == expected_label)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0

    return {"accuracy": accuracy, "correct": correct, "total": total}


def main() -> None:
    """Evaluate the trained classifier on test audio files."""
    print("Starting model evaluation...")

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    models_dir = data_dir / "models"
    positive_dir = data_dir / "validated_samples" / "positive"
    negative_dir = data_dir / "validated_samples" / "negative"

    # Load trained model
    try:
        classifier, config = load_trained_model(models_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Initialize YAMNet detector
    detector = YAMNetGongDetector()
    detector.load_model()

    # Get audio files (including subdirectories for positive samples)
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac"}

    # For positive samples, search recursively through video folders
    positive_files = []
    if positive_dir.exists():
        for file_path in positive_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                positive_files.append(file_path)

    # For negative samples, search in the main directory
    negative_files = []
    if negative_dir.exists():
        for file_path in negative_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                negative_files.append(file_path)

    if not positive_files and not negative_files:
        print("No audio files found for testing!")
        return

    # Test on positive samples (gongs)
    print(f"\n{'=' * 50}")
    print("TESTING ON POSITIVE SAMPLES (Gongs)")
    print(f"{'=' * 50}")
    positive_results = test_on_audio_files(classifier, detector, positive_files, 1)
    positive_metrics = calculate_test_metrics(positive_results, 1)

    # Test on negative samples (non-gongs)
    print(f"\n{'=' * 50}")
    print("TESTING ON NEGATIVE SAMPLES (Non-Gongs)")
    print(f"{'=' * 50}")
    negative_results = test_on_audio_files(classifier, detector, negative_files, 0)
    negative_metrics = calculate_test_metrics(negative_results, 0)

    # Overall results
    print(f"\n{'=' * 50}")
    print("OVERALL EVALUATION RESULTS")
    print(f"{'=' * 50}")

    if positive_metrics:
        print(
            f"Positive samples: {positive_metrics['correct']}/{positive_metrics['total']} "
            f"({positive_metrics['accuracy']:.1%} accuracy)"
        )

    if negative_metrics:
        print(
            f"Negative samples: {negative_metrics['correct']}/{negative_metrics['total']} "
            f"({negative_metrics['accuracy']:.1%} accuracy)"
        )

    # Combined accuracy
    if positive_metrics and negative_metrics:
        total_correct = positive_metrics["correct"] + negative_metrics["correct"]
        total_samples = positive_metrics["total"] + negative_metrics["total"]
        overall_accuracy = total_correct / total_samples
        print(
            f"\nOverall accuracy: {total_correct}/{total_samples} ({overall_accuracy:.1%})"
        )

        # Save evaluation results
        eval_results = {
            "positive_accuracy": positive_metrics["accuracy"],
            "negative_accuracy": negative_metrics["accuracy"],
            "overall_accuracy": overall_accuracy,
            "total_samples_tested": total_samples,
        }

        eval_path = models_dir / "evaluation_results.json"
        with open(eval_path, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\n✓ Evaluation results saved to: {eval_path}")

    print(f"\n{'=' * 50}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
