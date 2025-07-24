"""Train a simple classifier on YAMNet embeddings for gong detection.

This script loads the extracted embeddings and trains a scikit-learn
classifier to distinguish between gong and non-gong sounds.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_training_data(processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load embeddings and labels from CSV files.

    Args:
        processed_dir: Directory containing processed data files

    Returns:
        Tuple of (embeddings, labels) arrays
    """
    # Load embeddings
    embeddings_df = pd.read_csv(processed_dir / "embeddings.csv")
    embeddings = embeddings_df.values

    # Load labels
    labels_df = pd.read_csv(processed_dir / "labels.csv")
    labels = labels_df["label"].values

    print(f"Loaded {len(embeddings)} samples with {embeddings.shape[1]} features")
    return embeddings, labels


def train_model(x_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Args:
        x_train: Training embeddings
        y_train: Training labels

    Returns:
        Trained classifier
    """
    print("Training Random Forest classifier...")

    # Simple Random Forest - good balance of performance and simplicity
    classifier = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_depth=10,  # Prevent overfitting
        random_state=42,  # Reproducible results
        n_jobs=-1,  # Use all CPU cores
    )

    classifier.fit(x_train, y_train)
    print("✓ Training complete")

    return classifier


def evaluate_model(
    classifier: RandomForestClassifier, x_test: np.ndarray, y_test: np.ndarray
) -> dict[str, float]:
    """Evaluate the trained classifier.

    Args:
        classifier: Trained classifier
        x_test: Test embeddings
        y_test: Test labels

    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model...")

    # Make predictions
    y_pred = classifier.predict(x_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Print detailed results
    print(f"\n{'=' * 40}")
    print("MODEL PERFORMANCE")
    print(f"{'=' * 40}")
    print(f"Accuracy: {accuracy:.3f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=["Non-Gong", "Gong"]))

    return {"accuracy": accuracy}


def save_model(
    classifier: RandomForestClassifier, metrics: dict[str, float], models_dir: Path
) -> None:
    """Save the trained classifier and configuration.

    Args:
        classifier: Trained classifier to save
        metrics: Evaluation metrics
        models_dir: Directory to save model files
    """
    models_dir.mkdir(exist_ok=True)

    # Save the classifier
    model_path = models_dir / "classifier.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(classifier, f)

    # Save configuration and metrics
    config = {
        "model_type": "RandomForestClassifier",
        "model_params": classifier.get_params(),
        "performance": metrics,
        "feature_count": classifier.n_features_in_,
    }

    config_path = models_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Config saved to: {config_path}")


def main() -> None:
    """Train and save the gong classifier."""
    print("Starting classifier training...")

    # Setup paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"
    processed_dir = data_dir / "processed"
    models_dir = data_dir / "models"

    # Check if processed data exists
    if not (processed_dir / "embeddings.csv").exists():
        print("Error: No embeddings found! Run extract_embeddings.py first.")
        return

    # Load training data
    embeddings, labels = load_training_data(processed_dir)

    # Check if we have both classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Error: Need both positive and negative samples to train!")
        return

    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,  # 20% for testing
        random_state=42,  # Reproducible split
        stratify=labels,  # Keep class balance
    )

    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    # Train the model
    classifier = train_model(x_train, y_train)

    # Evaluate performance
    metrics = evaluate_model(classifier, x_test, y_test)

    # Save everything
    save_model(classifier, metrics, models_dir)

    print(f"\n{'=' * 40}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 40}")
    print("Your gong classifier is ready to use!")


if __name__ == "__main__":
    main()
