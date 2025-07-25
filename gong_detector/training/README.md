# Training Data

This folder is for organizing training data and utilities for improving the gong detection model.

## Structure

- `data/raw_samples/positive/[video_title]/` - Gong samples organized by video
- `data/raw_samples/negative/` - Non-gong samples for training
- `data/processed/` - Processed embeddings and labels
- `data/models/` - Trained model checkpoints
- `scripts/` - Data processing and training scripts

## Collecting Training Samples

### YouTube Detection with Sample Collection

```bash
# Activate virtual environment
source venv/bin/activate

# Basic detection with sample collection
python -m gong_detector.core.detect_from_youtube "YOUR_YOUTUBE_URL" --save_positive_samples

# With custom threshold (more strict)
python -m gong_detector.core.detect_from_youtube "YOUR_YOUTUBE_URL" --threshold 0.6 --save_positive_samples

# With time segment
python -m gong_detector.core.detect_from_youtube "YOUR_YOUTUBE_URL" --start_time 100 --duration 30 --save_positive_samples
```

### Human-in-the-Loop Workflow

1. Run detection with `--save_positive_samples`
2. Review samples in `data/raw_samples/positive/[video_title]/`
3. Keep good samples, delete false positives
4. Repeat until you have 50 confirmed samples across multiple videos
5. Run training pipeline

## Training Pipeline

```bash
# Extract embeddings from samples
python gong_detector/training/scripts/extract_embeddings.py

# Train classifier
python gong_detector/training/scripts/train_classifier.py

# Evaluate model
python gong_detector/training/scripts/evaluate_model.py
```
