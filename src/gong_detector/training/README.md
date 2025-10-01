# Training Data & Quick Start Guide

This folder is for organizing training data and utilities for improving the gong detection model.

## Structure

- `data/raw_samples/positive/[video_title]/` – Positive clips grouped by episode
- `data/raw_samples/negative/` – Negative clips used for balance
- `data/processed/` – Saved embeddings and labels
- `data/models/` – Classifier checkpoints
- `scripts/` – Small helpers for embedding, training, and evaluation

## Quick Start

### Basic Gong Detection

```bash
# Activate virtual environment
source venv/bin/activate

# Detect gongs in YouTube video
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID"

# Save detected samples for training
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID" --save_positive_samples
```

### Manual Sample Collection

For collecting samples that YAMNet missed or for manual verification:

```bash
# Start interactive collection
python -m gong_detector.core.manual_sample_collector
```

This command loops over your timestamps, pulls a clip, and drops a 3-second WAV into the positive folder so you can review it later.

### Negative Sample Collection

For collecting non-gong audio segments for training data:

```bash
# Collect 5 negative samples from a single video
python -m gong_detector.core.negative_sample_collector "https://www.youtube.com/watch?v=VIDEO_ID"

# Collect 10 negative samples with custom threshold
python -m gong_detector.core.negative_sample_collector "https://www.youtube.com/watch?v=VIDEO_ID" --num_samples 10 --threshold 0.3

# Bulk processing - collect from all videos in data/tbpn_ytlinks/tbpn_youtube_links.txt
python -m gong_detector.core.bulk_process --collect_negative_samples --sample_count 10
```

The collector skips over detected gongs, snags quiet patches, and writes out 3-second WAVs in the negative folder using the same naming scheme as the positives.

## Collecting Training Samples

### YouTube Detection with Sample Collection

```bash
# Activate virtual environment
source venv/bin/activate

# Basic detection with sample collection
python -m gong_detector.core.detect_from_youtube "YOUR_YOUTUBE_URL" --save_positive_samples

# Tighten the threshold
python -m gong_detector.core.detect_from_youtube "YOUR_YOUTUBE_URL" --threshold 0.6 --save_positive_samples

# Focus on a specific window
python -m gong_detector.core.detect_from_youtube "YOUR_YOUTUBE_URL" --start_time 100 --duration 30 --save_positive_samples
```

### Human-in-the-Loop Workflow

1. **Collect YAMNet samples**: Use `detect_from_youtube` with `--save_positive_samples`
2. **Collect manual samples**: Use `manual_sample_collector` for interactive collection of missed detections
3. **Review and clean**: Manually review samples in training data folders
4. **Train model**: Run training pipeline on cleaned data
5. **Evaluate**: Test model performance on validation data

## Training Pipeline

```bash
# Extract embeddings from collected samples
python gong_detector/training/scripts/extract_embeddings.py

# Train classifier
python gong_detector/training/scripts/train_classifier.py

# Evaluate model
python gong_detector/training/scripts/evaluate_model.py
```

## Comprehensive Detection Data Collection

### NEW: Bulk Processing with Comprehensive CSV Output

The bulk processing script now generates a comprehensive CSV file containing all detection metadata:

```bash
# Basic bulk processing with comprehensive CSV
python gong_detector/core/bulk_process.py

# With custom settings and run name
python gong_detector/core/bulk_process.py --threshold 0.5 --save_positive_samples --run_name "tbpn_batch_1"
```

**CSV Schema:**
- `detection_id` - Unique identifier for each detection
- `video_url` - Original YouTube URL
- `video_title` - Video title from YouTube
- `upload_date` / `upload_date_formatted` - Video upload date (raw/formatted)
- `video_duration_seconds` - Total video length
- `detection_timestamp_seconds` / `detection_timestamp_formatted` - When gong occurred
- `confidence` - YAMNet confidence score
- `video_max_confidence` - Highest confidence in that video
- `detection_threshold` - Threshold used for detection
- `processing_date` / `processing_time` - When analysis was performed
- `notes` / `validated` - For future human review workflow
