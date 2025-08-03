# Training Data & Quick Start Guide

This folder is for organizing training data and utilities for improving the gong detection model.

## Structure

- `data/raw_samples/positive/[video_title]/` - Gong samples organized by video
- `data/raw_samples/negative/` - Non-gong samples for training
- `data/processed/` - Processed embeddings and labels
- `data/models/` - Trained model checkpoints
- `scripts/` - Data processing and training scripts

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

This tool:
- Runs in interactive loop asking for YouTube links and timestamps
- Downloads the full YouTube video
- Extracts a 3-second segment around your specified timestamp
- Saves it in the same format as YAMNet-detected samples
- Organizes samples by video title in `gong_detector/training/data/raw_samples/positive/`
- Asks if you want to continue after each sample

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

This tool:
- **Automatically detects gongs** using YAMNet to avoid them
- **Finds safe regions** far from detected gongs
- **Extracts random segments** from safe regions
- **Saves samples** to `gong_detector/training/data/raw_samples/negative/`
- **Uses same organization** as positive samples (date-based folders)
- **Configurable sample count** per video
- **Works with bulk processing** for multiple videos

**Output:**
- `negative_at_HH_MM_SS_s_01.wav`
- `negative_at_HH_MM_SS_s_02.wav`
- etc.

Each sample contains 3 seconds of audio (0.75s before + 2.25s after the selected timestamp), matching the positive sample format.

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

**CSV Output Features:**
- **Rich metadata**: Each detection includes video title, upload date, duration, confidence, timestamps
- **Unique IDs**: Every detection has a UUID for easy referencing
- **Extensible schema**: Future-ready with validation and notes fields
- **Summary statistics**: Automatic calculation of detection patterns
- **Saved in `results/csv_results/`**: Organized with timestamps and run names

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

**Example Output:**
```
CSV saved to: results/csv_results/comprehensive_detections_tbpn_batch_1_20250726_184200.csv
Total detections: 47
Videos processed: 12
Average confidence: 0.682
Confidence range: 0.401 - 0.950
```

This CSV is perfect for:
- **Finding missed gongs**: Look for videos with low detection counts
- **Analyzing patterns**: Study confidence distributions and timing
- **Quality control**: Identify edge cases and validation priorities
- **Training data curation**: Select best samples for model improvement

## Complete Workflow

1. **Collect YAMNet samples**: Use `detect_from_youtube` with `--save_positive_samples`
2. **Collect manual samples**: Use `manual_sample_collector` for interactive collection of missed detections
3. **Collect negative samples**: Use `negative_sample_collector` or `bulk_process --collect_negative_samples`
4. **Generate comprehensive CSV**: Use `bulk_process.py` for systematic data collection
5. **Analyze detection patterns**: Review CSV data for insights and edge cases
6. **Review and clean**: Manually review samples in training data folders
7. **Train model**: Run training pipeline on cleaned data
8. **Evaluate**: Test model performance on validation data
