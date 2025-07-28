# TBPN Size Gong Analysis

A AED system using YAMNet and a custom head to identify size gong sounds in TBPN episodes.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic detection
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID"

# Enhanced detection with trained classifier
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID" --use_version_one

# Bulk processing
python -m gong_detector.core.bulk_process --version_one
```

## Project Structure

### **Core Detection (`gong_detector/core/`)**
- `yamnet_runner.py` - YAMNet model integration with trained classifier support
- `detect_from_youtube.py` - Single video detection with optimized batch processing  
- `bulk_process.py` - Multi-video processing with performance optimizations
- `manual_sample_collector.py` - Interactive training sample collection
- `negative_sample_collector.py` - Non-gong sample collection for training
- `models/` - Trained classifier files (classifier.pkl, config.json)
- `tbpn_youtube_links.txt` - YouTube URLs for bulk processing

### **Training Pipeline (`gong_detector/training/`)**
- `scripts/extract_embeddings.py` - Extract YAMNet features from audio samples
- `scripts/train_classifier.py` - Train Random Forest on embeddings  
- `scripts/evaluate_model.py` - Evaluate trained model performance
- `data/validated_samples/` - Curated training data (positive/negative)
- `data/processed/` - Processed embeddings and labels
- `data/models/` - Trained model checkpoints

### **Supporting Modules**
- `audio_utils.py` - Audio processing utilities (dBFS, slicing, normalization)
- `youtube_utils.py` - YouTube download and audio conversion
- `results_utils.py` - CSV export and result formatting
- `comprehensive_csv.py` - Rich metadata CSV generation
- `convert_audio.py` - Audio format conversion utilities

## Key Features

### **Trained Classifier Integration**
- **99.3% accuracy** on validation data
- **Conservative thresholds** (0.925 default) for high precision
- **Batch processing** (5000 embeddings per batch) for 5-10x speedup
- **Multi-threading** (8 inter-op + 4 intra-op threads) for CPU optimization

### **Performance Optimizations**
- **Memory efficient** for 16GB+ systems
- **Real-time monitoring** of resource usage
- **Configurable batch sizes** for different hardware
- **Automatic cleanup** of temporary files

### **Training Data Workflow**
- **Automatic collection** with `--save_positive_samples`
- **Interactive manual collection** for missed detections
- **Negative sample collection** for balanced training data
- **Seamless integration** with training pipeline

## Core Commands

### **Detection**
```bash
# Basic YAMNet detection
python -m gong_detector.core.detect_from_youtube "URL"

# Enhanced with trained classifier
python -m gong_detector.core.detect_from_youtube "URL" --use_version_one

# Custom threshold
python -m gong_detector.core.detect_from_youtube "URL" --threshold 0.95

# Time segment
python -m gong_detector.core.detect_from_youtube "URL" --start_time 300 --duration 60
```

### **Bulk Processing**
```bash
# Process all URLs in tbpn_youtube_links.txt
python -m gong_detector.core.bulk_process --version_one

# Custom settings
python -m gong_detector.core.bulk_process --version_one --threshold 0.95 --batch_size 10000
```

### **Training Data Collection**
```bash
# Collect positive samples
python -m gong_detector.core.detect_from_youtube "URL" --save_positive_samples

# Interactive manual collection
python -m gong_detector.core.manual_sample_collector

# Collect negative samples
python -m gong_detector.core.negative_sample_collector "URL"
```

### **Training Pipeline**
```bash
# Extract embeddings from validated samples
python gong_detector/training/scripts/extract_embeddings.py

# Train classifier
python gong_detector/training/scripts/train_classifier.py

# Evaluate model
python gong_detector/training/scripts/evaluate_model.py
```

## Technical Architecture

### **YAMNet Integration**
- **Pre-trained model** from TensorFlow Hub
- **521 audio classes** with gong at index 172
- **0.48s temporal resolution** for precise detection
- **16kHz mono input** with automatic preprocessing

### **Trained Classifier**
- **Random Forest** on YAMNet embeddings
- **1024 features** per audio segment
- **Binary classification** (gong vs non-gong)
- **Confidence scoring** for threshold-based filtering

### **Audio Processing Pipeline**
- **YouTube download** via yt-dlp
- **FFmpeg conversion** to 16kHz WAV
- **TensorFlow preprocessing** (normalization, resampling)
- **Batch inference** with optimized threading

## Development

### **Code Quality**
- **Type annotations** on all functions
- **PEP 257 docstrings** for documentation
- **Ruff linting** for code style
- **Pytest testing** for reliability

### **Performance Monitoring**
```python
# Get performance configuration
detector = YAMNetGongDetector()
perf_info = detector.get_performance_info()
print(f"Threads: {perf_info['tensorflow_threads']}")
print(f"Batch size: {perf_info['batch_size']}")
```

## Dependencies

- **TensorFlow 2.19+** for YAMNet model
- **scikit-learn** for Random Forest classifier
- **yt-dlp** for YouTube downloads
- **FFmpeg** for audio processing
- **pandas/numpy** for data handling
