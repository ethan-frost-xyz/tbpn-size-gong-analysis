# Quick Start: Gong Detection System

## **Core Commands**

### **Basic Detection:**
```bash
python -m gong_detector.core.pipeline.detection_pipeline "https://youtube.com/watch?v=VIDEO_ID"
```

### **With Trained Classifier (Enhanced):**
```bash
python -m gong_detector.core.pipeline.detection_pipeline "https://youtube.com/watch?v=VIDEO_ID" --use_version_one
```

### **Bulk Processing:**
```bash
python -m gong_detector.core.pipeline.bulk_processor --version_one
```

### **Sample Collection:**
```bash
# Collect positive samples for training
python -m gong_detector.core.pipeline.detection_pipeline "URL" --save_positive_samples

# Interactive manual collection
python -m gong_detector.core.training.manual_collector

# Collect negative samples
python -m gong_detector.core.training.negative_collector "URL"
```

## **Project Structure**

### **Core Detection (`/core`)**
- `detector/yamnet_runner.py` - YAMNet model integration with trained classifier support
- `pipeline/detection_pipeline.py` - Single video detection with optimized batch processing
- `pipeline/bulk_processor.py` - Multi-video processing with performance optimizations
- `training/manual_collector.py` - Interactive training sample collection
- `training/negative_collector.py` - Non-gong sample collection for training
- `utils/` - Audio processing, YouTube operations, and results utilities
- `data/` - CSV management and input data files
- `models/` - Trained classifier files (classifier.pkl, config.json)

### **Training Pipeline (`/training`)**
- `scripts/extract_embeddings.py` - Extract YAMNet features from audio samples
- `scripts/train_classifier.py` - Train Random Forest on embeddings
- `scripts/evaluate_model.py` - Evaluate trained model performance
- `data/validated_samples/` - Curated training data (positive/negative)

### **Performance Optimizations**
- **Batch Processing**: 5000 embeddings per batch (configurable)
- **Multi-threading**: 8 inter-op + 4 intra-op threads
- **Memory Efficient**: Optimized for 16GB+ systems
- **Conservative Defaults**: 0.925 threshold for high precision

## **Importable Modules**

### **Main API (from gong_detector.core)**
```python
from gong_detector.core import (
    # Core detector
    YAMNetGongDetector,
    
    # Detection pipeline
    detect_from_youtube_comprehensive,
    
    # Audio utilities
    compute_peak_dbfs, compute_rms_dbfs, compute_audio_levels,
    extract_audio_slice, get_slice_around_timestamp,
    analyze_audio_slice_levels, is_silent, normalize_waveform,
    get_audio_stats, DEFAULT_SAMPLE_RATE, SILENCE_FLOOR_DBFS,
    
    # Audio conversion
    convert_youtube_audio, validate_audio_file, get_audio_info,
    
    # YouTube utilities
    download_and_trim_youtube_audio, cleanup_old_temp_files,
    create_folder_name_from_date, create_folder_name_from_title,
    create_temp_audio_path, sanitize_title_for_folder, setup_directories,
    
    # Results utilities
    format_time, print_summary, save_positive_samples, save_results_to_csv,
    
    # Data management
    CSVManager, DetectionRecord,
    
    # Training utilities
    collect_negative_samples
)
```

### **Individual Modules**
```python
# Core detection engine
from gong_detector.core.detector import YAMNetGongDetector

# Processing pipelines
from gong_detector.core.pipeline import detect_from_youtube_comprehensive

# Utility functions
from gong_detector.core.utils import (
    compute_peak_dbfs, download_and_trim_youtube_audio, format_time
)

# Data management
from gong_detector.core.data import CSVManager, DetectionRecord

# Training utilities
from gong_detector.core.training import collect_negative_samples
```

## **Key Features**

- **Trained Classifier**: 99.3% accuracy on validation data
- **Batch Processing**: 5-10x faster than sequential processing
- **Conservative Detection**: High thresholds prevent false positives
- **Training Integration**: Seamless workflow from detection to model training
- **Performance Monitoring**: Real-time feedback on resource usage