# Quick Start: Gong Detection System

## **Master Menu Interface**

The `run_gong_detector.py` script provides an interactive menu system for accessing all gong detector functionality. It's located in the `src/gong_detector/` directory and serves as the main entry point for all gong detection tools.

**New: EBU R128 Compliance** - The system now provides complete broadcast-standard audio analysis with LUFS loudness and True Peak (dBTP) measurements.

### **Quick Start**

```bash
# In the project root, set up and activate a local venv (Python 3.12 recommended)
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate

# Launch the master menu
cd src/gong_detector && python run_gong_detector.py
```

### **Full Caching Workflow (Download All Files)**

To download and cache all YouTube URLs for fast offline testing and LUFS analysis:

```bash
# From project root, activate venv
source .venv/bin/activate

# Navigate to gong_detector directory
cd src/gong_detector

# Launch interactive menu
python run_gong_detector.py

# In the menu:
# 1. Select "Bulk Processing" (arrow keys + Enter)
# 2. Accept defaults for threshold (0.94) and trained classifier (y)
# 3. Choose save options:
#    - Save positive samples? (y/n) - choose based on need
#    - Save results to CSV file? (y) - recommended for analysis (includes LUFS + True Peak)
#    - Use local media (cache) if available? (y) - enables caching
#    - Local only (no downloads)? (n) - allow downloads to populate cache
```

This will:
- Process all 52 URLs from `data/tbpn_ytlinks/tbpn_youtube_links.txt`
- Download and cache raw audio to `data/local_media/raw/`
- Download and cache preprocessed audio to `data/local_media/preprocessed/`
- Save metadata to `data/local_media/index.json`
- Generate CSV results in `data/csv_results/` (with LUFS + True Peak analysis)
- Keep cached files for future offline runs and audio analysis

### **Fast Offline Testing (After Caching)**

Once files are cached, you can run repeated tests instantly:

```bash
# From src/gong_detector directory
python run_gong_detector.py

# In the menu, select "Bulk Processing" and choose:
# - Use local media (cache) if available? (y)
# - Local only (no downloads)? (y) - pure offline mode
```

Or via command line:
```bash
# From project root
source .venv/bin/activate
PYTHONPATH=src python -m gong_detector.core.pipeline.bulk_processor --local_only --version_one --csv
```

### **Menu Navigation**

- **↑/↓ Arrow Keys**: Navigate between menu options
- **Enter**: Select the highlighted option
- **q**: Quit the application

### **Available Options**

1. **Single Video Detection** - Detect gongs in a single YouTube video (supports local media cache)
2. **Bulk Processing** - Process multiple videos from `data/tbpn_ytlinks/tbpn_youtube_links.txt` (supports local media cache)
3. **Manual Sample Collection** - Extract specific timestamps for training data
4. **Negative Sample Collection** - Collect non-gong samples for training
5. **Audio Conversion** - Convert YouTube URLs or local files to WAV format
6. **Model Management** - Check YAMNet model status and configuration

### **Features**

- **Interactive Parameter Input**: All parameters are prompted interactively with sensible defaults (including “Use local media” and “Local only” toggles)
- **Error Handling**: Graceful error handling with user-friendly messages
- **Return to Menu**: After each operation, returns to the main menu
- **Keyboard Interrupt Support**: Ctrl+C to cancel operations

### **Example Workflow**

1. Start the menu: `python run_gong_detector.py`
2. Use arrow keys to select "Single Video Detection"
3. Press Enter
4. Enter YouTube URL when prompted
5. Set confidence threshold (or press Enter for default 0.94)
6. Choose other parameters as needed
7. Watch the detection run
8. Press Enter to return to main menu

### **Troubleshooting**

- **Import Errors**: Make sure you're in the virtual environment and in the correct directory
- **Terminal Issues**: The script requires a proper terminal (not piped input)
- **Model Loading**: Use "Model Management" to check if YAMNet is working correctly

### **File Location**

The master menu script is located at: `src/gong_detector/run_gong_detector.py`

---

## **Core Commands**

### **Basic Detection:**
```bash
python -m src.gong_detector.core.pipeline.detection_pipeline "https://youtube.com/watch?v=VIDEO_ID"
```

### **With Trained Classifier (Enhanced):**
```bash
python -m src.gong_detector.core.pipeline.detection_pipeline "https://youtube.com/watch?v=VIDEO_ID" --use_version_one
```

### **Bulk Processing:**
```bash
# Prefer local cache, fallback to download
python -m src.gong_detector.core.pipeline.bulk_processor --use_local_media --version_one

# Strict offline (error if not cached)
python -m src.gong_detector.core.pipeline.bulk_processor --local_only
```

### **Sample Collection:**
```bash
# Collect positive samples for training
python -m src.gong_detector.core.pipeline.detection_pipeline "URL" --save_positive_samples

# Interactive manual collection
python -m src.gong_detector.core.training.manual_collector

# Collect negative samples
python -m src.gong_detector.core.training.negative_collector "URL"
```

## **Project Structure**

### **Core Detection (`/core`)**
- `detector/yamnet_runner.py` - YAMNet model integration with trained classifier support
- `pipeline/detection_pipeline.py` - Single video detection with optimized batch processing
- `pipeline/bulk_processor.py` - Multi-video processing with performance optimizations
- `training/manual_collector.py` - Interactive training sample collection
- `training/negative_collector.py` - Non-gong sample collection for training
- `utils/` - Audio processing, YouTube operations, and results utilities
- `utils/local_media.py` - Local cache index and helpers
- `results/` - CSV management and output files (includes `csv_results/`)
- `data/tbpn_ytlinks/` - Input data files (includes `tbpn_youtube_links.txt`)
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
    
    # EBU R128 audio analysis
    compute_lufs_segments, compute_true_peak_segments,
    
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
    compute_peak_dbfs, download_and_trim_youtube_audio, format_time,
    compute_lufs_segments, compute_true_peak_segments
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
- **Dual-Cache System**: Raw and preprocessed audio cached in `data/local_media/`
- **EBU R128 Analysis**: Complete broadcast-standard audio analysis with LUFS loudness and True Peak (dBTP) measurements
- **Training Integration**: Seamless workflow from detection to model training
- **Performance Monitoring**: Real-time feedback on resource usage

## **Recent Fixes**

✅ **Fixed routing issues**: All pipeline scripts now work with correct module paths  
✅ **Fixed classifier loading**: Models now load from correct `core/models/` directory  
✅ **Fixed file paths**: YouTube links file moved to `core/data/` directory  
✅ **Tested all commands**: Bulk processing, individual detection, and training scripts all working  
✅ **EBU R128 compliance**: LUFS + True Peak measurements integrated across full pipeline