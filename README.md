# TBPN Gong Detection System

YAMNet-based audio event detection system for identifying gong sounds in YouTube videos with **EBU R128 compliance** (LUFS + True Peak measurements).

## Quick Start

```bash
# Setup environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Interactive menu (recommended)
cd src/gong_detector && python run_gong_detector.py

# Or direct command
python -m src.gong_detector.core.pipeline.detection_pipeline "YOUTUBE_URL"
```

## Key Features

- **Gong Detection**: YAMNet-based detection with trained classifier (99.3% accuracy)
- **EBU R128 Analysis**: Complete broadcast-standard audio analysis
  - **LUFS loudness** (ITU-R BS.1770-4)
  - **True Peak (dBTP)** measurements
  - **Batch weighting** across entire dataset
- **⚡ Performance**: Batch processing, multi-threading, dual-cache system
- **🔄 Interactive Menu**: User-friendly interface with all functionality
- **📁 Local Caching**: Offline processing with raw + preprocessed audio cache

## Project Structure

```
src/gong_detector/
├── core/
│   ├── detector/yamnet_runner.py      # YAMNet model + trained classifier
│   ├── pipeline/
│   │   ├── detection_pipeline.py      # Single video detection
│   │   └── bulk_processor.py          # Multi-video processing
│   ├── utils/
│   │   ├── youtube_utils.py           # EBU R128 analysis (LUFS + True Peak)
│   │   ├── audio_utils.py             # Audio processing utilities
│   │   └── local_media.py             # Dual-cache management
│   └── data/csv_manager.py            # CSV output with 39 fields
├── training/                          # Training data collection
└── run_gong_detector.py               # Interactive menu

data/
├── local_media/                       # Dual-cache system
│   ├── raw/                          # Original audio (LUFS/True Peak)
│   ├── preprocessed/                  # 16kHz mono WAVs
│   └── index.json                     # Cache metadata
├── csv_results/                       # Analysis outputs
└── tbpn_ytlinks/tbpn_youtube_links.txt # Input URLs
```

## Usage Examples

### Interactive Menu (Recommended)
```bash
cd src/gong_detector
python run_gong_detector.py
# Navigate with arrow keys, select options
```

### Command Line
```bash
# Single video with EBU R128 analysis
python -m src.gong_detector.core.pipeline.detection_pipeline "URL" --csv

# Bulk processing (52 videos)
python -m src.gong_detector.core.pipeline.bulk_processor --csv --use_local_media

# Test mode (first 5 videos)
python -m src.gong_detector.core.pipeline.bulk_processor --csv --test-run 5
```

### Programmatic API
```python
from gong_detector.core.detector import YAMNetGongDetector
from gong_detector.core.utils.youtube_utils import compute_lufs_segments, compute_true_peak_segments

# Initialize detector
detector = YAMNetGongDetector(use_trained_classifier=True)
detector.load_model()

# Process video
result = detector.process_video("YOUTUBE_URL")

# EBU R128 analysis
lufs_results = compute_lufs_segments("VIDEO_ID", timestamps, "integrated")
dbtp_results = compute_true_peak_segments("VIDEO_ID", timestamps, "integrated")
```

## CSV Output

The system generates comprehensive CSV files with **39 fields** including:

**Detection Data:**
- Video metadata (title, URL, duration, upload date)
- Detection timestamps and confidence scores
- YouTube timestamped links

**Audio Analysis:**
- **LUFS measurements**: `detection_integrated_lufs`, `detection_shortterm_lufs`, `detection_momentary_lufs`
- **True Peak measurements**: `detection_integrated_dbtp`, `detection_shortterm_dbtp`, `detection_momentary_dbtp`
- **Traditional metrics**: Peak dBFS, RMS dBFS, crest factor, clipping detection

**Batch Processing:**
- All measurements are **batch-weighted** across the entire dataset
- **EBU R128 reference levels**: -23.0 LUFS, -1.0 dBTP
- Consistent analysis across all videos

## Development

```bash
# Linting
ruff check src/
ruff check src/ --fix

# Testing
pytest tests/

# Format code
black src/
```

## Dependencies

**Core ML:**
- `tensorflow==2.19.0` - YAMNet model (pinned for stability)
- `librosa==0.11.0` - Audio processing (pinned to avoid v1.0 breaking changes)
- `pyloudnorm==0.1.1` - EBU R128 analysis

**Audio/Video:**
- `yt-dlp>=2023.7.6` - YouTube downloads
- `ffmpeg` - Audio conversion (system dependency)

**Utilities:**
- `numpy>=1.24.0,<2.0.0` - Data processing
- `pandas>=2.0.0,<3.0.0` - Data handling
- `scikit-learn==1.6.1` - Trained classifier
- `setuptools<81` - Prevents pkg_resources deprecation warnings

### Known Issues & Warnings

**Non-breaking warnings you may see:**
- `pkg_resources is deprecated` - From TensorFlow Hub, will be fixed in future releases
- `PySoundFile failed. Trying audioread instead` - Normal fallback behavior for some audio formats
- `__audioread_load Deprecated` - Librosa deprecation, non-breaking until v1.0
- `Short-term/Momentary LUFS approximated` - Normal when precise measurements aren't available

**Version pinning rationale:**
- Dependencies are pinned to tested, working versions to prevent compatibility issues
- Use `pip install -r requirements.txt` for exact reproducible environment

## Environment Setup

**Required:**
- Python 3.12+
- ffmpeg (for audio conversion)
- 16GB+ RAM (for batch processing)

**Optional:**
- GPU acceleration (TensorFlow)
- Local media cache (for offline processing)

## Recent Updates

✅ **EBU R128 Compliance**: Complete LUFS + True Peak integration  
✅ **Batch Weighting**: Measurements normalized across entire dataset  
✅ **Interactive Menu**: User-friendly interface with all features  
✅ **Dual-Cache System**: Raw + preprocessed audio for offline processing  
✅ **Comprehensive CSV**: 39 fields with broadcast-standard analysis  
✅ **Performance Optimized**: Batch processing, multi-threading, memory efficient
