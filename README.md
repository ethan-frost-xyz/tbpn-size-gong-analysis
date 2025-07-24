# YAMNet Gong Detection

A comprehensive audio event detection system using YAMNet to identify gong sounds in podcast episodes. This project provides both a learning tutorial and production-ready tools for audio analysis.

## Overview

This system demonstrates how to build a modular audio processing pipeline from scratch, combining machine learning with practical audio engineering. The project includes complete implementations of audio preprocessing, YAMNet model integration, batch processing capabilities, and export functionality.

## Features

- **Audio Processing**: Loading, preprocessing, and analyzing audio files with proper normalization
- **Machine Learning**: Pre-trained YAMNet model integration for audio event classification  
- **Modular Design**: Clean, reusable components with clear separation of concerns
- **Batch Processing**: Efficient handling of multiple audio files with progress tracking
- **YouTube Integration**: Direct audio extraction and processing from YouTube URLs
- **Export Capabilities**: Generate audio snippets and analysis reports in multiple formats

## Prerequisites

- Python 3.9 or higher (Python 3.12 recommended)
- Basic understanding of Python programming
- Familiarity with command line tools
- macOS or Linux (for ffmpeg support)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Basic gong detection on an audio file
python -m gong_detector.core.detect_from_youtube --help

# Convert and analyze YouTube audio
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID"
```

## Installation

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tbpn-size-gong-analysis.git
cd tbpn-size-gong-analysis

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install ffmpeg

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run tests to verify installation
pytest tests/ -v

# Run linting
ruff check gong_detector/
```

## Project Structure

```text
tbpn-size-gong-analysis/
├── gong_detector/           # Main package
│   ├── core/               # Core functionality
│   │   ├── yamnet_runner.py    # YAMNet model integration
│   │   ├── audio_utils.py      # Audio processing utilities
│   │   ├── convert_audio.py    # Audio format conversion
│   │   └── detect_from_youtube.py  # YouTube detection script
│   ├── training/           # Training data and scripts
│   └── requirements.txt    # Package dependencies
├── tests/                  # Test suite
├── legacy/                 # Archived code and examples
├── pyproject.toml         # Project configuration
└── README.md              # This file
```

## Core Components

### YAMNet Integration (yamnet_runner.py)

Provides the main `YAMNetGongDetector` class for audio event detection:

- Loading pre-trained YAMNet model from TensorFlow Hub
- Audio preprocessing (mono conversion, 16kHz resampling, normalization)
- Inference execution and confidence score extraction
- Detection post-processing with configurable thresholds

```python
from gong_detector.core import YAMNetGongDetector

detector = YAMNetGongDetector()
detector.load_model()
waveform, sample_rate = detector.load_and_preprocess_audio("audio.wav")
scores, embeddings, spectrogram = detector.run_inference(waveform)
detections = detector.detect_gongs(scores, confidence_threshold=0.5)
```

### Audio Utilities (audio_utils.py)

Essential audio processing functions for level analysis and manipulation:

- **dBFS Calculations**: Peak and RMS level measurements
- **Audio Slicing**: Extract segments around specific timestamps  
- **Normalization**: Amplitude adjustment and silence detection
- **Statistics**: Comprehensive audio file analysis

Key functions include `compute_peak_dbfs()`, `extract_audio_slice()`, and `analyze_audio_slice_levels()`.

### Audio Conversion (convert_audio.py)

Handles format conversion and YouTube audio extraction:

- **YouTube Downloads**: Using yt-dlp for reliable video processing
- **Format Conversion**: FFmpeg integration for audio format handling
- **Validation**: File format verification and metadata extraction
- **Error Handling**: Robust processing with detailed error reporting

### YouTube Detection (detect_from_youtube.py)

Command-line interface for end-to-end YouTube audio analysis:

- Direct URL processing with optional time segment selection
- Temporary file management with automatic cleanup
- CSV export functionality for analysis results
- Progress reporting and summary statistics

## Usage Examples

### Basic Detection

```bash
# Analyze entire video
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID"

# Analyze specific time segment
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID" --start_time 300 --duration 60

# Adjust detection sensitivity
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID" --threshold 0.3

# Save results to CSV
python -m gong_detector.core.detect_from_youtube "https://youtube.com/watch?v=VIDEO_ID" --save_csv results.csv
```

### Programmatic Usage

```python
from gong_detector.core import YAMNetGongDetector, convert_youtube_audio

# Convert audio file
audio_path = convert_youtube_audio("https://youtube.com/watch?v=VIDEO_ID", "audio.wav")

# Initialize detector
detector = YAMNetGongDetector()
detector.load_model()

# Process audio
waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)
scores, _, _ = detector.run_inference(waveform)

# Extract detections
detections = detector.detect_gongs(scores, confidence_threshold=0.4)
detector.print_detections(detections)

# Convert to DataFrame for analysis
df = detector.detections_to_dataframe(detections)
df.to_csv("detections.csv", index=False)
```

## API Reference

### YAMNetGongDetector Class

**Methods:**

- `load_model()`: Initialize YAMNet from TensorFlow Hub
- `load_and_preprocess_audio(path)`: Load and prepare audio for analysis
- `run_inference(waveform)`: Execute model inference
- `detect_gongs(scores, threshold)`: Extract gong detections from scores
- `print_detections(detections)`: Display results in formatted table
- `detections_to_dataframe(detections)`: Convert results to pandas DataFrame

**Key Concepts:**

- **YAMNet**: Pre-trained neural network for 521 audio event classes
- **Gong Class Index**: Class 172 in the YAMNet taxonomy (metallic percussion instruments)
- **Confidence Threshold**: Minimum score for positive detection (typically 0.3-0.7)
- **Hop Length**: Time resolution between predictions (approximately 0.48 seconds)

### Audio Utility Functions

**Level Analysis:**

- `compute_peak_dbfs(waveform)`: Peak amplitude in dBFS
- `compute_rms_dbfs(waveform)`: RMS amplitude in dBFS  
- `compute_audio_levels(waveform)`: Combined peak and RMS analysis

**Audio Manipulation:**

- `extract_audio_slice(waveform, timestamp, before, after)`: Extract time segment
- `normalize_waveform(waveform, target_dbfs)`: Amplitude normalization
- `is_silent(waveform, threshold)`: Silence detection

**Conversion Functions:**

- `convert_youtube_audio(url, output_path)`: Download and convert YouTube audio
- `validate_audio_file(path)`: Verify file format compatibility
- `get_audio_info(path)`: Extract metadata using ffprobe

## Testing

The project includes comprehensive test coverage using pytest:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage reporting
pytest tests/ --cov=gong_detector --cov-report=html

# Run specific test modules
pytest tests/test_yamnet_runner.py -v
pytest tests/test_audio_utils.py -v
```

## Troubleshooting

### Common Issues

**TensorFlow Import Errors:**

```bash
# Install compatible TensorFlow version
pip install tensorflow>=2.15.0
pip install tensorflow-hub>=0.14.0
```

**FFmpeg Not Found:**

```bash
# macOS installation
brew install ffmpeg

# Ubuntu/Debian installation  
sudo apt-get install ffmpeg

# Verify installation
ffmpeg -version
```

**SSL Certificate Issues (macOS):**

```bash
# Install certificates for Python
/Applications/Python\ 3.12/Install\ Certificates.command
```

**Audio File Processing:**

- Ensure files are in supported formats (WAV, MP3, M4A, FLAC)
- Check file permissions and verify paths are correct
- Validate audio files are not corrupted using ffprobe

### Debug Mode

Enable detailed logging by adding debug prints:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or add manual debug output
print(f"Audio shape: {waveform.shape}")  
print(f"Sample rate: {sample_rate}Hz")
print(f"Detection count: {len(detections)}")
```

## Development

### Code Style

This project follows strict code quality standards:

- **Type Annotations**: All functions include proper type hints
- **Docstrings**: Complete PEP 257 compliant documentation
- **Linting**: Ruff for code style enforcement
- **Testing**: Comprehensive pytest coverage

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass and linting is clean
5. Submit a pull request

### Architecture Decisions

- **Simplicity First**: Code prioritizes readability over complex abstractions
- **Modular Design**: Clear separation between audio processing, ML inference, and I/O
- **Error Handling**: Graceful degradation with informative error messages
- **Performance**: Efficient batch processing for multiple files

## Technical Background

### YAMNet Model Details

YAMNet is a pre-trained audio event classifier developed by Google Research. Key characteristics:

- **Architecture**: MobileNet-based convolutional neural network
- **Training Data**: AudioSet dataset with 521 sound classes
- **Input Format**: 16kHz mono audio with variable length support
- **Output**: Per-frame predictions with 0.48 second temporal resolution
- **Gong Classification**: Class 172 represents metallic percussion instruments

### Audio Processing Pipeline

1. **Loading**: TensorFlow audio decoder for WAV files
2. **Resampling**: Signal processing to 16kHz using TensorFlow operations
3. **Normalization**: Amplitude scaling to [-1, 1] range
4. **Inference**: Model execution on preprocessed waveform
5. **Post-processing**: Confidence thresholding and temporal filtering

## License

This project is available under the MIT License. See LICENSE file for details.

## References

- [YAMNet Paper](https://arxiv.org/abs/1609.09430) - Original model architecture and training details
- [TensorFlow Hub YAMNet](https://tfhub.dev/google/yamnet/1) - Pre-trained model repository
- [AudioSet](https://research.google.com/audioset/) - Training dataset information
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html) - Audio processing reference
