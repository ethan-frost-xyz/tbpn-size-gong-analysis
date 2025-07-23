# 🎵 YAMNet Gong Detection Tutorial

A step-by-step tutorial for building an audio event detection pipeline using
YAMNet to detect gong sounds in podcast episodes. This tutorial teaches you how to
build a modular system from scratch.

## 🎯 What You'll Learn

- **Audio Processing**: Loading, preprocessing, and analyzing audio files
- **Machine Learning**: Using pre-trained models (YAMNet) for audio classification
- **Modular Design**: Building reusable components and clean architecture
- **Batch Processing**: Handling multiple files efficiently
- **Audio Export**: Creating video snippets from detected events

## 📋 Prerequisites

- Python 3.9+ (Python 3.12 recommended)
- Basic understanding of Python
- Familiarity with command line tools
- macOS (for ffmpeg support)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r gong_detector/requirements.txt

# Test the system
cd gong_detector
python test_yamnet.py audio.wav
```

## 📚 Tutorial: Building the System Step by Step

### Part 1: Test YAMNet Gong Detection (`test_yamnet.py`)

**Goal**: Create a "hello world" test to prove the audio + model setup works.

**What You'll Learn**:

- Loading pre-trained YAMNet model from TensorFlow Hub
- Audio preprocessing (mono, 16kHz, normalization)
- Running inference and extracting confidence scores
- Basic error handling and validation

**Key Concepts**:

- **YAMNet**: A pre-trained neural network that can classify 521 different audio
  events
- **Gong Class Index**: YAMNet has a specific class for "gong" sounds (index 138)
- **Confidence Threshold**: Filtering detections based on model confidence

**Step-by-Step Process**:

1. Load YAMNet model from TensorFlow Hub
2. Load and preprocess audio file (convert to mono, 16kHz)
3. Run inference to get confidence scores for all classes
4. Extract gong class scores (index 138)
5. Filter detections above confidence threshold (0.5)
6. Print timestamps and confidence values
7. Save results to CSV for analysis

**Try It**:

```bash
python test_yamnet.py audio.wav
```

### Part 2: Audio Downloader and Converter (`convert_audio.py`)

**Goal**: Convert YouTube URLs or local files to the required format.

**What You'll Learn**:

- Using `yt-dlp` for YouTube downloads
- Audio format conversion with `ffmpeg`
- Command-line argument parsing
- Error handling for network/file operations

**Key Concepts**:

- **yt-dlp**: Modern YouTube downloader (replacement for youtube-dl)
- **ffmpeg**: Powerful audio/video processing tool
- **Audio Requirements**: YAMNet needs mono, 16kHz WAV files

**Step-by-Step Process**:

1. Accept YouTube URL or local file path as input
2. Download audio using yt-dlp (if YouTube URL)
3. Convert to WAV format using ffmpeg
4. Force mono channel and 16kHz sample rate
5. Save as `audio.wav` (or configurable path)
6. Clean up temporary files

**Try It**:

```bash
# Convert YouTube video
python convert_audio.py "https://youtube.com/watch?v=example"

# Convert local file
python convert_audio.py input.mp3 output.wav
```

### Part 3: Batch Gong Detection Tool (`batch_detect.py`)

**Goal**: Process multiple audio files and save detection results.

**What You'll Learn**:

- Batch file processing
- Error handling for malformed audio
- Multiple output formats (CSV/JSON)
- Progress tracking and summaries

**Key Concepts**:

- **Batch Processing**: Efficiently handle multiple files
- **Graceful Degradation**: Continue processing even if some files fail
- **Output Formats**: CSV for spreadsheet analysis, JSON for programmatic use

**Step-by-Step Process**:

1. Find all `.wav` files in a directory
2. Load YAMNet model once (efficient for multiple files)
3. Process each file:
   - Load and preprocess audio
   - Run YAMNet inference
   - Extract gong detections
   - Save results to CSV/JSON
4. Handle errors gracefully (skip failed files)
5. Print summary statistics

**Try It**:

```bash
# Process all WAV files in current directory
python batch_detect.py .

# Use different threshold and output format
python batch_detect.py . --threshold 0.7 --format json
```

### Part 4: Decibel Estimation Utility (`audio_utils.py`)

**Goal**: Analyze audio levels and extract audio slices around events.

**What You'll Learn**:

- Audio level calculations (peak, RMS, dBFS)
- Audio slicing and manipulation
- Working with numpy arrays
- Audio analysis concepts

**Key Concepts**:

- **dBFS**: Decibels relative to full scale (digital audio measurement)
- **Peak vs RMS**: Different ways to measure audio levels
- **Audio Slicing**: Extracting portions of audio around specific timestamps

**Key Functions**:

- `compute_peak_dbfs()`: Calculate peak level in dBFS
- `compute_rms_dbfs()`: Calculate RMS level in dBFS
- `extract_audio_slice()`: Get audio around a timestamp
- `analyze_audio_slice_levels()`: Combined analysis function

**Try It**:

```python
from audio_utils import compute_peak_dbfs, extract_audio_slice
import numpy as np

# Create test audio
audio = np.random.random(16000) * 0.5
peak_db = compute_peak_dbfs(audio)
print(f"Peak level: {peak_db:.1f} dBFS")
```

### Part 5: MP4 Export with Gong Snippets (`export_snippets.py`)

**Goal**: Create video snippets from detected gong events.

**What You'll Learn**:

- Using ffmpeg for video creation
- Loading detection data from CSV/JSON
- Audio-to-video conversion
- File naming and organization

**Key Concepts**:

- **Audio-Only Video**: MP4 files with audio track but no visual content
- **Context Windows**: Extract audio before and after the detected event
- **ffmpeg Integration**: Using subprocess to call ffmpeg command-line tool

**Step-by-Step Process**:

1. Load detection data from CSV/JSON file
2. For each detected gong:
   - Extract 20 seconds before and 5 seconds after
   - Create MP4 file with audio-only video track
   - Use descriptive filename (e.g., `ep12_gong_053s.mp4`)
3. Optional: Calculate and display dB levels for each snippet

**Try It**:

```bash
# Export snippets from detection results
python export_snippets.py audio.wav detections.csv --output-dir snippets
```

### Part 6: Refactor into Importable Modules (`yamnet_runner.py`)

**Goal**: Create clean, reusable modules with proper structure.

**What You'll Learn**:

- Modular design principles
- Import/export patterns
- Clean separation of concerns
- Professional project structure

**Key Concepts**:

- **Modularity**: Each file has a single, clear responsibility
- **Reusability**: Functions can be imported and used in other scripts
- **Main Pattern**: `if __name__ == '__main__':` for script execution

**Architecture**:

- `yamnet_runner.py`: Core YAMNet functionality
- `audio_utils.py`: Audio processing utilities
- `convert_audio.py`: Audio conversion tools
- `export_snippets.py`: Video export functionality
- Script files: Command-line interfaces

## 🔧 Usage Examples

### Basic Detection

```bash
# Single file detection
python detect.py audio.wav --threshold 0.5

# Save results to CSV
python detect.py audio.wav --output detections.csv
```

### Batch Processing

```bash
# Process all WAV files in a directory
python batch_detect.py /path/to/audio/files/

# Use different settings
python batch_detect.py . --threshold 0.7 --format json --output-dir results/
```

### Audio Conversion

```bash
# Convert YouTube video
python convert_audio.py "https://youtube.com/watch?v=example"

# Convert local file
python convert_audio.py input.mp3
```

### Export Snippets

```bash
# Create video snippets from detections
python export_snippets.py source.wav detections.csv --episode-name "episode_01"
```

## 📁 Project Structure

```text
gong_detector/
├── detect.py            # Simple CLI detection script
├── convert_audio.py     # YouTube/audio conversion
├── batch_detect.py      # Batch processing tool
├── export_snippets.py   # MP4 export functionality
├── audio_utils.py       # Audio processing utilities
├── yamnet_runner.py     # Core YAMNet functionality
├── test_yamnet.py       # "Hello world" test
├── requirements.txt     # Dependencies
└── audio.wav           # Sample audio file
```

## 🧪 Testing Your Understanding

After completing each part, test your knowledge:

1. **Part 1**: Can you modify the confidence threshold and see how it affects
   detections?
2. **Part 2**: Try converting different audio formats (MP3, M4A, etc.)
3. **Part 3**: Process a folder with mixed audio files and handle errors
4. **Part 4**: Calculate audio levels for different types of sounds
5. **Part 5**: Create snippets with different time windows
6. **Part 6**: Import and use the modules in your own scripts

## 🐛 Troubleshooting

### Common Issues

**TensorFlow Import Error**:

```bash
# Install TensorFlow with proper version for Python 3.12
pip install tensorflow>=2.15.0
```

**ffmpeg Not Found**:

```bash
# Install ffmpeg on macOS
brew install ffmpeg
```

**SSL Certificate Issues (macOS)**:

```bash
# Install certificates for Python 3.12
/Applications/Python\ 3.12/Install\ Certificates.command
```

**Audio File Issues**:

- Ensure files are in supported formats (WAV, MP3, M4A)
- Check file permissions and paths
- Verify audio files aren't corrupted

### Debug Mode

Add debug prints to understand what's happening:

```python
print(f"Audio shape: {waveform.shape}")
print(f"Sample rate: {sample_rate}Hz")
print(f"Model loaded: {self.model is not None}")
```

## 🎓 Next Steps

Once you've mastered this tutorial:

1. **Extend Detection**: Add support for other audio events
2. **Improve Accuracy**: Fine-tune confidence thresholds
3. **Add Visualization**: Create plots of detection results
4. **Web Interface**: Build a simple web app for upload/analysis
5. **Real-time Processing**: Process audio streams in real-time

## 📖 Additional Resources

- [YAMNet Paper](https://arxiv.org/abs/1609.09430)
- [TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
- [ffmpeg Documentation](https://ffmpeg.org/documentation.html)
- [Audio Processing with Python](https://librosa.org/)

---

## Happy Coding! 🎵

This tutorial teaches you to build a complete audio analysis pipeline from scratch.
Each part builds on the previous, creating a solid foundation for audio machine
learning projects.
