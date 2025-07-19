# 🛠 Development Guide: Building the Gong Detection System

This guide walks you through building each part of the gong detection system from scratch. Follow along step by step to understand how everything works.

## 🎯 Development Philosophy

- **Start Simple**: Begin with basic functionality, then add complexity
- **Test Early**: Verify each component works before moving to the next
- **Modular Design**: Each file has a single, clear responsibility
- **Error Handling**: Graceful failure modes for robust operation

---

## Part 1: Test YAMNet Gong Detection

**File**: `test_yamnet.py`  
**Goal**: Create a "hello world" test for YAMNet gong detection

### Step 1: Understand YAMNet

YAMNet is a pre-trained neural network that can classify 521 different audio events. It's designed for mobile devices and runs efficiently on CPU.

**Key Facts**:
- Input: 16kHz mono audio
- Output: Confidence scores for 521 classes
- Gong class index: 138
- Each prediction covers ~0.96 seconds of audio

### Step 2: Create the Basic Structure

```python
#!/usr/bin/env python3
"""Test YAMNet gong detection functionality."""

import sys
from typing import List, Tuple

from yamnet_runner import YAMNetGongDetector


def test_yamnet_gong_detection(audio_path: str = "audio.wav") -> None:
    """Test the complete YAMNet gong detection pipeline."""
    print("🎵 Testing YAMNet Gong Detection")
    print("=" * 50)
    
    # Your code will go here...


if __name__ == "__main__":
    audio_file = "audio.wav"
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    
    test_yamnet_gong_detection(audio_file)
```

### Step 3: Add Model Loading

```python
def test_yamnet_gong_detection(audio_path: str = "audio.wav") -> None:
    """Test the complete YAMNet gong detection pipeline."""
    print("🎵 Testing YAMNet Gong Detection")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = YAMNetGongDetector()
        
        # Load model
        detector.load_model()
        
        # Verify gong class exists
        if detector.class_names is None:
            raise RuntimeError("Class names not loaded")
            
        assert detector.gong_class_index < len(detector.class_names), \
            f"Gong class index {detector.gong_class_index} out of range"
        assert "gong" in detector.class_names[detector.gong_class_index].lower(), \
            f"Class at index {detector.gong_class_index} is not gong: {detector.class_names[detector.gong_class_index]}"
        
        print(f"✅ Gong class verification passed: '{detector.class_names[detector.gong_class_index]}'")
        
        # Continue with audio processing...
```

### Step 4: Add Audio Processing

```python
        # Load and preprocess audio (should be mono, 16kHz)
        waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)
        
        # Verify audio properties
        print(f"Audio shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate}Hz")
        print(f"Duration: {len(waveform) / sample_rate:.2f} seconds")
        
        # Run inference
        scores, embeddings, spectrogram = detector.run_inference(waveform)
```

### Step 5: Add Detection Logic

```python
        # Detect gongs with confidence > 0.5
        detections = detector.detect_gongs(scores, confidence_threshold=0.5)
        
        # Print results
        if detections:
            print("\n" + "="*50)
            print("GONG DETECTIONS (confidence > 0.5)")
            print("="*50)
            print(f"{'Timestamp (s)':<15} {'Confidence':<12}")
            print("-" * 27)
            
            for timestamp, confidence in detections:
                print(f"{timestamp:<15.2f} {confidence:<12.4f}")
            
            print("="*50)
            
            # Convert to DataFrame and display
            df = detector.detections_to_dataframe(detections)
            print(f"\n📊 Detection DataFrame shape: {df.shape}")
            if not df.empty:
                print("DataFrame contents:")
                print(df.to_string(index=False))
                
                # Save to CSV
                output_path = "test_gong_detections.csv"
                df.to_csv(output_path, index=False)
                print(f"💾 Test detections saved to: {output_path}")
        else:
            print("No gong detections found with confidence > 0.5")
```

### Step 6: Add Error Handling

```python
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
```

### Testing Part 1

```bash
# Test with sample audio
python test_yamnet.py audio.wav

# Test with different file
python test_yamnet.py your_audio.wav
```

**What You Learned**:
- How YAMNet works and what it outputs
- Audio preprocessing requirements
- Confidence threshold filtering
- Basic error handling and validation

---

## Part 2: Audio Downloader and Converter

**File**: `convert_audio.py`  
**Goal**: Convert YouTube URLs or local files to WAV format

### Step 1: Understand the Requirements

YAMNet needs:
- Mono audio (single channel)
- 16kHz sample rate
- WAV format
- Normalized amplitude

### Step 2: Create Basic Structure

```python
#!/usr/bin/env python3
"""Simple audio converter for YouTube URLs and local files."""

import os
import subprocess
from typing import Optional

import yt_dlp  # type: ignore


def convert_youtube_audio(url_or_path: str, output_wav_path: str = "audio.wav") -> str:
    """Convert YouTube URL or local file to WAV format."""
    # Your code will go here...


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python convert_audio.py <youtube_url_or_file> [output.wav]")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    output_arg = sys.argv[2] if len(sys.argv) > 2 else "audio.wav"
    
    try:
        convert_youtube_audio(input_arg, output_arg)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
```

### Step 3: Add Input Handling

```python
def convert_youtube_audio(url_or_path: str, output_wav_path: str = "audio.wav") -> str:
    """Convert YouTube URL or local file to WAV format."""
    # Handle local file
    if os.path.exists(url_or_path):
        print(f"Converting local file: {url_or_path}")
        input_path = url_or_path
    else:
        # Download from YouTube
        print(f"Downloading from YouTube: {url_or_path}")
        input_path = _download_audio(url_or_path)
    
    # Convert to WAV
    _convert_to_wav(input_path, output_wav_path)
    
    # Clean up downloaded file
    if input_path != url_or_path and os.path.exists(input_path):
        os.remove(input_path)
    
    print(f"Conversion complete! Saved to: {output_wav_path}")
    return output_wav_path
```

### Step 4: Add YouTube Download Function

```python
def _download_audio(url: str) -> str:
    """Download audio from YouTube URL."""
    ydl_opts = {
        "format": "bestaudio/best",
        "extractaudio": True,
        "audioformat": "mp3",
        "outtmpl": "temp_%(title)s.%(ext)s",
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".mp4", ".mp3")
        ydl.download([url])
        return filename
```

### Step 5: Add FFmpeg Conversion

```python
def _convert_to_wav(input_path: str, output_path: str) -> None:
    """Convert audio to WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-i", input_path,
        "-ac", "1",           # mono
        "-ar", "16000",       # 16kHz
        "-y",                 # overwrite
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)
```

### Testing Part 2

```bash
# Convert YouTube video
python convert_audio.py "https://youtube.com/watch?v=example"

# Convert local file
python convert_audio.py input.mp3 output.wav
```

**What You Learned**:
- Using yt-dlp for YouTube downloads
- FFmpeg command-line usage
- Audio format conversion
- Error handling for network operations

---

## Part 3: Batch Gong Detection Tool

**File**: `batch_detect.py`  
**Goal**: Process multiple audio files efficiently

### Step 1: Understand Batch Processing

Key concepts:
- Process multiple files with one model load
- Handle errors gracefully
- Provide progress feedback
- Support multiple output formats

### Step 2: Create Basic Structure

```python
#!/usr/bin/env python3
"""Batch gong detection tool."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from yamnet_runner import YAMNetGongDetector


def find_wav_files(directory: str) -> List[Path]:
    """Find all .wav files in a directory."""
    # Your code will go here...


def process_single_file(
    detector: YAMNetGongDetector, 
    audio_path: Path, 
    confidence_threshold: float = 0.5
) -> Tuple[List[Tuple[float, float]], str]:
    """Process a single audio file for gong detection."""
    # Your code will go here...


def main() -> None:
    """Main batch detection execution."""
    # Your code will go here...


if __name__ == "__main__":
    main()
```

### Step 3: Add File Discovery

```python
def find_wav_files(directory: str) -> List[Path]:
    """Find all .wav files in a directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    wav_files = list(dir_path.glob("*.wav"))
    wav_files.sort()  # Process in consistent order
    
    print(f"Found {len(wav_files)} .wav files in {directory}")
    return wav_files
```

### Step 4: Add Single File Processing

```python
def process_single_file(
    detector: YAMNetGongDetector, 
    audio_path: Path, 
    confidence_threshold: float = 0.5
) -> Tuple[List[Tuple[float, float]], str]:
    """Process a single audio file for gong detection."""
    try:
        print(f"Processing: {audio_path.name}")
        
        # Load and preprocess audio
        waveform, sample_rate = detector.load_and_preprocess_audio(str(audio_path))
        
        # Run inference
        scores, _, _ = detector.run_inference(waveform)
        
        # Detect gongs
        detections = detector.detect_gongs(scores, confidence_threshold=confidence_threshold)
        
        print(f"  → Found {len(detections)} detections")
        return detections, ""
        
    except Exception as e:
        error_msg = f"Error processing {audio_path.name}: {e}"
        print(f"  ❌ {error_msg}")
        return [], error_msg
```

### Step 5: Add Output Functions

```python
def save_detections_csv(detections: List[Tuple[float, float]], output_path: Path) -> None:
    """Save detections to CSV format."""
    with open(output_path, 'w') as f:
        f.write("timestamp_seconds,confidence\n")
        for timestamp, confidence in detections:
            f.write(f"{timestamp:.3f},{confidence:.6f}\n")


def save_detections_json(detections: List[Tuple[float, float]], output_path: Path) -> None:
    """Save detections to JSON format."""
    detection_list = [
        {"timestamp_seconds": timestamp, "confidence": confidence}
        for timestamp, confidence in detections
    ]
    
    with open(output_path, 'w') as f:
        json.dump(detection_list, f, indent=2)
```

### Step 6: Add Main Processing Loop

```python
def main() -> None:
    """Main batch detection execution."""
    parser = argparse.ArgumentParser(description="Batch gong detection on WAV files")
    parser.add_argument("directory", help="Directory containing .wav files")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                       help="Confidence threshold for detection (default: 0.5)")
    parser.add_argument("--format", "-f", choices=["csv", "json"], default="csv",
                       help="Output format (default: csv)")
    parser.add_argument("--output-dir", "-o", help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.directory)
    output_dir.mkdir(exist_ok=True)
    
    print("🎵 Batch Gong Detection")
    print("=" * 50)
    print(f"Input directory: {args.directory}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Output format: {args.format}")
    print()
    
    try:
        # Find WAV files
        wav_files = find_wav_files(args.directory)
        if not wav_files:
            print("No .wav files found!")
            return
        
        # Initialize detector (once for all files)
        print("Loading YAMNet model...")
        detector = YAMNetGongDetector()
        detector.load_model()
        print()
        
        # Process each file
        results: Dict[str, int] = {}
        errors: List[str] = []
        
        for audio_path in wav_files:
            detections, error_msg = process_single_file(
                detector, audio_path, args.threshold
            )
            
            if error_msg:
                errors.append(error_msg)
                continue
            
            # Save detections
            base_name = audio_path.stem
            if args.format == "csv":
                output_path = output_dir / f"{base_name}_detections.csv"
                save_detections_csv(detections, output_path)
            else:
                output_path = output_dir / f"{base_name}_detections.json"
                save_detections_json(detections, output_path)
            
            results[audio_path.name] = len(detections)
            print(f"  💾 Saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 50)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 50)
        
        if results:
            print("Detection counts per file:")
            for filename, count in results.items():
                print(f"  {filename:<30} {count:>3} detections")
            
            total_detections = sum(results.values())
            print(f"\nTotal files processed: {len(results)}")
            print(f"Total detections: {total_detections}")
        
        if errors:
            print(f"\nErrors encountered: {len(errors)}")
            for error in errors:
                print(f"  ❌ {error}")
        
        print("✅ Batch processing complete!")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        sys.exit(1)
```

### Testing Part 3

```bash
# Process all WAV files in current directory
python batch_detect.py .

# Use different settings
python batch_detect.py . --threshold 0.7 --format json --output-dir results/
```

**What You Learned**:
- Batch file processing patterns
- Error handling and graceful degradation
- Multiple output format support
- Progress tracking and summaries

---

## Part 4: Decibel Estimation Utility

**File**: `audio_utils.py`  
**Goal**: Analyze audio levels and extract audio slices

### Step 1: Understand Audio Analysis

Key concepts:
- **dBFS**: Decibels relative to full scale (digital audio measurement)
- **Peak vs RMS**: Different ways to measure audio levels
- **Audio Slicing**: Extracting portions of audio around timestamps

### Step 2: Create Basic Structure

```python
"""Audio utilities for decibel estimation and waveform manipulation."""

from typing import Tuple
import numpy as np


# Constants
SILENCE_FLOOR_DBFS = -80.0  # Silence floor threshold in dBFS
MIN_AMPLITUDE = 1e-8  # Minimum amplitude to avoid log(0) errors


def compute_peak_dbfs(waveform: np.ndarray) -> float:
    """Compute peak dBFS level from a numpy waveform."""
    # Your code will go here...


def compute_rms_dbfs(waveform: np.ndarray) -> float:
    """Compute RMS dBFS level from a numpy waveform."""
    # Your code will go here...


def extract_audio_slice(
    waveform: np.ndarray,
    timestamp: float,
    duration_before: float = 20.0,
    duration_after: float = 5.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """Extract an audio slice around a specific timestamp."""
    # Your code will go here...
```

### Step 3: Add Peak Level Calculation

```python
def compute_peak_dbfs(waveform: np.ndarray) -> float:
    """Compute peak dBFS level from a numpy waveform."""
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS
        
    # Find peak amplitude
    peak_amplitude = np.max(np.abs(waveform))
    
    # Handle silence case
    if peak_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS
        
    # Convert to dBFS: 20 * log10(amplitude)
    peak_dbfs = 20.0 * np.log10(peak_amplitude)
    
    return float(peak_dbfs)
```

### Step 4: Add RMS Level Calculation

```python
def compute_rms_dbfs(waveform: np.ndarray) -> float:
    """Compute RMS dBFS level from a numpy waveform."""
    if len(waveform) == 0:
        return SILENCE_FLOOR_DBFS
        
    # Compute RMS amplitude
    rms_amplitude = np.sqrt(np.mean(waveform ** 2))
    
    # Handle silence case
    if rms_amplitude < MIN_AMPLITUDE:
        return SILENCE_FLOOR_DBFS
        
    # Convert to dBFS: 20 * log10(amplitude)
    rms_dbfs = 20.0 * np.log10(rms_amplitude)
    
    return float(rms_dbfs)
```

### Step 5: Add Audio Slicing

```python
def extract_audio_slice(
    waveform: np.ndarray,
    timestamp: float,
    duration_before: float = 20.0,
    duration_after: float = 5.0,
    sample_rate: int = 16000
) -> np.ndarray:
    """Extract an audio slice around a specific timestamp."""
    if len(waveform) == 0:
        return np.array([])
        
    # Convert times to sample indices
    start_time = timestamp - duration_before
    end_time = timestamp + duration_after
    
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)
    
    # Handle boundaries with zero-padding
    slice_length = end_sample - start_sample
    audio_slice = np.zeros(slice_length, dtype=waveform.dtype)
    
    # Calculate valid range within original waveform
    waveform_start = max(0, start_sample)
    waveform_end = min(len(waveform), end_sample)
    
    # Calculate corresponding range in output slice
    slice_start = max(0, -start_sample)
    slice_end = slice_start + (waveform_end - waveform_start)
    
    # Copy valid audio data
    if waveform_start < waveform_end and slice_start < slice_end:
        audio_slice[slice_start:slice_end] = waveform[waveform_start:waveform_end]
    
    return audio_slice
```

### Step 6: Add Helper Functions

```python
def compute_audio_levels(waveform: np.ndarray) -> Tuple[float, float]:
    """Compute both peak and RMS dBFS levels from a waveform."""
    peak_dbfs = compute_peak_dbfs(waveform)
    rms_dbfs = compute_rms_dbfs(waveform)
    
    return peak_dbfs, rms_dbfs


def analyze_audio_slice_levels(
    waveform: np.ndarray,
    timestamp: float,
    context_seconds: float = 20.0,
    sample_rate: int = 16000
) -> Tuple[float, float]:
    """Extract audio slice and compute its dBFS levels."""
    audio_slice = extract_audio_slice(
        waveform, timestamp, context_seconds/2, context_seconds/2, sample_rate
    )
    
    return compute_audio_levels(audio_slice)
```

### Testing Part 4

```python
from audio_utils import compute_peak_dbfs, extract_audio_slice
import numpy as np

# Create test audio
audio = np.random.random(16000) * 0.5
peak_db = compute_peak_dbfs(audio)
print(f"Peak level: {peak_db:.1f} dBFS")

# Extract slice
slice_audio = extract_audio_slice(audio, 5.0, 2.0, 2.0)
print(f"Slice length: {len(slice_audio)} samples")
```

**What You Learned**:
- Audio level calculations (dBFS, peak, RMS)
- Audio slicing and manipulation
- Working with numpy arrays
- Audio analysis concepts

---

## Part 5: MP4 Export with Gong Snippets

**File**: `export_snippets.py`  
**Goal**: Create video snippets from detected gong events

### Step 1: Understand Video Export

Key concepts:
- **Audio-Only Video**: MP4 files with audio track but no visual content
- **Context Windows**: Extract audio before and after the detected event
- **FFmpeg Integration**: Using subprocess to call ffmpeg command-line tool

### Step 2: Create Basic Structure

```python
#!/usr/bin/env python3
"""Export gong detection snippets as MP4 files."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import pandas as pd

from audio_utils import analyze_audio_slice_levels


class GongSnippetExporter:
    """Export detected gong snippets as MP4 files using ffmpeg."""
    
    def __init__(
        self,
        source_audio_path: str,
        output_directory: str = "gong_snippets",
        episode_name: str = "episode"
    ) -> None:
        """Initialize the gong snippet exporter."""
        # Your code will go here...


def main() -> None:
    """Main execution."""
    # Your code will go here...


if __name__ == "__main__":
    main()
```

### Step 3: Add Detection Loading

```python
    def load_detections_from_csv(self, csv_path: str) -> List[Tuple[float, float]]:
        """Load gong detections from CSV file."""
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Detection CSV not found: {csv_path}")
        
        print(f"📊 Loading detections from: {csv_file}")
        
        try:
            # Try pandas first (handles various CSV formats well)
            df = pd.read_csv(csv_file)
            
            # Look for common column names
            timestamp_col = None
            confidence_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'timestamp' in col_lower or 'time' in col_lower:
                    timestamp_col = col
                elif 'confidence' in col_lower or 'score' in col_lower:
                    confidence_col = col
                    
            if timestamp_col is None:
                # Assume first column is timestamp
                timestamp_col = df.columns[0]
                print(f"⚠️  Using first column '{timestamp_col}' as timestamp")
                
            if confidence_col is None:
                # Assume second column is confidence, or default to 1.0
                if len(df.columns) > 1:
                    confidence_col = df.columns[1]
                    print(f"⚠️  Using second column '{confidence_col}' as confidence")
                else:
                    print("⚠️  No confidence column found, using 1.0 for all detections")
            
            detections = []
            for _, row in df.iterrows():
                timestamp = float(row[timestamp_col])
                confidence = float(row[confidence_col]) if confidence_col else 1.0
                detections.append((timestamp, confidence))
                
            print(f"✅ Loaded {len(detections)} detections from CSV")
            return detections
            
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file {csv_path}: {e}")
```

### Step 4: Add Snippet Extraction

```python
    def extract_snippet(
        self,
        timestamp: float,
        output_filename: str,
        duration_before: float = 20.0,
        duration_after: float = 5.0
    ) -> bool:
        """Extract audio snippet around timestamp using ffmpeg."""
        try:
            # Calculate start and end times
            start_time = max(0, timestamp - duration_before)
            end_time = timestamp + duration_after
            
            # Build ffmpeg command
            cmd = [
                "ffmpeg",
                "-i", str(self.source_audio_path),
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-c:a", "aac",  # Audio codec
                "-b:a", "128k",  # Bitrate
                "-y",  # Overwrite output
                str(self.output_directory / output_filename)
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            print(f"Command: {' '.join(cmd)}")
            return False
        except Exception as e:
            print(f"❌ Error extracting snippet: {e}")
            return False
```

### Step 5: Add Main Export Function

```python
    def export_all_snippets(
        self,
        detections: List[Tuple[float, float]],
        duration_before: float = 20.0,
        duration_after: float = 5.0,
        include_audio_analysis: bool = True
    ) -> Dict[str, Any]:
        """Export all detected gong snippets."""
        print(f"🎬 Exporting {len(detections)} gong snippets...")
        
        successful_exports = 0
        failed_exports = 0
        export_details = []
        
        for i, (timestamp, confidence) in enumerate(detections):
            # Generate filename
            filename = self.generate_snippet_filename(timestamp, i)
            output_filename = f"{filename}.mp4"
            
            print(f"  [{i+1}/{len(detections)}] Exporting: {output_filename}")
            
            # Extract snippet
            success = self.extract_snippet(
                timestamp, output_filename, duration_before, duration_after
            )
            
            if success:
                successful_exports += 1
                print(f"    ✅ Exported successfully")
                
                # Optional: Get audio levels
                if include_audio_analysis:
                    levels = self.get_snippet_audio_levels(
                        timestamp, duration_before, duration_after
                    )
                    if levels:
                        peak_db, rms_db = levels
                        print(f"    📊 Audio levels: Peak={peak_db:.1f}dBFS, RMS={rms_db:.1f}dBFS")
                
                export_details.append({
                    "filename": output_filename,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "status": "success"
                })
            else:
                failed_exports += 1
                export_details.append({
                    "filename": output_filename,
                    "timestamp": timestamp,
                    "confidence": confidence,
                    "status": "failed"
                })
        
        # Print summary
        print(f"\n📊 Export Summary:")
        print(f"  ✅ Successful: {successful_exports}")
        print(f"  ❌ Failed: {failed_exports}")
        print(f"  📁 Output directory: {self.output_directory}")
        
        return {
            "successful_exports": successful_exports,
            "failed_exports": failed_exports,
            "total_detections": len(detections),
            "export_details": export_details
        }
```

### Testing Part 5

```bash
# Export snippets from detection results
python export_snippets.py audio.wav detections.csv --output-dir snippets

# Use different time windows
python export_snippets.py audio.wav detections.csv --before 30 --after 10
```

**What You Learned**:
- Using ffmpeg for video creation
- Loading detection data from CSV/JSON
- Audio-to-video conversion
- File naming and organization

---

## Part 6: Refactor into Importable Modules

**File**: `yamnet_runner.py`  
**Goal**: Create clean, reusable modules with proper structure

### Step 1: Understand Modular Design

Key principles:
- **Single Responsibility**: Each module has one clear purpose
- **Reusability**: Functions can be imported and used elsewhere
- **Clean Interfaces**: Clear input/output contracts
- **Error Handling**: Graceful failure modes

### Step 2: Extract Core YAMNet Functionality

Move the YAMNet class from `detector.py` to `yamnet_runner.py`:

```python
"""YAMNet runner module for gong detection."""

import os
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


class YAMNetGongDetector:
    """YAMNet-based gong sound detector for audio analysis."""
    
    def __init__(self) -> None:
        """Initialize the YAMNet gong detector."""
        self.model: Optional[Any] = None
        self.class_names: Optional[List[str]] = None
        self.gong_class_index: int = 138  # YAMNet class index for "gong"
        
    def load_model(self) -> None:
        """Load the YAMNet model from TensorFlow Hub."""
        # Implementation from Part 1...
        
    def load_and_preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file for YAMNet inference."""
        # Implementation from Part 1...
        
    def run_inference(self, waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run YAMNet inference on audio waveform."""
        # Implementation from Part 1...
        
    def detect_gongs(
        self, 
        scores: np.ndarray, 
        confidence_threshold: float = 0.5
    ) -> List[Tuple[float, float]]:
        """Detect gong sounds based on YAMNet scores."""
        # Implementation from Part 1...
        
    def detections_to_dataframe(self, detections: List[Tuple[float, float]]) -> pd.DataFrame:
        """Convert detections to a pandas DataFrame."""
        # Implementation from Part 1...
```

### Step 3: Update Script Files

Update all script files to use the new module:

```python
# In test_yamnet.py, detect.py, batch_detect.py
from yamnet_runner import YAMNetGongDetector
```

### Step 4: Add Main Pattern

Ensure all scripts follow the main pattern:

```python
if __name__ == "__main__":
    # Script execution code here
    main()
```

### Testing Part 6

```python
# Test importing the module
from yamnet_runner import YAMNetGongDetector

detector = YAMNetGongDetector()
detector.load_model()
print("✅ Module imported successfully!")

# Test using in your own scripts
import yamnet_runner
detector = yamnet_runner.YAMNetGongDetector()
```

**What You Learned**:
- Modular design principles
- Import/export patterns
- Clean separation of concerns
- Professional project structure

---

## 🎓 Next Steps

Now that you've built the complete system:

1. **Experiment**: Try different confidence thresholds and see how they affect detection
2. **Extend**: Add support for other audio events beyond gongs
3. **Optimize**: Improve performance for large audio files
4. **Visualize**: Add plotting capabilities for detection results
5. **Deploy**: Create a web interface or API

## 🧪 Testing Your Understanding

After completing each part, test your knowledge:

1. **Part 1**: Can you modify the confidence threshold and see how it affects detections?
2. **Part 2**: Try converting different audio formats (MP3, M4A, etc.)
3. **Part 3**: Process a folder with mixed audio files and handle errors
4. **Part 4**: Calculate audio levels for different types of sounds
5. **Part 5**: Create snippets with different time windows
6. **Part 6**: Import and use the modules in your own scripts

---

**Congratulations! 🎉** You've built a complete audio analysis pipeline from scratch. Each part builds on the previous, creating a solid foundation for audio machine learning projects. 