# Audio Utilities Documentation

## Overview
Audio utilities for gong detection, providing functions for audio analysis, conversion, processing, and loudness measurement.

## audio_utils.py

### Core Functions

**Level Analysis**
- `compute_peak_dbfs(waveform)` - Calculate peak dBFS level
- `compute_rms_dbfs(waveform)` - Calculate RMS dBFS level  
- `compute_audio_levels(waveform)` - Get both peak and RMS levels
- `compute_loudness_metrics(waveform)` - Comprehensive audio analysis including clipping detection

**Audio Extraction**
- `extract_audio_slice(waveform, timestamp, duration_before, duration_after)` - Extract audio around timestamp
- `get_slice_around_timestamp(waveform, timestamp, context_seconds)` - Extract centered audio slice
- `analyze_audio_slice_levels(waveform, timestamp, context_seconds)` - Extract slice and analyze levels

**Analysis & Processing**
- `is_silent(waveform, threshold_dbfs)` - Check if audio is silent
- `normalize_waveform(waveform, target_level_dbfs)` - Normalize to target level
- `get_audio_stats(waveform)` - Get comprehensive audio statistics
- `compute_crest_factor(waveform)` - Calculate peak-to-RMS ratio

### Key Constants
- `SILENCE_FLOOR_DBFS: -80.0` - Silence threshold
- `DEFAULT_SAMPLE_RATE: 16000` - Default sample rate
- `DEFAULT_TARGET_LEVEL_DBFS: -3.0` - Default normalization target

## convert_audio.py

### Core Functions

**Audio Conversion**
- `convert_youtube_audio(url_or_path, output_wav_path)` - Convert YouTube URL or local file to WAV
- `_convert_to_wav(input_path, output_path)` - Convert audio to WAV using ffmpeg
- `_download_audio(url)` - Download audio from YouTube

**Validation & Info**
- `validate_audio_file(file_path)` - Check if file is valid audio
- `get_audio_info(file_path)` - Get audio file metadata using ffprobe

## youtube_utils.py

### LUFS Analysis Functions

**Loudness Measurement**
- `compute_lufs_segments(video_id, timestamps, measurement_type, index)` - Compute LUFS loudness for audio segments from raw audio using BS.1770-4 K-weighting and EBU R128 gating

**Dual-Cache Management**
- `save_raw_to_cache(temp_path, video_id)` - Save raw audio to cache
- `ensure_full_preprocessed_from_raw(raw_path, video_id)` - Convert raw to preprocessed audio
- `trim_from_preprocessed(preprocessed_path, output_path, start_time, duration)` - Trim segments from preprocessed audio

### Features
- YouTube download with yt-dlp
- Automatic format conversion to WAV
- Mono audio output at 16kHz
- Cookie support for bot detection bypass
- Error handling for common issues

### Requirements
- ffmpeg (for audio conversion)
- yt-dlp (for YouTube downloads)
- pyloudnorm (for LUFS analysis)
- librosa (for audio loading)
- Internet connection (for YouTube downloads)

## Usage Examples

```python
# Analyze audio levels
from gong_detector.core.utils.audio_utils import compute_loudness_metrics
import numpy as np

waveform = np.array([...])  # Your audio data
metrics = compute_loudness_metrics(waveform)
print(f"Peak: {metrics['peak_dbfs']:.1f} dBFS")
print(f"RMS: {metrics['rms_dbfs']:.1f} dBFS")

# Convert YouTube video
from gong_detector.core.utils.convert_audio import convert_youtube_audio

wav_path = convert_youtube_audio("https://youtube.com/watch?v=...", "output.wav")

# Compute LUFS loudness
from gong_detector.core.utils.youtube_utils import compute_lufs_segments

timestamps = [(10.0, 15.0), (60.0, 65.0)]  # 5-second segments
lufs_results = compute_lufs_segments("VIDEO_ID", timestamps, "integrated")
for result in lufs_results:
    if result["valid"]:
        print(f"Segment {result['start_time']}-{result['end_time']}s: {result['lufs']:.1f} LUFS")
```

## Error Handling
- Silent audio returns `SILENCE_FLOOR_DBFS` (-80.0 dBFS)
- Empty waveforms handled gracefully
- YouTube bot detection provides helpful error messages
- File validation before processing 