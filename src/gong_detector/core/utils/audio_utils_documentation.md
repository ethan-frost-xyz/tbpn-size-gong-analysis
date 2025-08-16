# Audio Utilities Documentation

## Overview
Audio utilities for gong detection, providing functions for audio analysis, conversion, processing, and loudness measurement.

**🔄 REFACTORING UPDATE**: The `youtube_utils.py` module has been refactored into smaller, focused modules:
- `youtube/` - YouTube downloading, caching, and processing
- `loudness/` - LUFS and True Peak analysis
- `file_utils.py` - File system operations

Existing imports continue to work for backward compatibility.

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

## Modular Structure (New)

### youtube/ Package

**downloader.py**
- `download_youtube_audio(url, output_template, yt_dlp_options)` - Download audio from YouTube using yt-dlp
- `download_and_process_youtube_audio(url, output_path, start_time, duration)` - High-level download and process function
- `get_cookies_path()` - Get path to cookies file for bot detection bypass

**cache_manager.py**
- `save_raw_to_cache(temp_path, video_id)` - Save raw audio to cache
- `ensure_full_preprocessed_from_raw(raw_path, video_id)` - Convert raw to preprocessed audio

**audio_processor.py**
- `trim_from_preprocessed(preprocessed_path, output_path, start_time, duration)` - Trim segments from preprocessed audio
- `convert_and_trim_audio(input_file, output_path, start_time, duration)` - Convert and trim audio

**metadata_utils.py**
- `video_id_from_url(url)` - Extract YouTube video ID from URL
- `create_folder_name_from_title(video_title)` - Create folder name from video title
- `create_folder_name_from_date(upload_date)` - Create folder name from upload date
- `sanitize_title_for_folder(title)` - Sanitize title for filesystem use

### loudness/ Package

**lufs_analyzer.py**
- `compute_lufs_segments(video_id, timestamps, measurement_type, index)` - Compute LUFS loudness for audio segments using BS.1770-4 K-weighting and EBU R128 gating

**true_peak_analyzer.py**
- `compute_true_peak_segments(video_id, timestamps, measurement_type, index)` - Compute True Peak (dBTP) for audio segments using ITU-R BS.1770-4 standard with 4x oversampling

**batch_processor.py**
- `compute_batch_weighted_lufs(all_video_data, measurement_type, reference_lufs)` - Batch-weighted LUFS across multiple videos
- `compute_batch_weighted_dbtp(all_video_data, measurement_type, reference_dbtp)` - Batch-weighted True Peak across multiple videos

### file_utils.py

**File System Operations**
- `setup_directories()` - Create necessary directories and return paths
- `cleanup_old_temp_files(temp_dir, max_age_hours)` - Clean up old temporary audio files
- `create_temp_audio_path(temp_dir)` - Create unique temporary audio file path

## youtube_utils.py (Backward Compatibility)

**⚠️ DEPRECATED**: This module now imports from the new modular structure for backward compatibility. New code should use the specific modules above.

### Features
- YouTube download with yt-dlp
- Automatic format conversion to WAV
- Mono audio output at 16kHz
- Cookie support for bot detection bypass
- Error handling for common issues
- **EBU R128 compliance** (LUFS + True Peak with 4x oversampling)
- **Detection-level audio analysis** (integrated into gong detection pipeline)

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

# INTEGRATED DETECTION PIPELINE (Recommended)
# LUFS and True Peak are automatically computed for each gong detection
from gong_detector.core.pipeline.detection_pipeline import detect_from_youtube_comprehensive

result = detect_from_youtube_comprehensive(
    youtube_url="https://youtube.com/watch?v=...",
    threshold=0.94,
    use_version_one=True,
    use_local_media=True
)

# Access LUFS and True Peak for each detection
for i, detection in enumerate(result['detections']):
    lufs_metrics = result['detection_lufs_metrics'][i]
    dbtp_metrics = result['detection_dbtp_metrics'][i]
    
    print(f"Detection {i+1} at {detection[2]:.1f}s:")
    print(f"  LUFS: {lufs_metrics['integrated_lufs']:.1f} LUFS")
    print(f"  True Peak: {dbtp_metrics['integrated_dbtp']:.1f} dBTP")

# MANUAL ANALYSIS (Advanced)
# Compute LUFS loudness manually
from gong_detector.core.utils.loudness import compute_lufs_segments

timestamps = [(10.0, 15.0), (60.0, 65.0)]  # 5-second segments
lufs_results = compute_lufs_segments("VIDEO_ID", timestamps, "integrated")
for result in lufs_results:
    if result["valid"]:
        print(f"Segment {result['start_time']}-{result['end_time']}s: {result['lufs']:.1f} LUFS")

# Compute True Peak (dBTP) with 4x oversampling
from gong_detector.core.utils.loudness import compute_true_peak_segments

dbtp_results = compute_true_peak_segments("VIDEO_ID", timestamps, "integrated")
for result in dbtp_results:
    if result["valid"]:
        print(f"Segment {result['start_time']}-{result['end_time']}s: {result['dbtp']:.1f} dBTP")

# YouTube downloading with new modular approach
from gong_detector.core.utils.youtube import download_and_process_youtube_audio

audio_path, title, date = download_and_process_youtube_audio(
    "https://youtube.com/watch?v=...", 
    "output.wav",
    start_time=10,
    duration=30
)

# BACKWARD COMPATIBLE APPROACH (Still works)
from gong_detector.core.utils.youtube_utils import compute_lufs_segments, download_and_trim_youtube_audio
# ... same function calls as before
```

## Error Handling
- Silent audio returns `SILENCE_FLOOR_DBFS` (-80.0 dBFS)
- Empty waveforms handled gracefully
- YouTube bot detection provides helpful error messages
- File validation before processing
- **LUFS/True Peak failures**: Graceful fallback to zeros with error logging
- **Improved True Peak**: 4x oversampling prevents measurement failures 