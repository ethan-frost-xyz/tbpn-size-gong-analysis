# Decision Pathing Documentation for run_gong_detector.py

## Overview

The `run_gong_detector.py` script provides an interactive menu system for accessing all gong detector functionality. This document maps out the decision paths, user inputs, and script flows for each option.

## Main Menu Structure

```
GONG DETECTOR MASTER MENU
=========================
  ▶ Single Video Detection
    Bulk Processing
    Manual Sample Collection
    Negative Sample Collection
    Audio Conversion
    Model Management
=========================
```

## 1. Single Video Detection

### Decision Path
```
Single Video Detection
├── YouTube URL (required)
├── Confidence threshold (default: 0.94)
├── Use trained classifier (y/n, default: y)
├── Batch size (default: 2000)
├── Save positive samples (y/n, default: n)
├── Keep temporary audio files (y/n, default: n)
├── Start time in seconds (optional)
└── Duration in seconds (optional)
```

### Script Flow
1. **Input Validation**: All numeric inputs are validated
2. **Parameter Processing**: Optional trimming parameters converted to integers
3. **Function Call**: `detect_from_youtube_comprehensive()`
4. **Output**: Results summary with total duration and detection count

### Key Parameters
- **threshold**: Controls detection sensitivity (0.0-1.0)
- **use_version_one**: Toggles between base YAMNet and trained classifier (defaults to trained classifier)
- **batch_size**: Affects memory usage and processing speed
- **start_time/duration**: Enables partial video processing

### Output Information
- **Total Duration**: Video length in seconds
- **Detection Count**: Number of gongs found
- **Processing Status**: Success/failure indicators

## 2. Bulk Processing

### Decision Path
```
Bulk Processing
├── Links file detection (automatic)
│   └── Required: data/tbpn_ytlinks/tbpn_youtube_links.txt
├── Confidence threshold (default: 0.94)
├── Use trained classifier (y/n, default: y)
├── Save positive samples (y/n, default: n)
└── Save results to CSV (y/n, default: n)
```

### Script Flow
1. **File Validation**: Checks for required YouTube links file
2. **Parameter Collection**: User inputs for processing options
3. **sys.argv Setup**: Configures command-line arguments for bulk processor
4. **Function Call**: `bulk_processor_main()`

### File Requirements
- **Links File**: Must contain YouTube URLs (one per line)
- **Location**: `data/tbpn_ytlinks/tbpn_youtube_links.txt` (exact path required)
- **Format**: Plain text file with .txt extension

### Command Line Arguments Generated
- `--threshold`: Detection confidence threshold
- `--version_one`: Use trained classifier (when enabled)
- `--save_positive_samples`: Save detected samples (when enabled)
- `--csv`: Save results to CSV (when enabled)

## 3. Manual Sample Collection

### Decision Path
```
Manual Sample Collection
├── YouTube URL (required)
├── Timestamp in seconds (required)
└── Confidence value (default: 1.0)
```

### Script Flow
1. **Input Collection**: URL and timestamp for gong occurrence
2. **Function Call**: `process_single_sample()`
3. **Output**: Success/failure status

### Use Case
- **Purpose**: Training data collection
- **Confidence**: Typically 1.0 for manual detections
- **Output**: Saves audio sample for model training

### Training Data Output
- **Location**: Training data directory
- **Format**: Audio samples with metadata
- **Use**: Model training and validation

## 4. Negative Sample Collection

### Decision Path
```
Negative Sample Collection
├── YouTube URL (required)
├── Number of samples (default: 5)
├── Detection threshold (default: 0.4)
├── Maximum confidence threshold (optional)
└── Keep temporary audio files (y/n, default: n)
```

### Script Flow
1. **Parameter Collection**: All negative sample parameters
2. **Function Call**: `collect_negative_samples()`
3. **Output**: Number of samples collected

### Key Parameters
- **num_samples**: How many negative samples to collect
- **threshold**: Minimum confidence to avoid (gong detection threshold)
- **max_threshold**: Upper bound for confidence range (optional)

### Collection Strategy
- **Purpose**: Find audio segments without gongs
- **Method**: Uses gong detection to avoid positive samples
- **Output**: Negative training samples for model improvement

## 5. Audio Conversion

### Decision Path
```
Audio Conversion
├── Input source (YouTube URL or local file path)
└── Output WAV file path (default: converted_audio.wav)
```

### Script Flow
1. **Input Validation**: Checks if source is URL or local file
2. **Function Call**: `convert_youtube_audio()`
3. **Output**: Success/failure with file path

### Supported Inputs
- **YouTube URLs**: Direct video links
- **Local Files**: Audio/video files on disk
- **Output**: Always WAV format

### Conversion Process
- **YouTube**: Downloads and converts to WAV
- **Local Files**: Converts existing audio/video to WAV
- **Quality**: Maintains original audio quality
- **Format**: Standard WAV format for processing

## 6. Model Management

### Decision Path
```
Model Management
└── No user input required (automatic)
```

### Script Flow
1. **Model Loading**: Attempts to load YAMNet model
2. **Performance Info**: Displays TensorFlow configuration
3. **Classifier Loading**: Attempts to load trained classifier
4. **Status Report**: Success/failure for each component

### Output Information
- **Model Status**: ✓/✗ for YAMNet loading
- **Performance Config**: Thread counts and batch size
- **Classifier Status**: ✓/⚠ for trained classifier

### System Health Check
- **YAMNet Model**: Core audio classification model
- **Trained Classifier**: Custom gong detection model
- **Performance Settings**: TensorFlow thread configuration
- **Memory Usage**: Batch size and processing parameters

## Input Validation Rules

### Numeric Inputs
- **Float Validation**: Ensures valid decimal numbers
- **Integer Validation**: Ensures whole numbers
- **Default Values**: Provided for most parameters
- **Error Handling**: Re-prompts on invalid input

### Boolean Inputs
- **Yes/No Format**: Accepts 'y', 'yes', 'n', 'no'
- **Case Insensitive**: Converts to lowercase
- **Default Values**: Clear indication of default choice
- **Error Handling**: Re-prompts on invalid input

### String Inputs
- **Required Fields**: Must not be empty
- **Optional Fields**: Can be skipped with Enter
- **Default Values**: Shown in parentheses

## Error Handling

### Common Error Scenarios
1. **Invalid URLs**: YouTube URL validation
2. **File Not Found**: Links file or audio file missing
3. **Model Loading**: TensorFlow/YAMNet initialization issues
4. **Network Issues**: YouTube download failures
5. **Permission Errors**: File system access issues

### User Experience
- **Clear Error Messages**: Specific failure reasons
- **Graceful Degradation**: Continues to menu after errors
- **Input Validation**: Prevents invalid parameter combinations
- **Keyboard Interrupt**: Allows cancellation with Ctrl+C

## Configuration Dependencies

### Required Files
- **YAMNet Model**: TensorFlow model files
- **Trained Classifier**: Custom model weights (optional)
- **Links File**: YouTube URLs for bulk processing
- **Output Directories**: For saving samples and results

### Environment Requirements
- **Python Dependencies**: TensorFlow, librosa, yt-dlp, etc.
- **System Resources**: Sufficient RAM for batch processing
- **Network Access**: For YouTube video downloads
- **Disk Space**: For temporary audio files and results

## Decision Tree Summary

```
Main Menu
├── Single Video Detection
│   ├── URL Input → Parameter Collection → Detection → Results
│   └── Optional: Trimming, Sample Saving, Audio Keeping
├── Bulk Processing
│   ├── File Validation → Parameter Collection → Batch Processing
│   └── Optional: CSV Export, Sample Saving
├── Manual Sample Collection
│   ├── URL + Timestamp → Sample Extraction → Training Data
│   └── Purpose: Model Training Enhancement
├── Negative Sample Collection
│   ├── URL + Parameters → Negative Sample Detection → Training Data
│   └── Purpose: Model Training Enhancement
├── Audio Conversion
│   ├── Source + Output → Format Conversion → WAV File
│   └── Purpose: Audio Processing Pipeline
└── Model Management
    ├── Automatic → Status Check → Configuration Display
    └── Purpose: System Health Check
```

## Default Behavior Summary

### Detection Paths (All Default to Trained Classifier)
- **Single Video Detection**: `use_version_one = True`
- **Bulk Processing**: `use_version_one = True`
- **Manual Sample Collection**: Uses trained classifier when applicable
- **Negative Sample Collection**: Uses trained classifier for detection

### File Requirements (Strict)
- **Bulk Processing**: Requires exact file `data/tbpn_ytlinks/tbpn_youtube_links.txt`
- **No Fallback**: System will not search for alternative files

### User Experience
- **Consistent Defaults**: Trained classifier enabled by default
- **Clear Error Messages**: Specific guidance for missing files
- **Input Validation**: Robust error handling for all inputs

This documentation provides a complete map of all decision points, user inputs, and script flows for the gong detector interactive menu system. 