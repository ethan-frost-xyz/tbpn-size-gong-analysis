"""Configuration settings for the gong detector system."""

# Dual-cache settings
DUAL_CACHE = True  # Always cache both raw and preprocessed audio
DUAL_CACHE_FALLBACK = False  # Allow fallback to single cache if dual-cache fails

# Audio processing settings
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_FORMAT = "s16"  # 16-bit signed PCM

# YAMNet settings
YAMNET_TARGET_SAMPLE_RATE = 16000
YAMNET_WINDOW_DURATION = 0.96  # seconds

# Detection settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.94
DEFAULT_BATCH_SIZE = 2000

# Memory safety settings
MEMORY_SAFETY_ENABLED = True
LOW_MEMORY_THRESHOLD_GB = 4.0  # Trigger warnings below this
CRITICAL_MEMORY_THRESHOLD_GB = 2.0  # Stop processing below this
MEMORY_RESERVE_PERCENT = 25  # Reserve this % of total memory

# System-specific batch size limits
BATCH_SIZE_LIMITS = {
    "low_memory": 1000,  # <= 8GB total RAM
    "medium_memory": 2000,  # <= 16GB total RAM
    "high_memory": 4000,  # > 16GB total RAM
}

# Audio chunking for large files (seconds)
AUDIO_CHUNK_LIMITS = {
    "low_memory": 600,  # 10 minutes for <= 8GB RAM
    "medium_memory": 1200,  # 20 minutes for <= 16GB RAM
    "high_memory": 1800,  # 30 minutes for > 16GB RAM
}

# File paths
LOCAL_MEDIA_BASE = "data/local_media"
LOCAL_MEDIA_RAW = "data/local_media/raw"
LOCAL_MEDIA_PREPROCESSED = "data/local_media/preprocessed"
LOCAL_MEDIA_INDEX = "data/local_media/index.json"

# Temporary files
TEMP_AUDIO_DIR = "data/temp_audio"
CSV_RESULTS_DIR = "data/csv_results"

# YouTube settings
YT_DLP_RETRIES = 20
YT_DLP_FRAGMENT_RETRIES = 20
YT_DLP_CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
