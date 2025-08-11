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
