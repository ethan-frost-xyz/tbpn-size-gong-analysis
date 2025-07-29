"""Application settings and configuration."""

import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings."""

    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    SRC_DIR = BASE_DIR / "src"
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"

    # Audio settings
    SAMPLE_RATE = 16000
    AUDIO_FORMAT = "wav"

    # Model settings
    MODEL_PATH = SRC_DIR / "gong_detector" / "core" / "models"

    # Database settings (if needed)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # API settings
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "5000"))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "app.log"


# Global settings instance
settings = Settings()
