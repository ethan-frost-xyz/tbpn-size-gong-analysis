"""Pytest configuration and common fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_audio_file():
    """Provide path to a sample audio file for testing."""
    return Path(__file__).parent / "data" / "sample_audio.wav" 