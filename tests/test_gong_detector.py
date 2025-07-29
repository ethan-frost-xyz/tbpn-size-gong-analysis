"""Test the core gong_detector functionality."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from gong_detector.core.data.csv_manager import CSVManager
from gong_detector.core.detector.yamnet_runner import YAMNetGongDetector


def test_yamnet_detector_initialization():
    """Test that YAMNet detector can be initialized."""
    detector = YAMNetGongDetector()
    assert detector is not None


def test_csv_manager_initialization():
    """Test that CSV manager can be initialized."""
    csv_manager = CSVManager()
    assert csv_manager is not None


def test_core_imports():
    """Test that core modules can be imported."""
    try:
        from gong_detector.core.pipeline import detection_pipeline

        assert detection_pipeline is not None
    except ImportError as e:
        pytest.fail(f"Failed to import detection_pipeline: {e}")

    try:
        from gong_detector.core.utils import audio_utils

        assert audio_utils is not None
    except ImportError as e:
        pytest.fail(f"Failed to import audio_utils: {e}")


def test_project_structure():
    """Test that essential project structure exists."""
    base_dir = Path(__file__).parent.parent

    # Check essential directories exist
    assert (base_dir / "src" / "gong_detector").exists(), (
        "gong_detector package should exist"
    )
    assert (base_dir / "src" / "gong_detector" / "core").exists(), (
        "core module should exist"
    )
    assert (base_dir / "src" / "gong_detector" / "core" / "detector").exists(), (
        "detector module should exist"
    )
    assert (base_dir / "src" / "gong_detector" / "core" / "pipeline").exists(), (
        "pipeline module should exist"
    )
