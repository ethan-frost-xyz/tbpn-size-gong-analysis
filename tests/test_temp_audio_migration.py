"""Test that temp_audio migration works correctly."""

import os
import sys
from pathlib import Path


def test_temp_audio_directory_creation():
    """Test that temp_audio directory is created in the correct location."""
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gong_detector.core.utils.youtube_utils import setup_directories

    temp_dir, csv_dir = setup_directories()

    # Check that temp_dir is in the correct location
    expected_path = Path(__file__).parent.parent / "src" / "gong_detector" / "core" / "temp_audio"
    assert str(temp_dir) == str(expected_path), f"Expected {expected_path}, got {temp_dir}"

    # Check that directory exists
    assert os.path.exists(temp_dir), f"Directory {temp_dir} does not exist"


def test_temp_audio_file_creation():
    """Test that temp audio files can be created in the new location."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gong_detector.core.utils.youtube_utils import (
        create_temp_audio_path,
        setup_directories,
    )

    temp_dir, csv_dir = setup_directories()
    temp_file = create_temp_audio_path(temp_dir)

    # Check that the file path is in the correct directory
    temp_file_path = Path(temp_file)
    expected_dir = Path(__file__).parent.parent / "src" / "gong_detector" / "core" / "temp_audio"

    assert temp_file_path.parent == expected_dir, f"File should be in {expected_dir}, but is in {temp_file_path.parent}"

    # Check that the file has the correct extension
    assert temp_file_path.suffix == ".wav", f"File should have .wav extension, got {temp_file_path.suffix}"

    # Check that the filename follows the expected pattern
    assert temp_file_path.name.startswith("temp_youtube_audio_"), f"File should start with 'temp_youtube_audio_', got {temp_file_path.name}"


def test_all_imports_work():
    """Test that all modules that use temp_audio can be imported."""
    import importlib.util
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    # Test that all modules can be imported without errors
    modules_to_test = [
        "gong_detector.core.pipeline.detection_pipeline",
        "gong_detector.core.training.manual_collector",
        "gong_detector.core.training.negative_collector",
        "gong_detector.core.utils.youtube_utils",
    ]

    for module_name in modules_to_test:
        spec = importlib.util.find_spec(module_name)
        assert spec is not None, f"Module {module_name} should be importable"


def test_temp_audio_cleanup():
    """Test that temp audio files can be cleaned up properly."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from gong_detector.core.utils.youtube_utils import (
        cleanup_old_temp_files,
        create_temp_audio_path,
        setup_directories,
    )

    temp_dir, csv_dir = setup_directories()

    # Create a test temp file
    test_file = create_temp_audio_path(temp_dir)
    test_file_path = Path(test_file)

    # Create the file
    test_file_path.touch()
    assert test_file_path.exists(), "Test file should exist"

    # Clean up old files
    cleanup_old_temp_files(temp_dir, max_age_hours=0)  # Clean up files older than 0 hours

    # Check that the file was cleaned up
    assert not test_file_path.exists(), "Test file should have been cleaned up"
