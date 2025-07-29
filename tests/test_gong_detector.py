"""Test the core gong_detector functionality."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gong_detector import (
    DEFAULT_SAMPLE_RATE,
    SILENCE_FLOOR_DBFS,
    CSVManager,
    DetectionRecord,
    YAMNetGongDetector,
    __version__,
    analyze_audio_slice_levels,
    cleanup_old_temp_files,
    collect_negative_samples,
    compute_audio_levels,
    compute_peak_dbfs,
    compute_rms_dbfs,
    convert_youtube_audio,
    create_folder_name_from_date,
    create_folder_name_from_title,
    create_temp_audio_path,
    detect_from_youtube_comprehensive,
    download_and_trim_youtube_audio,
    extract_audio_slice,
    format_time,
    get_audio_info,
    get_audio_stats,
    get_slice_around_timestamp,
    is_silent,
    normalize_waveform,
    print_summary,
    sanitize_title_for_folder,
    save_positive_samples,
    save_results_to_csv,
    setup_directories,
    validate_audio_file,
)


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
    # Test that all main functions are importable
    assert CSVManager is not None
    assert YAMNetGongDetector is not None
    assert DetectionRecord is not None
    assert detect_from_youtube_comprehensive is not None
    assert convert_youtube_audio is not None
    assert validate_audio_file is not None
    assert get_audio_info is not None
    assert compute_peak_dbfs is not None
    assert compute_rms_dbfs is not None
    assert compute_audio_levels is not None
    assert extract_audio_slice is not None
    assert get_slice_around_timestamp is not None
    assert analyze_audio_slice_levels is not None
    assert is_silent is not None
    assert normalize_waveform is not None
    assert get_audio_stats is not None
    assert download_and_trim_youtube_audio is not None
    assert format_time is not None
    assert print_summary is not None
    assert cleanup_old_temp_files is not None
    assert create_folder_name_from_date is not None
    assert create_folder_name_from_title is not None
    assert create_temp_audio_path is not None
    assert sanitize_title_for_folder is not None
    assert setup_directories is not None
    assert save_positive_samples is not None
    assert save_results_to_csv is not None
    assert collect_negative_samples is not None
    assert SILENCE_FLOOR_DBFS is not None
    assert DEFAULT_SAMPLE_RATE is not None
    assert __version__ is not None


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
