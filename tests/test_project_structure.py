"""Test the project structure and imports."""

import pytest
from pathlib import Path


def test_project_structure():
    """Test that the project structure follows best practices."""
    base_dir = Path(__file__).parent.parent
    
    # Check essential directories exist
    assert (base_dir / "src").exists(), "src directory should exist"
    assert (base_dir / "tests").exists(), "tests directory should exist"
    assert (base_dir / "config").exists(), "config directory should exist"
    assert (base_dir / "data").exists(), "data directory should exist"
    assert (base_dir / "docs").exists(), "docs directory should exist"
    assert (base_dir / "logs").exists(), "logs directory should exist"
    
    # Check src structure
    src_dir = base_dir / "src"
    assert (src_dir / "gong_detector").exists(), "gong_detector package should exist"
    assert (src_dir / "gong_detector" / "core").exists(), "core module should exist"
    
    # Check test structure
    tests_dir = base_dir / "tests"
    assert (tests_dir / "unit").exists(), "unit tests directory should exist"
    assert (tests_dir / "integration").exists(), "integration tests directory should exist"
    assert (tests_dir / "functional").exists(), "functional tests directory should exist"


def test_imports():
    """Test that key modules can be imported."""
    try:
        from src.gong_detector.core import __init__ as core_init
        assert core_init is not None
    except ImportError as e:
        pytest.fail(f"Failed to import core module: {e}")
    
    try:
        from config import settings
        assert settings is not None
    except ImportError as e:
        pytest.fail(f"Failed to import settings: {e}")


def test_config_structure():
    """Test configuration structure."""
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / "config"
    
    assert (config_dir / "settings.py").exists(), "settings.py should exist"
    assert (config_dir / "__init__.py").exists(), "config __init__.py should exist"


def test_data_structure():
    """Test data directory structure."""
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    # Check that data subdirectories exist
    assert (data_dir / "csv_dir").exists(), "csv_dir should exist"
    assert (data_dir / "csv_results").exists(), "csv_results should exist"
    assert (data_dir / "tbpn_ytlinks").exists(), "tbpn_ytlinks should exist" 