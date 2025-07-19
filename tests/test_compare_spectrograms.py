"""
Tests for the spectrogram comparison script.

This module contains tests for the compare_spectrograms.py functionality.
"""

import pytest
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

import sys
import tempfile
import numpy as np
from unittest.mock import patch, MagicMock

# Add the gong_detector directory to the path for imports
gong_detector_path = Path(__file__).parent.parent / "gong_detector"
sys.path.insert(0, str(gong_detector_path))

try:
    from gong_detector.compare_spectrograms import (
        load_audio,
        compute_mel_spectrogram,
        get_audio_stats,
        plot_spectrograms
    )
except ImportError:
    # Fallback for when running tests from different directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gong_detector.compare_spectrograms import (
        load_audio,
        compute_mel_spectrogram,
        get_audio_stats,
        plot_spectrograms
    )


def test_load_audio_valid_file() -> None:
    """Test loading a valid audio file."""
    # Create a mock audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Mock librosa.load to return test data
        with patch('librosa.load') as mock_load:
            mock_audio = np.random.random(16000)  # 1 second at 16kHz
            mock_load.return_value = (mock_audio, 16000)
            
            audio, sr = load_audio(tmp_path)
            
            assert sr == 16000
            assert len(audio) == 16000
            assert isinstance(audio, np.ndarray)
            mock_load.assert_called_once_with(tmp_path, sr=16000, mono=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_load_audio_file_not_found() -> None:
    """Test loading a non-existent audio file."""
    with pytest.raises(FileNotFoundError):
        load_audio("nonexistent_file.wav")


def test_compute_mel_spectrogram() -> None:
    """Test mel spectrogram computation."""
    # Create test audio data
    audio = np.random.random(16000)  # 1 second at 16kHz
    sr = 16000
    
    # Mock librosa functions
    with patch('librosa.feature.melspectrogram') as mock_mel:
        with patch('librosa.power_to_db') as mock_db:
            mock_mel.return_value = np.random.random((128, 32))
            mock_db.return_value = np.random.random((128, 32))
            
            result = compute_mel_spectrogram(audio, sr)
            
            assert result.shape == (128, 32)
            mock_mel.assert_called_once()
            mock_db.assert_called_once()


def test_get_audio_stats() -> None:
    """Test audio statistics calculation."""
    # Create test audio data
    audio = np.array([0.5, -0.3, 0.8, -0.1, 0.2])
    sr = 16000
    
    stats = get_audio_stats(audio, sr)
    
    assert 'duration' in stats
    assert 'peak_amplitude' in stats
    assert 'rms' in stats
    assert 'db_rms' in stats
    assert stats['duration'] == len(audio) / sr
    assert stats['peak_amplitude'] == 0.8


def test_plot_spectrograms_mocked() -> None:
    """Test spectrogram plotting with mocked matplotlib."""
    # Create test audio data
    tbpn_audio = np.random.random(16000)
    reference_audio = np.random.random(16000)
    sr = 16000
    
    # Mock matplotlib and librosa.display
    with patch('matplotlib.pyplot.subplots') as mock_subplots:
        with patch('librosa.display.specshow') as mock_specshow:
            with patch('matplotlib.pyplot.show') as mock_show:
                with patch('matplotlib.pyplot.tight_layout') as mock_layout:
                    # Mock the subplot creation
                    mock_fig = MagicMock()
                    mock_ax1 = MagicMock()
                    mock_ax2 = MagicMock()
                    mock_subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
                    
                    # Mock the spectrogram display
                    mock_img = MagicMock()
                    mock_specshow.return_value = mock_img
                    
                    plot_spectrograms(
                        tbpn_audio, 
                        reference_audio, 
                        sr, 
                        "test_tbpn.wav", 
                        "test_reference.wav"
                    )
                    
                    # Verify the plotting functions were called
                    mock_subplots.assert_called_once()
                    assert mock_specshow.call_count == 2  # Called twice for both spectrograms
                    mock_show.assert_called_once()
                    mock_layout.assert_called_once()


def test_main_function_with_args() -> None:
    """Test the main function with command line arguments."""
    with patch('sys.argv', ['compare_spectrograms.py', 'test1.wav', 'test2.wav']):
        with patch('compare_spectrograms.load_audio') as mock_load:
            with patch('compare_spectrograms.plot_spectrograms') as mock_plot:
                # Mock audio loading
                mock_audio = np.random.random(16000)
                mock_load.return_value = (mock_audio, 16000)
                
                # Import and run main
                try:
                    from gong_detector.compare_spectrograms import main
                except ImportError:
                    from gong_detector.compare_spectrograms import main
                main()
                
                # Verify functions were called
                assert mock_load.call_count == 2
                mock_plot.assert_called_once() 