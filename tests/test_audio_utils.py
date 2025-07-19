"""Test audio utilities for decibel estimation and slicing functionality."""

from typing import TYPE_CHECKING

import numpy as np
import pytest  # type: ignore

from gong_detector.audio_utils import (
    SILENCE_FLOOR_DBFS,
    MIN_AMPLITUDE,
    compute_peak_dbfs,
    compute_rms_dbfs, 
    compute_audio_levels,
    extract_audio_slice,
    get_slice_around_timestamp,
    analyze_audio_slice_levels,
    is_silent
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # type: ignore
    from _pytest.fixtures import FixtureRequest  # type: ignore
    from _pytest.logging import LogCaptureFixture  # type: ignore
    from _pytest.monkeypatch import MonkeyPatch  # type: ignore
    from pytest_mock.plugin import MockerFixture  # type: ignore


class TestDecibelCalculations:
    """Test dBFS calculation functions."""

    def test_compute_peak_dbfs_empty_waveform(self) -> None:
        """Test peak dBFS calculation with empty waveform."""
        waveform = np.array([])
        result = compute_peak_dbfs(waveform)
        assert result == SILENCE_FLOOR_DBFS

    def test_compute_peak_dbfs_silent_waveform(self) -> None:
        """Test peak dBFS calculation with silent waveform."""
        waveform = np.zeros(1000)
        result = compute_peak_dbfs(waveform)
        assert result == SILENCE_FLOOR_DBFS

    def test_compute_peak_dbfs_full_scale(self) -> None:
        """Test peak dBFS calculation with full-scale signal."""
        waveform = np.array([1.0, -1.0, 0.5, -0.5])
        result = compute_peak_dbfs(waveform)
        expected = 20.0 * np.log10(1.0)  # Should be 0 dBFS
        assert abs(result - expected) < 0.001

    def test_compute_peak_dbfs_half_scale(self) -> None:
        """Test peak dBFS calculation with half-scale signal."""
        waveform = np.array([0.5, -0.5, 0.25])
        result = compute_peak_dbfs(waveform)
        expected = 20.0 * np.log10(0.5)  # Should be ~-6 dBFS
        assert abs(result - expected) < 0.001

    def test_compute_rms_dbfs_empty_waveform(self) -> None:
        """Test RMS dBFS calculation with empty waveform."""
        waveform = np.array([])
        result = compute_rms_dbfs(waveform)
        assert result == SILENCE_FLOOR_DBFS

    def test_compute_rms_dbfs_silent_waveform(self) -> None:
        """Test RMS dBFS calculation with silent waveform."""
        waveform = np.zeros(1000)
        result = compute_rms_dbfs(waveform)
        assert result == SILENCE_FLOOR_DBFS

    def test_compute_rms_dbfs_sine_wave(self) -> None:
        """Test RMS dBFS calculation with sine wave."""
        # Create a sine wave with RMS = 1/sqrt(2) ≈ 0.707
        t = np.linspace(0, 1, 1000, endpoint=False)
        waveform = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        result = compute_rms_dbfs(waveform)
        expected_rms = 1.0 / np.sqrt(2)
        expected_dbfs = 20.0 * np.log10(expected_rms)
        assert abs(result - expected_dbfs) < 0.1  # Allow small numerical error

    def test_compute_audio_levels(self) -> None:
        """Test combined peak and RMS calculation."""
        waveform = np.array([1.0, -0.5, 0.5, 0.0])
        peak_dbfs, rms_dbfs = compute_audio_levels(waveform)
        
        expected_peak = 20.0 * np.log10(1.0)  # 0 dBFS
        expected_rms = 20.0 * np.log10(np.sqrt(np.mean(waveform ** 2)))
        
        assert abs(peak_dbfs - expected_peak) < 0.001
        assert abs(rms_dbfs - expected_rms) < 0.001


class TestAudioSlicing:
    """Test audio slicing functionality."""

    def test_extract_audio_slice_empty_waveform(self) -> None:
        """Test audio slice extraction with empty waveform."""
        waveform = np.array([])
        result = extract_audio_slice(waveform, timestamp=5.0)
        assert len(result) == 0

    def test_extract_audio_slice_normal_case(self) -> None:
        """Test audio slice extraction in normal case."""
        # Create 10-second audio at 16kHz
        sample_rate = 16000
        duration = 10.0
        waveform = np.sin(np.linspace(0, 2 * np.pi * 440 * duration, int(sample_rate * duration)))
        
        # Extract 2-second slice around 5-second mark
        result = extract_audio_slice(
            waveform=waveform,
            timestamp=5.0,
            duration_before=1.0,
            duration_after=1.0,
            sample_rate=sample_rate
        )
        
        expected_length = int(2.0 * sample_rate)  # 2 seconds total
        assert len(result) == expected_length

    def test_extract_audio_slice_boundary_conditions(self) -> None:
        """Test audio slice extraction at boundaries."""
        sample_rate = 16000
        waveform = np.ones(sample_rate)  # 1 second of audio
        
        # Extract slice that goes beyond the start
        result = extract_audio_slice(
            waveform=waveform,
            timestamp=0.2,  # Near start
            duration_before=0.5,  # Goes before start
            duration_after=0.3,
            sample_rate=sample_rate
        )
        
        expected_length = int(0.8 * sample_rate)  # 0.8 seconds total
        assert len(result) == expected_length

    def test_get_slice_around_timestamp(self) -> None:
        """Test simple centered slice extraction."""
        sample_rate = 16000
        waveform = np.ones(5 * sample_rate)  # 5 seconds
        
        result = get_slice_around_timestamp(
            waveform=waveform,
            timestamp=2.5,  # Center
            context_seconds=2.0,  # 1 second before and after
            sample_rate=sample_rate
        )
        
        expected_length = 2 * sample_rate  # 2 seconds total
        assert len(result) == expected_length

    def test_analyze_audio_slice_levels(self) -> None:
        """Test combined slice extraction and level analysis."""
        sample_rate = 16000
        # Create audio with known level
        waveform = np.full(5 * sample_rate, 0.5)  # 5 seconds at half scale
        
        peak_dbfs, rms_dbfs = analyze_audio_slice_levels(
            waveform=waveform,
            timestamp=2.5,
            context_seconds=2.0,
            sample_rate=sample_rate
        )
        
        expected_peak = 20.0 * np.log10(0.5)  # ~-6 dBFS
        expected_rms = 20.0 * np.log10(0.5)   # Same for constant signal
        
        assert abs(peak_dbfs - expected_peak) < 0.001
        assert abs(rms_dbfs - expected_rms) < 0.001


class TestSilenceDetection:
    """Test silence detection functionality."""

    def test_is_silent_empty_waveform(self) -> None:
        """Test silence detection with empty waveform."""
        waveform = np.array([])
        assert is_silent(waveform) is True

    def test_is_silent_zero_waveform(self) -> None:
        """Test silence detection with zero waveform."""
        waveform = np.zeros(1000)
        assert is_silent(waveform) is True

    def test_is_silent_very_quiet_waveform(self) -> None:
        """Test silence detection with very quiet waveform."""
        waveform = np.full(1000, MIN_AMPLITUDE / 2)  # Very quiet
        assert is_silent(waveform) is True

    def test_is_silent_loud_waveform(self) -> None:
        """Test silence detection with loud waveform."""
        waveform = np.full(1000, 0.5)  # Half scale
        assert is_silent(waveform) is False

    def test_is_silent_custom_threshold(self) -> None:
        """Test silence detection with custom threshold."""
        # Create audio at -50 dBFS
        amplitude = 10 ** (-50 / 20)
        waveform = np.full(1000, amplitude)
        
        # Should not be silent with lenient threshold
        assert is_silent(waveform, threshold_dbfs=-60.0) is False
        
        # Should be silent with strict threshold
        assert is_silent(waveform, threshold_dbfs=-40.0) is True


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_values(self) -> None:
        """Test with very small but non-zero values."""
        waveform = np.array([MIN_AMPLITUDE * 2, -MIN_AMPLITUDE * 2])
        
        peak_dbfs = compute_peak_dbfs(waveform)
        rms_dbfs = compute_rms_dbfs(waveform)
        
        # Should not return silence floor
        assert peak_dbfs > SILENCE_FLOOR_DBFS
        assert rms_dbfs > SILENCE_FLOOR_DBFS

    def test_mixed_loud_and_quiet(self) -> None:
        """Test with mix of loud and quiet audio."""
        # Start with silence, then loud section
        quiet_part = np.zeros(8000)  # 0.5 seconds of silence
        loud_part = np.full(8000, 0.8)  # 0.5 seconds at high level
        waveform = np.concatenate([quiet_part, loud_part])
        
        # Extract slice from quiet part
        quiet_peak, quiet_rms = analyze_audio_slice_levels(
            waveform, timestamp=0.25, context_seconds=0.4, sample_rate=16000
        )
        
        # Extract slice from loud part
        loud_peak, loud_rms = analyze_audio_slice_levels(
            waveform, timestamp=0.75, context_seconds=0.4, sample_rate=16000
        )
        
        assert quiet_peak == SILENCE_FLOOR_DBFS
        assert loud_peak > quiet_peak

    def test_slice_extraction_beyond_boundaries(self) -> None:
        """Test slice extraction that goes well beyond audio boundaries."""
        sample_rate = 16000
        waveform = np.ones(sample_rate)  # 1 second
        
        # Try to extract 10 seconds around 0.5 seconds
        result = extract_audio_slice(
            waveform=waveform,
            timestamp=0.5,
            duration_before=5.0,
            duration_after=5.0,
            sample_rate=sample_rate
        )
        
        expected_length = 10 * sample_rate
        assert len(result) == expected_length
        
        # Most of it should be zero-padded
        non_zero_samples = np.count_nonzero(result)
        assert non_zero_samples == len(waveform)  # Only original audio is non-zero 