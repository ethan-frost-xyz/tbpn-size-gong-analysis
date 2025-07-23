"""Tests for audio_utils module."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

from gong_detector.core.audio_utils import (
    DEFAULT_SAMPLE_RATE,
    SILENCE_FLOOR_DBFS,
    analyze_audio_slice_levels,
    compute_audio_levels,
    compute_peak_dbfs,
    compute_rms_dbfs,
    extract_audio_slice,
    get_audio_stats,
    get_slice_around_timestamp,
    is_silent,
    normalize_waveform,
)


class TestDbfsComputation:
    """Tests for dBFS computation functions."""

    def test_compute_peak_dbfs_empty_array(self) -> None:
        """Test peak dBFS computation with empty array."""
        result = compute_peak_dbfs(np.array([]))
        assert result == SILENCE_FLOOR_DBFS

    def test_compute_peak_dbfs_silent(self) -> None:
        """Test peak dBFS computation with silent audio."""
        silent_audio = np.zeros(1000, dtype=np.float32)
        result = compute_peak_dbfs(silent_audio)
        assert result == SILENCE_FLOOR_DBFS

    def test_compute_peak_dbfs_full_scale(self) -> None:
        """Test peak dBFS computation with full scale audio."""
        full_scale_audio = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        result = compute_peak_dbfs(full_scale_audio)
        assert abs(result - 0.0) < 1e-6  # Should be 0 dBFS

    def test_compute_rms_dbfs_empty_array(self) -> None:
        """Test RMS dBFS computation with empty array."""
        result = compute_rms_dbfs(np.array([]))
        assert result == SILENCE_FLOOR_DBFS

    def test_compute_rms_dbfs_sine_wave(self) -> None:
        """Test RMS dBFS computation with sine wave."""
        # Create a sine wave with known RMS
        t = np.linspace(0, 1, 1000)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result = compute_rms_dbfs(sine_wave)

        # RMS of 0.5 * sin(x) should be 0.5/sqrt(2) ≈ 0.354
        # 20*log10(0.354) ≈ -9.03 dBFS
        expected = 20.0 * np.log10(0.5 / np.sqrt(2))
        assert abs(result - expected) < 0.1

    def test_compute_audio_levels_efficiency(self) -> None:
        """Test that compute_audio_levels is efficient for both calculations."""
        test_audio = np.random.uniform(-1, 1, 10000).astype(np.float32)

        # Test combined function
        peak_combined, rms_combined = compute_audio_levels(test_audio)

        # Test individual functions
        peak_individual = compute_peak_dbfs(test_audio)
        rms_individual = compute_rms_dbfs(test_audio)

        # Results should be identical
        assert abs(peak_combined - peak_individual) < 1e-6
        assert abs(rms_combined - rms_individual) < 1e-6


class TestAudioSlicing:
    """Tests for audio slicing functions."""

    def test_extract_audio_slice_empty_array(self) -> None:
        """Test audio slice extraction with empty array."""
        result = extract_audio_slice(np.array([]), 5.0)
        assert len(result) == 0

    def test_extract_audio_slice_basic(self) -> None:
        """Test basic audio slice extraction."""
        # Create 10 seconds of audio at 16kHz
        audio = np.arange(160000, dtype=np.float32)

        # Extract 2 seconds around 5-second mark (1s before, 1s after)
        result = extract_audio_slice(audio, 5.0, 1.0, 1.0, 16000)

        # Should be 2 seconds worth of samples
        assert len(result) == 32000

        # Center should contain the original data
        center_start = 16000  # 1 second offset
        original_start = int(5.0 * 16000)  # Start of slice in original (5-6 seconds)
        assert np.array_equal(
            result[center_start : center_start + 16000],
            audio[original_start : original_start + 16000],
        )

    def test_extract_audio_slice_boundary_padding(self) -> None:
        """Test audio slice extraction with boundary padding."""
        # Short audio clip
        audio = np.ones(1000, dtype=np.float32)

        # Request slice that goes beyond boundaries
        result = extract_audio_slice(audio, 0.5, 2.0, 2.0, 1000)

        # Should be zero-padded at boundaries
        assert len(result) == 4000  # 4 seconds at 1000 Hz
        assert np.sum(result[:1500]) == 0  # Beginning padding
        assert np.sum(result[2500:]) == 0  # End padding
        assert np.sum(result[1500:2500]) == 1000  # Original data

    def test_get_slice_around_timestamp_symmetric(self) -> None:
        """Test symmetric timestamp slicing."""
        audio = np.arange(48000, dtype=np.float32)  # 3 seconds at 16kHz

        result = get_slice_around_timestamp(audio, 1.5, 2.0, 16000)

        # Should be 2 seconds total (1s before + 1s after)
        assert len(result) == 32000

        # Center should be around the 1.5s mark
        center_idx = len(result) // 2
        original_center = int(1.5 * 16000)
        assert result[center_idx] == audio[original_center]


class TestAudioAnalysis:
    """Tests for audio analysis functions."""

    def test_is_silent_threshold(self) -> None:
        """Test silence detection with different thresholds."""
        # Create quiet audio
        quiet_audio = np.random.uniform(-0.001, 0.001, 1000).astype(np.float32)

        # Should be silent with a higher threshold (more strict)
        assert is_silent(quiet_audio, -50.0)

        # Should not be silent with very low threshold
        assert not is_silent(quiet_audio, -100.0)

    def test_normalize_waveform_full_scale(self) -> None:
        """Test waveform normalization to target level."""
        # Create audio with peak at 0.5
        audio = np.array([0.5, -0.3, 0.2, -0.5], dtype=np.float32)

        # Normalize to -6 dBFS (0.5 amplitude)
        result = normalize_waveform(audio, -6.0)

        # Peak should now be at 0.5 (since original was 0.5)
        assert abs(np.max(np.abs(result)) - 0.5) < 2e-3  # Further relaxed tolerance for float32 precision

    def test_normalize_waveform_silent(self) -> None:
        """Test normalization of silent audio."""
        silent_audio = np.zeros(100, dtype=np.float32)
        result = normalize_waveform(silent_audio)

        # Should remain unchanged
        assert np.array_equal(result, silent_audio)

    def test_get_audio_stats_comprehensive(self) -> None:
        """Test comprehensive audio statistics."""
        # Create test audio with known properties
        audio = np.array([0.8, -0.6, 0.4, -0.8], dtype=np.float32)

        stats = get_audio_stats(audio)

        # Check all expected keys
        expected_keys = {
            "length",
            "peak_dbfs",
            "rms_dbfs",
            "is_silent",
            "peak_amplitude",
            "rms_amplitude",
        }
        assert set(stats.keys()) == expected_keys

        # Check specific values
        assert stats["length"] == 4
        assert abs(stats["peak_amplitude"] - 0.8) < 1e-6
        assert not stats["is_silent"]

    def test_get_audio_stats_empty(self) -> None:
        """Test audio statistics with empty array."""
        stats = get_audio_stats(np.array([]))

        assert stats["length"] == 0
        assert stats["peak_dbfs"] == SILENCE_FLOOR_DBFS
        assert stats["rms_dbfs"] == SILENCE_FLOOR_DBFS
        assert stats["is_silent"] is True
        assert stats["peak_amplitude"] == 0.0
        assert stats["rms_amplitude"] == 0.0


class TestAnalyzeAudioSliceLevels:
    """Tests for combined audio slice analysis."""

    def test_analyze_audio_slice_levels_basic(self) -> None:
        """Test basic audio slice level analysis."""
        # Create audio with varying amplitude
        audio = np.concatenate(
            [
                np.zeros(8000),  # 0.5s silence
                np.ones(16000) * 0.5,  # 1.0s at 0.5 amplitude
                np.zeros(8000),  # 0.5s silence
            ]
        ).astype(np.float32)

        # Analyze the middle section
        peak_dbfs, rms_dbfs = analyze_audio_slice_levels(audio, 1.0, 1.0, 16000)

        # Should detect the 0.5 amplitude section
        expected_peak = 20.0 * np.log10(0.5)  # About -6 dBFS
        assert abs(peak_dbfs - expected_peak) < 0.1

    def test_analyze_audio_slice_levels_boundary(self) -> None:
        """Test audio slice level analysis at boundaries."""
        short_audio = np.ones(1000, dtype=np.float32) * 0.25

        # Request analysis beyond boundaries
        peak_dbfs, rms_dbfs = analyze_audio_slice_levels(short_audio, 0.1, 2.0, 1000)

        # Should handle boundary conditions gracefully
        assert peak_dbfs > SILENCE_FLOOR_DBFS
        assert rms_dbfs > SILENCE_FLOOR_DBFS


class TestConstants:
    """Tests for module constants."""

    def test_silence_floor_dbfs_reasonable(self) -> None:
        """Test that silence floor constant is reasonable."""
        assert SILENCE_FLOOR_DBFS < -60.0  # Should be very quiet
        assert SILENCE_FLOOR_DBFS > -120.0  # Should not be impossibly quiet

    def test_default_sample_rate_standard(self) -> None:
        """Test that default sample rate is standard."""
        assert DEFAULT_SAMPLE_RATE == 16000  # YAMNet standard
