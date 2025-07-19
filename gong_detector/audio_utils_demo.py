"""Demonstration of audio_utils module functionality.

This script shows how to use the audio utilities for decibel estimation
and audio slicing in the context of gong detection analysis.
"""

from typing import List, Tuple
import numpy as np
from audio_utils import (
    compute_peak_dbfs,
    compute_rms_dbfs,
    analyze_audio_slice_levels,
    extract_audio_slice,
    is_silent
)


def demo_basic_level_analysis() -> None:
    """Demonstrate basic audio level analysis."""
    print("🎵 Basic Audio Level Analysis Demo")
    print("-" * 40)
    
    # Create sample audio with different levels
    quiet_audio = np.full(16000, 0.1)    # Quiet audio: -20 dBFS
    loud_audio = np.full(16000, 0.8)     # Loud audio: ~-2 dBFS
    
    for name, waveform in [("Quiet", quiet_audio), ("Loud", loud_audio)]:
        peak_dbfs = compute_peak_dbfs(waveform)
        rms_dbfs = compute_rms_dbfs(waveform)
        silent = is_silent(waveform)
        
        print(f"{name} audio:")
        print(f"  Peak: {peak_dbfs:.1f} dBFS")
        print(f"  RMS:  {rms_dbfs:.1f} dBFS")
        print(f"  Silent: {silent}")
        print()


def demo_gong_detection_context() -> None:
    """Demonstrate audio analysis in gong detection context."""
    print("🔔 Gong Detection Context Demo")
    print("-" * 40)
    
    # Simulate a podcast with background and gong
    sample_rate = 16000
    duration = 10.0  # 10 seconds
    
    # Create background audio (speech-like)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    background = np.random.normal(0, 0.05, len(t))  # Low-level background
    
    # Add a "gong" event at 5 seconds (simulated with higher amplitude)
    gong_start = int(5.0 * sample_rate)
    gong_duration = int(1.0 * sample_rate)
    background[gong_start:gong_start + gong_duration] += 0.3 * np.sin(2 * np.pi * 200 * t[gong_start:gong_start + gong_duration])
    
    # Simulate YAMNet detected a gong at 5.2 seconds
    gong_timestamp = 5.2
    
    print(f"Analyzing audio around gong detection at {gong_timestamp}s...")
    
    # Extract 20 seconds of context (as mentioned in the requirements)
    context_peak, context_rms = analyze_audio_slice_levels(
        waveform=background,
        timestamp=gong_timestamp,
        context_seconds=20.0,  # Will be zero-padded beyond audio boundaries
        sample_rate=sample_rate
    )
    
    # Also analyze just the gong region
    gong_slice = extract_audio_slice(
        waveform=background,
        timestamp=gong_timestamp,
        duration_before=0.5,
        duration_after=0.5,
        sample_rate=sample_rate
    )
    gong_peak = compute_peak_dbfs(gong_slice)
    gong_rms = compute_rms_dbfs(gong_slice)
    
    print(f"Context analysis (20s around gong):")
    print(f"  Peak: {context_peak:.1f} dBFS")
    print(f"  RMS:  {context_rms:.1f} dBFS")
    print()
    
    print(f"Gong region analysis (1s around detection):")
    print(f"  Peak: {gong_peak:.1f} dBFS")  
    print(f"  RMS:  {gong_rms:.1f} dBFS")
    print()
    
    # This could be used for scoring gong detections
    if gong_peak > context_peak + 3:  # 3 dB above context
        print("✅ Strong gong detection - significant level increase")
    else:
        print("⚠️  Weak gong detection - minimal level increase")


def demo_batch_analysis() -> None:
    """Demonstrate batch analysis of multiple detections."""
    print("📊 Batch Detection Analysis Demo")
    print("-" * 40)
    
    # Simulate multiple gong detections with timestamps
    detections = [
        (12.5, 0.75),   # timestamp, confidence
        (45.2, 0.82),
        (78.9, 0.61),
        (134.1, 0.89)
    ]
    
    # Create simulated podcast audio (2.5 minutes)
    sample_rate = 16000
    total_duration = 150.0  # 2.5 minutes
    waveform = np.random.normal(0, 0.03, int(total_duration * sample_rate))
    
    print(f"Analyzing {len(detections)} gong detections...")
    print()
    
    analysis_results: List[Tuple[float, float, float, float]] = []
    
    for timestamp, confidence in detections:
        # Analyze 20 seconds of context around each detection
        peak_dbfs, rms_dbfs = analyze_audio_slice_levels(
            waveform=waveform,
            timestamp=timestamp,
            context_seconds=20.0,
            sample_rate=sample_rate
        )
        
        analysis_results.append((timestamp, confidence, peak_dbfs, rms_dbfs))
        
        print(f"Detection at {timestamp:6.1f}s (conf: {confidence:.2f}):")
        print(f"  Context Peak: {peak_dbfs:6.1f} dBFS")
        print(f"  Context RMS:  {rms_dbfs:6.1f} dBFS")
        
        # Simple scoring based on audio levels
        if rms_dbfs > -30:
            score = "High"
        elif rms_dbfs > -50:
            score = "Medium" 
        else:
            score = "Low"
        print(f"  Audio Quality: {score}")
        print()
    
    # Summary statistics
    avg_peak = np.mean([r[2] for r in analysis_results])
    avg_rms = np.mean([r[3] for r in analysis_results])
    
    print("Summary:")
    print(f"  Average Peak: {avg_peak:.1f} dBFS")
    print(f"  Average RMS:  {avg_rms:.1f} dBFS")


def main() -> None:
    """Run all demonstrations."""
    print("🎵 Audio Utils Demo for YAMNet Gong Detection")
    print("=" * 60)
    print()
    
    demo_basic_level_analysis()
    demo_gong_detection_context()
    demo_batch_analysis()
    
    print("=" * 60)
    print("✅ Demo complete! The audio_utils module is ready for")
    print("   post-detection analysis and scoring in your gong")
    print("   detection pipeline.")


if __name__ == "__main__":
    main() 