#!/usr/bin/env python3
"""Test YAMNet gong detection functionality.

This is the definitive "hello world" test to prove the audio + model setup works.
Tests YAMNet model loading, audio processing, and gong detection on a sample file.
"""

import sys
from typing import List, Tuple

from yamnet_runner import YAMNetGongDetector


def test_yamnet_gong_detection(audio_path: str = "audio.wav") -> None:
    """Test the complete YAMNet gong detection pipeline.
    
    Args:
        audio_path: Path to the test audio file (should be mono, 16kHz)
    """
    print("🎵 Testing YAMNet Gong Detection")
    print("=" * 50)
    
    try:
        # Initialize detector
        detector = YAMNetGongDetector()
        
        # Load model
        detector.load_model()
        
        # Verify gong class exists and is accessible
        if detector.class_names is None:
            raise RuntimeError("Class names not loaded")
            
        assert detector.gong_class_index < len(detector.class_names), \
            f"Gong class index {detector.gong_class_index} out of range"
        assert "gong" in detector.class_names[detector.gong_class_index].lower(), \
            f"Class at index {detector.gong_class_index} is not gong: {detector.class_names[detector.gong_class_index]}"
        
        print(f"✅ Gong class verification passed: '{detector.class_names[detector.gong_class_index]}'")
        
        # Load and preprocess audio (should be mono, 16kHz)
        waveform, sample_rate = detector.load_and_preprocess_audio(audio_path)
        
        # Verify audio properties
        print(f"Audio shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate}Hz")
        print(f"Duration: {len(waveform) / sample_rate:.2f} seconds")
        
        # Run inference
        scores, embeddings, spectrogram = detector.run_inference(waveform)
        
        # Detect gongs with confidence > 0.5
        detections = detector.detect_gongs(scores, confidence_threshold=0.5)
        
        # Print results
        if detections:
            print("\n" + "="*50)
            print("GONG DETECTIONS (confidence > 0.5)")
            print("="*50)
            print(f"{'Timestamp (s)':<15} {'Confidence':<12}")
            print("-" * 27)
            
            for timestamp, confidence in detections:
                print(f"{timestamp:<15.2f} {confidence:<12.4f}")
            
            print("="*50)
            
            # Convert to DataFrame and display
            df = detector.detections_to_dataframe(detections)
            print(f"\n📊 Detection DataFrame shape: {df.shape}")
            if not df.empty:
                print("DataFrame contents:")
                print(df.to_string(index=False))
                
                # Save to CSV
                output_path = "test_gong_detections.csv"
                df.to_csv(output_path, index=False)
                print(f"💾 Test detections saved to: {output_path}")
        else:
            print("No gong detections found with confidence > 0.5")
            
        print(f"\n🎉 Test completed successfully!")
        print(f"Found {len(detections)} gong detections")
        
        # Assert that we can access the gong class (main requirement)
        gong_scores = scores[:, detector.gong_class_index]
        print(f"✅ Successfully accessed gong class scores: {len(gong_scores)} predictions")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    """Main execution point for testing YAMNet gong detection."""
    # Default to audio.wav, or use command line argument
    audio_file = "audio.wav"
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    
    print(f"Using audio file: {audio_file}")
    test_yamnet_gong_detection(audio_file) 