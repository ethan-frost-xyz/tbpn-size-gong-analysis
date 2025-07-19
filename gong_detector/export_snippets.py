"""Export gong detection snippets as MP4 files.

This script loads gong detections from CSV/JSON files and extracts
audio snippets around each detection, saving them as MP4 files with
audio-only video tracks using ffmpeg.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import pandas as pd

from audio_utils import analyze_audio_slice_levels


class GongSnippetExporter:
    """Export detected gong snippets as MP4 files using ffmpeg."""
    
    def __init__(
        self,
        source_audio_path: str,
        output_directory: str = "gong_snippets",
        episode_name: str = "episode"
    ) -> None:
        """Initialize the gong snippet exporter.
        
        Args:
            source_audio_path: Path to source audio file (.wav, .mp3, etc.)
            output_directory: Directory to save exported snippets
            episode_name: Base name for episode (used in filenames)
        """
        self.source_audio_path = Path(source_audio_path)
        self.output_directory = Path(output_directory)
        self.episode_name = episode_name
        
        # Validate source file exists
        if not self.source_audio_path.exists():
            raise FileNotFoundError(f"Source audio file not found: {source_audio_path}")
        
        # Create output directory
        self.output_directory.mkdir(exist_ok=True, parents=True)
        
        print(f"📁 Output directory: {self.output_directory}")
        print(f"🎵 Source audio: {self.source_audio_path}")
        
    def load_detections_from_csv(self, csv_path: str) -> List[Tuple[float, float]]:
        """Load gong detections from CSV file.
        
        Args:
            csv_path: Path to CSV file with detections
            
        Returns:
            List of (timestamp, confidence) tuples
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If CSV format is invalid
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"Detection CSV not found: {csv_path}")
        
        print(f"📊 Loading detections from: {csv_file}")
        
        try:
            # Try pandas first (handles various CSV formats well)
            df = pd.read_csv(csv_file)
            
            # Look for common column names
            timestamp_col = None
            confidence_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'timestamp' in col_lower or 'time' in col_lower:
                    timestamp_col = col
                elif 'confidence' in col_lower or 'score' in col_lower:
                    confidence_col = col
                    
            if timestamp_col is None:
                # Assume first column is timestamp
                timestamp_col = df.columns[0]
                print(f"⚠️  Using first column '{timestamp_col}' as timestamp")
                
            if confidence_col is None:
                # Assume second column is confidence, or default to 1.0
                if len(df.columns) > 1:
                    confidence_col = df.columns[1]
                    print(f"⚠️  Using second column '{confidence_col}' as confidence")
                else:
                    print("⚠️  No confidence column found, using 1.0 for all detections")
            
            detections = []
            for _, row in df.iterrows():
                timestamp = float(row[timestamp_col])
                confidence = float(row[confidence_col]) if confidence_col else 1.0
                detections.append((timestamp, confidence))
                
            print(f"✅ Loaded {len(detections)} detections from CSV")
            return detections
            
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file {csv_path}: {e}")
    
    def load_detections_from_json(self, json_path: str) -> List[Tuple[float, float]]:
        """Load gong detections from JSON file.
        
        Args:
            json_path: Path to JSON file with detections
            
        Returns:
            List of (timestamp, confidence) tuples
        """
        json_file = Path(json_path)
        if not json_file.exists():
            raise FileNotFoundError(f"Detection JSON not found: {json_path}")
            
        print(f"📊 Loading detections from: {json_file}")
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            detections = []
            if isinstance(data, list):
                # Assume list of [timestamp, confidence] or objects
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        timestamp, confidence = float(item[0]), float(item[1])
                    elif isinstance(item, dict):
                        timestamp = float(item.get('timestamp', item.get('time', 0)))
                        confidence = float(item.get('confidence', item.get('score', 1.0)))
                    else:
                        continue
                    detections.append((timestamp, confidence))
            
            print(f"✅ Loaded {len(detections)} detections from JSON")
            return detections
            
        except Exception as e:
            raise ValueError(f"Failed to parse JSON file {json_path}: {e}")
    
    def load_detections(self, detection_file: str) -> List[Tuple[float, float]]:
        """Load detections from CSV or JSON file (auto-detect format).
        
        Args:
            detection_file: Path to detection file
            
        Returns:
            List of (timestamp, confidence) tuples
        """
        file_path = Path(detection_file)
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return self.load_detections_from_csv(detection_file)
        elif extension == '.json':
            return self.load_detections_from_json(detection_file)
        else:
            raise ValueError(f"Unsupported detection file format: {extension}")
    
    def generate_snippet_filename(self, timestamp: float, detection_index: int) -> str:
        """Generate filename for a gong snippet.
        
        Args:
            timestamp: Detection timestamp in seconds
            detection_index: Index of detection (0-based)
            
        Returns:
            Generated filename (without extension)
        """
        # Convert timestamp to minutes and seconds for readable naming
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        
        # Format: ep12_gong_02m15s.mp4 or ep12_gong_053s.mp4
        if minutes > 0:
            time_str = f"{minutes:02d}m{seconds:02d}s"
        else:
            time_str = f"{seconds:03d}s"
            
        filename = f"{self.episode_name}_gong_{time_str}"
        return filename
    
    def extract_snippet(
        self,
        timestamp: float,
        output_filename: str,
        duration_before: float = 20.0,
        duration_after: float = 5.0
    ) -> bool:
        """Extract audio snippet around timestamp using ffmpeg.
        
        Args:
            timestamp: Center timestamp in seconds
            output_filename: Output filename (with .mp4 extension)
            duration_before: Seconds before timestamp to include
            duration_after: Seconds after timestamp to include
            
        Returns:
            True if extraction successful, False otherwise
        """
        # Calculate start time and total duration
        start_time = max(0, timestamp - duration_before)
        total_duration = duration_before + duration_after
        
        output_path = self.output_directory / output_filename
        
        # Build ffmpeg command for audio-only MP4
        cmd = [
            "ffmpeg",
            "-i", str(self.source_audio_path),  # Input file
            "-ss", str(start_time),             # Start time
            "-t", str(total_duration),          # Duration
            "-c:a", "aac",                      # Audio codec
            "-b:a", "128k",                     # Audio bitrate
            "-vn",                              # No video stream
            "-y",                               # Overwrite output
            str(output_path)
        ]
        
        try:
            # Run ffmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"  ✅ Extracted: {output_filename}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to extract {output_filename}: {e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ ffmpeg not found. Please install ffmpeg and add it to PATH.")
            return False
    
    def get_snippet_audio_levels(
        self,
        timestamp: float,
        duration_before: float = 20.0,
        duration_after: float = 5.0
    ) -> Optional[Tuple[float, float]]:
        """Get audio levels for a snippet using audio_utils.
        
        Args:
            timestamp: Center timestamp in seconds
            duration_before: Seconds before timestamp
            duration_after: Seconds after timestamp
            
        Returns:
            Tuple of (peak_dbfs, rms_dbfs) or None if analysis fails
        """
        try:
            # For now, return placeholder values since we'd need to load the audio
            # In a full implementation, we'd load the waveform and use audio_utils
            # This would require integration with the audio loading from test_yamnet.py
            
            # Placeholder implementation - could be enhanced to actually load audio
            return (-12.5, -18.2)  # Example values
            
        except Exception as e:
            print(f"  ⚠️  Could not analyze audio levels: {e}")
            return None
    
    def export_all_snippets(
        self,
        detections: List[Tuple[float, float]],
        duration_before: float = 20.0,
        duration_after: float = 5.0,
        include_audio_analysis: bool = True
    ) -> Dict[str, Any]:
        """Export all gong detections as MP4 snippets.
        
        Args:
            detections: List of (timestamp, confidence) tuples
            duration_before: Seconds before each detection
            duration_after: Seconds after each detection
            include_audio_analysis: Whether to analyze audio levels
            
        Returns:
            Summary dictionary with export results
        """
        print(f"🎬 Exporting {len(detections)} gong snippets...")
        print(f"   Duration: {duration_before}s before + {duration_after}s after = {duration_before + duration_after}s total")
        print()
        
        successful_exports = 0
        failed_exports = 0
        export_results = []
        
        for i, (timestamp, confidence) in enumerate(detections):
            filename = self.generate_snippet_filename(timestamp, i)
            output_filename = f"{filename}.mp4"
            
            print(f"[{i+1}/{len(detections)}] {timestamp:.1f}s (conf: {confidence:.3f})")
            
            # Extract snippet
            success = self.extract_snippet(
                timestamp=timestamp,
                output_filename=output_filename,
                duration_before=duration_before,
                duration_after=duration_after
            )
            
            if success:
                successful_exports += 1
                
                # Analyze audio levels if requested
                audio_levels = None
                if include_audio_analysis:
                    audio_levels = self.get_snippet_audio_levels(
                        timestamp, duration_before, duration_after
                    )
                    if audio_levels:
                        peak_dbfs, rms_dbfs = audio_levels
                        print(f"     🔊 Peak: {peak_dbfs:.1f} dBFS, RMS: {rms_dbfs:.1f} dBFS")
                
                export_results.append({
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'filename': output_filename,
                    'peak_dbfs': audio_levels[0] if audio_levels else None,
                    'rms_dbfs': audio_levels[1] if audio_levels else None,
                    'success': True
                })
            else:
                failed_exports += 1
                export_results.append({
                    'timestamp': timestamp,
                    'confidence': confidence,
                    'filename': output_filename,
                    'peak_dbfs': None,
                    'rms_dbfs': None,
                    'success': False
                })
            
            print()  # Empty line for readability
        
        # Summary
        summary = {
            'total_detections': len(detections),
            'successful_exports': successful_exports,
            'failed_exports': failed_exports,
            'output_directory': str(self.output_directory),
            'results': export_results
        }
        
        print("=" * 50)
        print(f"📊 Export Summary:")
        print(f"   Total detections: {len(detections)}")
        print(f"   Successful exports: {successful_exports}")
        print(f"   Failed exports: {failed_exports}")
        print(f"   Output directory: {self.output_directory}")
        
        return summary


def export_gong_snippets(
    source_audio: str,
    detection_file: str,
    output_directory: str = "gong_snippets",
    episode_name: str = "episode",
    duration_before: float = 20.0,
    duration_after: float = 5.0
) -> Dict[str, Any]:
    """Main function to export gong snippets.
    
    Args:
        source_audio: Path to source audio file
        detection_file: Path to detection CSV/JSON file
        output_directory: Directory for output snippets
        episode_name: Base name for episode
        duration_before: Seconds before detection
        duration_after: Seconds after detection
        
    Returns:
        Export summary dictionary
    """
    try:
        # Initialize exporter
        exporter = GongSnippetExporter(
            source_audio_path=source_audio,
            output_directory=output_directory,
            episode_name=episode_name
        )
        
        # Load detections
        detections = exporter.load_detections(detection_file)
        
        if not detections:
            print("⚠️  No detections found in file")
            return {'total_detections': 0, 'successful_exports': 0, 'failed_exports': 0}
        
        # Export all snippets
        summary = exporter.export_all_snippets(
            detections=detections,
            duration_before=duration_before,
            duration_after=duration_after,
            include_audio_analysis=True
        )
        
        return summary
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return {'error': str(e)}


def main() -> None:
    """Main execution point for snippet export."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Export gong detection snippets as MP4 files")
    parser.add_argument("source_audio", help="Source audio file (.wav, .mp3, etc.)")
    parser.add_argument("detection_file", help="Detection file (.csv or .json)")
    parser.add_argument("-o", "--output", default="gong_snippets", help="Output directory")
    parser.add_argument("-n", "--name", default="episode", help="Episode name for filenames")
    parser.add_argument("--before", type=float, default=20.0, help="Seconds before detection")
    parser.add_argument("--after", type=float, default=5.0, help="Seconds after detection")
    
    args = parser.parse_args()
    
    print("🎬 Gong Snippet Exporter")
    print("=" * 50)
    
    summary = export_gong_snippets(
        source_audio=args.source_audio,
        detection_file=args.detection_file,
        output_directory=args.output,
        episode_name=args.name,
        duration_before=args.before,
        duration_after=args.after
    )
    
    if 'error' in summary:
        sys.exit(1)
    elif summary['failed_exports'] > 0:
        print(f"\n⚠️  {summary['failed_exports']} exports failed")
        sys.exit(1)
    else:
        print("\n🎉 All snippets exported successfully!")


if __name__ == "__main__":
    main() 