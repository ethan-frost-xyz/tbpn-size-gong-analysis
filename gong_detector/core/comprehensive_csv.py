"""Comprehensive CSV management for gong detection results.

This module provides a standardized, extensible system for collecting and
storing detection metadata across all videos in a bulk processing run.
Each row represents one detection event with rich contextual information.
"""

import csv
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class DetectionRecord:
    """Comprehensive record for a single gong detection event.

    This schema captures all available metadata for each detection,
    making it easy to analyze patterns and find edge cases later.
    """

    # Key fields
    video_title: str
    detection_timestamp_formatted: str  # HH:MM:SS format
    confidence: str  # Formatted to 3 decimal places
    video_max_confidence: str  # Formatted to 3 decimal places

    # Video metadata
    upload_date: str  # YYYYMMDD format
    upload_date_formatted: str  # YYYY-MM-DD format
    video_duration_seconds: float

    # Detection specifics
    detection_timestamp_seconds: float
    window_start_seconds: float

    # Processing context
    detection_threshold: str  # Formatted to 3 decimal places
    processing_date: str  # ISO format YYYY-MM-DD
    processing_time: str  # ISO format HH:MM:SS

    # Unique identifiers
    detection_id: str
    video_url: str

    # Future extensibility placeholders
    notes: str = ""
    validated: str = ""  # For future human validation
    model_version: str = "yamnet_default"


class ComprehensiveCSVManager:
    """Manages comprehensive CSV generation for gong detection results."""

    def __init__(self, csv_results_dir: str = "csv_results"):
        """Initialize CSV manager.

        Args:
            csv_results_dir: Directory to store CSV files
        """
        self.csv_results_dir = Path(csv_results_dir)
        self.csv_results_dir.mkdir(exist_ok=True)
        self.detection_records: list[DetectionRecord] = []

    def add_video_detections(
        self,
        video_url: str,
        video_title: str,
        upload_date: str,
        video_duration: float,
        max_confidence: float,
        threshold: float,
        detections: list[tuple[float, float, float]],
    ) -> None:
        """Add all detections from a single video to the comprehensive record.

        Args:
            video_url: YouTube URL
            video_title: Video title
            upload_date: Upload date in YYYYMMDD format
            video_duration: Total video duration in seconds
            max_confidence: Maximum confidence score in the video
            threshold: Detection threshold used
            detections: List of (window_start, confidence, display_timestamp) tuples
        """
        # Format upload date for human readability
        upload_date_formatted = self._format_upload_date(upload_date)

        # Current processing time
        now = datetime.now()
        processing_date = now.strftime("%Y-%m-%d")
        processing_time = now.strftime("%H:%M:%S")

        # Create a record for each detection
        for window_start, confidence, display_timestamp in detections:
            record = DetectionRecord(
                detection_id=str(uuid.uuid4()),
                video_url=video_url,
                video_title=video_title,
                upload_date=upload_date,
                upload_date_formatted=upload_date_formatted,
                video_duration_seconds=video_duration,
                detection_timestamp_seconds=display_timestamp,
                detection_timestamp_formatted=self._format_time(display_timestamp),
                window_start_seconds=window_start,
                confidence=f"{confidence:.3f}",
                video_max_confidence=f"{max_confidence:.3f}",
                detection_threshold=f"{threshold:.3f}",
                processing_date=processing_date,
                processing_time=processing_time,
            )
            self.detection_records.append(record)

    def save_comprehensive_csv(self, run_name: Optional[str] = None) -> str:
        """Save all collected detections to a comprehensive CSV file.

        Args:
            run_name: Optional name for this bulk run

        Returns:
            Path to the saved CSV file
        """
        if not self.detection_records:
            raise ValueError("No detection records to save")

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            filename = f"comprehensive_detections_{run_name}_{timestamp}.csv"
        else:
            filename = f"comprehensive_detections_{timestamp}.csv"

        csv_path = self.csv_results_dir / filename

        # Write CSV with all fields
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(DetectionRecord.__annotations__.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for record in self.detection_records:
                writer.writerow(asdict(record))

        return str(csv_path)

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics for all collected detections.

        Returns:
            Dictionary with summary statistics
        """
        if not self.detection_records:
            return {}

        # Calculate summary stats
        total_detections = len(self.detection_records)
        unique_videos = len({record.video_url for record in self.detection_records})

        confidences = [float(record.confidence) for record in self.detection_records]
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        min_confidence = min(confidences)

        return {
            "total_detections": total_detections,
            "unique_videos": unique_videos,
            "average_confidence": round(avg_confidence, 4),
            "max_confidence": round(max_confidence, 4),
            "min_confidence": round(min_confidence, 4),
        }

    def _format_upload_date(self, upload_date: str) -> str:
        """Format upload date from YYYYMMDD to YYYY-MM-DD.

        Args:
            upload_date: Date in YYYYMMDD format

        Returns:
            Date in YYYY-MM-DD format, or original if invalid
        """
        if not upload_date or len(upload_date) != 8:
            return upload_date

        try:
            year = upload_date[:4]
            month = upload_date[4:6]
            day = upload_date[6:8]
            return f"{year}-{month}-{day}"
        except (ValueError, IndexError):
            return upload_date

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string in HH:MM:SS format
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def clear_records(self) -> None:
        """Clear all collected detection records."""
        self.detection_records.clear()
