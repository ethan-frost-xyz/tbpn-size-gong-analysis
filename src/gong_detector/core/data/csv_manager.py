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
    upload_date_formatted: str  # YYYY-MM-DD format
    detection_timestamp_formatted: str  # HH:MM:SS format
    confidence: str  # Formatted to 3 decimal places
    youtube_timestamped_link: str  # YouTube URL with timestamp

    # Company metadata (manual)
    host_name: str = "" # john or jordy or other
    company_name: str = "" # company name
    funding_amount: str = "" # in millions
    funding_valuation: str = "" # in millions
    funding_round: str = "" # seed, series a, series b, etc.

    # Detection metadata
    detection_timestamp_seconds: float
    window_start_seconds: float
    video_max_confidence: str  # Formatted to 3 decimal places
    detection_threshold: str  # Formatted to 3 decimal places
    max_threshold: str  # Formatted to 3 decimal places (empty if not used)
    processing_date: str  # ISO format YYYY-MM-DD
    processing_time: str  # ISO format HH:MM:SS
    detection_id: str

    # Video metadata
    upload_date: str  # YYYYMMDD format
    video_duration_seconds: float
    video_url: str

    # Audio loudness metrics
    detection_peak_dbfs: str = ""  # Peak dBFS at detection timestamp
    detection_rms_dbfs: str = ""  # RMS dBFS at detection timestamp
    detection_crest_factor: str = ""  # Crest factor at detection timestamp
    detection_likely_clipped: str = ""  # Whether detection audio is likely clipped
    detection_peak_amplitude: str = ""  # Peak amplitude at detection timestamp
    detection_rms_amplitude: str = ""  # RMS amplitude at detection timestamp

    # Video-level audio metrics
    video_peak_dbfs: str = ""  # Peak dBFS for entire video
    video_rms_dbfs: str = ""  # RMS dBFS for entire video
    video_crest_factor: str = ""  # Crest factor for entire video
    video_likely_clipped: str = ""  # Whether video audio is likely clipped
    video_peak_amplitude: str = ""  # Peak amplitude for entire video
    video_rms_amplitude: str = ""  # RMS amplitude for entire video


class CSVManager:
    """Manages comprehensive CSV generation for gong detection results."""

    def __init__(self, csv_results_dir: str = "data/csv_results"):
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
        max_threshold: Optional[float],
        detections: list[tuple[float, float, float]],
        video_loudness_metrics: Optional[dict[str, float]] = None,
        detection_loudness_metrics: Optional[list[dict[str, float]]] = None,
    ) -> None:
        """Add all detections from a single video to the comprehensive record.

        Args:
            video_url: YouTube URL
            video_title: Video title
            upload_date: Upload date in YYYYMMDD format
            video_duration: Total video duration in seconds
            max_confidence: Maximum confidence score in the video
            threshold: Detection threshold used
            max_threshold: Maximum threshold used (if any)
            detections: List of (window_start, confidence, display_timestamp) tuples
            video_loudness_metrics: Optional dict with video-level loudness metrics
            detection_loudness_metrics: Optional list of dicts with detection-level loudness metrics
        """
        # Format upload date for human readability
        upload_date_formatted = self._format_upload_date(upload_date)

        # Current processing time
        now = datetime.now()
        processing_date = now.strftime("%Y-%m-%d")
        processing_time = now.strftime("%H:%M:%S")

        # Validate loudness metrics
        if detection_loudness_metrics is not None and len(
            detection_loudness_metrics
        ) != len(detections):
            raise ValueError(
                "detection_loudness_metrics length must match detections length"
            )

        # Create a record for each detection
        for i, (window_start, confidence, display_timestamp) in enumerate(detections):
            # Generate timestamped YouTube link
            timestamped_link = self._create_timestamped_youtube_link(
                video_url, int(display_timestamp)
            )

            # Get detection-level loudness metrics if available
            detection_metrics = (
                detection_loudness_metrics[i] if detection_loudness_metrics else {}
            )

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
                youtube_timestamped_link=timestamped_link,
                video_max_confidence=f"{max_confidence:.3f}",
                detection_threshold=f"{threshold:.3f}",
                max_threshold=(
                    f"{max_threshold:.3f}" if max_threshold is not None else ""
                ),
                processing_date=processing_date,
                processing_time=processing_time,
                # Detection-level loudness metrics
                detection_peak_dbfs=f"{detection_metrics.get('peak_dbfs', 0):.2f}",
                detection_rms_dbfs=f"{detection_metrics.get('rms_dbfs', 0):.2f}",
                detection_crest_factor=f"{detection_metrics.get('crest_factor', 0):.2f}",
                detection_likely_clipped=str(
                    detection_metrics.get("likely_clipped", False)
                ),
                detection_peak_amplitude=f"{detection_metrics.get('peak_amplitude', 0):.6f}",
                detection_rms_amplitude=f"{detection_metrics.get('rms_amplitude', 0):.6f}",
                # Video-level loudness metrics
                video_peak_dbfs=f"{video_loudness_metrics.get('peak_dbfs', 0):.2f}"
                if video_loudness_metrics
                else "",
                video_rms_dbfs=f"{video_loudness_metrics.get('rms_dbfs', 0):.2f}"
                if video_loudness_metrics
                else "",
                video_crest_factor=f"{video_loudness_metrics.get('crest_factor', 0):.2f}"
                if video_loudness_metrics
                else "",
                video_likely_clipped=str(
                    video_loudness_metrics.get("likely_clipped", False)
                )
                if video_loudness_metrics
                else "",
                video_peak_amplitude=f"{video_loudness_metrics.get('peak_amplitude', 0):.6f}"
                if video_loudness_metrics
                else "",
                video_rms_amplitude=f"{video_loudness_metrics.get('rms_amplitude', 0):.6f}"
                if video_loudness_metrics
                else "",
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

        # Generate filename with readable date and time
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

        # Calculate loudness statistics
        loudness_stats = {}
        if any(record.detection_peak_dbfs for record in self.detection_records):
            peak_dbfs_values = [
                float(record.detection_peak_dbfs)
                for record in self.detection_records
                if record.detection_peak_dbfs
            ]
            rms_dbfs_values = [
                float(record.detection_rms_dbfs)
                for record in self.detection_records
                if record.detection_rms_dbfs
            ]
            crest_factor_values = [
                float(record.detection_crest_factor)
                for record in self.detection_records
                if record.detection_crest_factor
            ]

            if peak_dbfs_values:
                loudness_stats.update(
                    {
                        "avg_detection_peak_dbfs": round(
                            sum(peak_dbfs_values) / len(peak_dbfs_values), 2
                        ),
                        "max_detection_peak_dbfs": round(max(peak_dbfs_values), 2),
                        "min_detection_peak_dbfs": round(min(peak_dbfs_values), 2),
                    }
                )

            if rms_dbfs_values:
                loudness_stats.update(
                    {
                        "avg_detection_rms_dbfs": round(
                            sum(rms_dbfs_values) / len(rms_dbfs_values), 2
                        ),
                        "max_detection_rms_dbfs": round(max(rms_dbfs_values), 2),
                        "min_detection_rms_dbfs": round(min(rms_dbfs_values), 2),
                    }
                )

            if crest_factor_values:
                loudness_stats.update(
                    {
                        "avg_detection_crest_factor": round(
                            sum(crest_factor_values) / len(crest_factor_values), 2
                        ),
                        "max_detection_crest_factor": round(
                            max(crest_factor_values), 2
                        ),
                        "min_detection_crest_factor": round(
                            min(crest_factor_values), 2
                        ),
                    }
                )

        return {
            "total_detections": total_detections,
            "unique_videos": unique_videos,
            "average_confidence": round(avg_confidence, 4),
            "max_confidence": round(max_confidence, 4),
            "min_confidence": round(min_confidence, 4),
            **loudness_stats,
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

    def _create_timestamped_youtube_link(
        self, video_url: str, timestamp_seconds: int
    ) -> str:
        """Create a YouTube URL with timestamp parameter.

        Args:
            video_url: Original YouTube URL
            timestamp_seconds: Timestamp in seconds (no decimals)

        Returns:
            YouTube URL with timestamp parameter
        """
        # Handle different YouTube URL formats
        if "youtube.com/watch?v=" in video_url:
            # Standard YouTube URL
            if "&t=" in video_url or "?t=" in video_url:
                # URL already has timestamp, replace it
                base_url = video_url.split("&t=")[0].split("?t=")[0]
                return f"{base_url}&t={timestamp_seconds}"
            else:
                # Add timestamp parameter
                separator = "&" if "?" in video_url else "?"
                return f"{video_url}{separator}t={timestamp_seconds}"
        elif "youtu.be/" in video_url:
            # Short YouTube URL
            video_id = video_url.split("youtu.be/")[1].split("?")[0]
            return f"https://www.youtube.com/watch?v={video_id}&t={timestamp_seconds}"
        else:
            # Unknown format, return original URL
            return video_url

    def clear_records(self) -> None:
        """Clear all collected detection records."""
        self.detection_records.clear()
