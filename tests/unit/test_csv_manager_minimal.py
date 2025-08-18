from pathlib import Path

from gong_detector.core.data.csv_manager import CSVManager


def test_csv_manager_basic_flow(tmp_path):
    output_dir = tmp_path / "csv_results"
    manager = CSVManager(csv_results_dir=str(output_dir))

    detections = [(5.0, 0.97, 12.0)]
    video_loudness_metrics = {
        "peak_dbfs": -1.23,
        "rms_dbfs": -10.5,
        "crest_factor": 4.2,
        "likely_clipped": False,
        "peak_amplitude": 0.98,
        "rms_amplitude": 0.12,
    }
    detection_loudness_metrics = [
        {
            "peak_dbfs": -2.34,
            "rms_dbfs": -12.3,
            "crest_factor": 3.9,
            "likely_clipped": False,
            "peak_amplitude": 0.88,
            "rms_amplitude": 0.11,
        }
    ]

    manager.add_video_detections(
        video_url="https://youtube.com/watch?v=dummy",
        video_title="Dummy Video",
        upload_date="20250101",
        video_duration=60.0,
        max_confidence=0.975,
        threshold=0.94,
        max_threshold=None,
        detections=detections,
        video_loudness_metrics=video_loudness_metrics,
        detection_loudness_metrics=detection_loudness_metrics,
    )

    csv_path = manager.save_comprehensive_csv(run_name="unit")
    assert Path(csv_path).exists()

    # Spot-check that the CSV has header + one row
    content = Path(csv_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(content) >= 2

    # Summary stats should be populated
    stats = manager.get_summary_stats()
    assert stats.get("total_detections") == 1
    assert "average_confidence" in stats
