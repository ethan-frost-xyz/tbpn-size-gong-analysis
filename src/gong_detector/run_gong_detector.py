#!/usr/bin/env python3
"""Master loader script for Gong Detector tools.

Interactive menu system for accessing all gong detector functionality.
"""

import os
import sys
from pathlib import Path
from typing import Callable, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from local modules
from gong_detector.core.detector.yamnet_runner import YAMNetGongDetector
from gong_detector.core.pipeline.bulk_processor import main as bulk_processor_main
from gong_detector.core.pipeline.detection_pipeline import (
    detect_from_youtube_comprehensive,
)
from gong_detector.core.training.manual_collector import process_single_sample
from gong_detector.core.training.negative_collector import collect_negative_samples
from gong_detector.core.utils.convert_audio import convert_youtube_audio


class MenuItem:
    """Represents a menu item with title and action."""

    def __init__(self, title: str, action: Callable[[], None]):
        """Initialize menu item with title and action."""
        self.title = title
        self.action = action


class InteractiveMenu:
    """Interactive menu system with arrow key navigation."""

    def __init__(self, title: str, items: list[MenuItem]):
        """Initialize interactive menu with title and items."""
        self.title = title
        self.items = items
        self.selected_index = 0

    def display(self) -> None:
        """Display the menu."""
        os.system("clear" if os.name == "posix" else "cls")
        print(f"\n{'=' * 50}")
        print(f"  {self.title}")
        print(f"{'=' * 50}\n")

        for i, item in enumerate(self.items):
            if i == self.selected_index:
                print(f"  ▶ {item.title}")
            else:
                print(f"    {item.title}")
        print(f"\n{'=' * 50}")
        print("  ↑/↓: Navigate  Enter: Select  q: Quit")

    def run(self) -> None:
        """Run the interactive menu."""
        import termios
        import tty

        def get_key() -> str:
            """Get a single keypress."""
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    ch = sys.stdin.read(2)
                    if ch == "[A":
                        return "UP"
                    elif ch == "[B":
                        return "DOWN"
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        while True:
            self.display()
            key = get_key()

            if key == "q":
                print("\nGoodbye!")
                break
            elif key == "UP":
                self.selected_index = (self.selected_index - 1) % len(self.items)
            elif key == "DOWN":
                self.selected_index = (self.selected_index + 1) % len(self.items)
            elif key == "\r":  # Enter key
                selected_item = self.items[self.selected_index]
                print(f"\nExecuting: {selected_item.title}")
                try:
                    selected_item.action()
                except KeyboardInterrupt:
                    print("\nOperation cancelled.")
                except Exception as e:
                    print(f"\nError: {e}")

                input("\nPress Enter to continue...")


def get_user_input(prompt: str, default: Optional[str] = None) -> str:
    """Get user input with optional default value."""
    if default:
        user_input = input(f"{prompt} (default: {default}): ").strip()
        return user_input if user_input else default
    else:
        return input(f"{prompt}: ").strip()


def get_float_input(prompt: str, default: Optional[float] = None) -> float:
    """Get float input with validation."""
    while True:
        try:
            user_input = get_user_input(prompt, str(default) if default else None)
            return float(user_input)
        except ValueError:
            print("Please enter a valid number.")


def get_int_input(prompt: str, default: Optional[int] = None) -> int:
    """Get integer input with validation."""
    while True:
        try:
            user_input = get_user_input(prompt, str(default) if default else None)
            return int(user_input)
        except ValueError:
            print("Please enter a valid integer.")


def get_yes_no_input(prompt: str, default: bool = False) -> bool:
    """Get yes/no input."""
    default_str = "y" if default else "n"
    while True:
        user_input = get_user_input(f"{prompt} (y/n)", default_str).lower()
        if user_input in ["y", "yes"]:
            return True
        elif user_input in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'.")


def single_video_detection() -> None:
    """Run single video gong detection."""
    print("\n=== Single Video Gong Detection ===\n")

    # Get parameters
    youtube_url = get_user_input("Enter YouTube URL")
    threshold = get_float_input("Confidence threshold", 0.94)
    use_version_one = get_yes_no_input("Use trained classifier (version one)?", False)
    batch_size = get_int_input("Batch size", 2000)
    should_save_samples = get_yes_no_input("Save positive samples?", False)
    keep_audio = get_yes_no_input("Keep temporary audio files?", False)

    # Optional trimming
    start_time = get_user_input("Start time in seconds (optional, press Enter to skip)")
    duration = get_user_input("Duration in seconds (optional, press Enter to skip)")

    # Convert to appropriate types
    start_time_int = int(start_time) if start_time else None
    duration_int = int(duration) if duration else None

    print(f"\nProcessing: {youtube_url}")
    print(f"Threshold: {threshold}")
    print(f"Using trained classifier: {use_version_one}")
    print(f"Batch size: {batch_size}")

    # Run detection
    result = detect_from_youtube_comprehensive(
        youtube_url=youtube_url,
        threshold=threshold,
        start_time=start_time_int,
        duration=duration_int,
        should_save_positive_samples=should_save_samples,
        keep_audio=keep_audio,
        use_version_one=use_version_one,
        batch_size=batch_size,
    )

    print("\nDetection complete!")
    print(f"Total duration: {result.get('total_duration', 0):.2f} seconds")
    print(f"Detections found: {len(result.get('detections', []))}")


def bulk_processing() -> None:
    """Run bulk processing of multiple videos."""
    print("\n=== Bulk Processing ===\n")

    # Check if links file exists
    links_file = Path("data/tbpn_ytlinks/tbpn_youtube_links.txt")

    # If not found, try other available files in data/tbpn_ytlinks/
    if not links_file.exists():
        data_dir = Path("data/tbpn_ytlinks")
        if data_dir.exists():
            # Look for any .txt files (excluding .gitkeep)
            txt_files = [f for f in data_dir.glob("*.txt") if f.name != ".gitkeep"]
            if txt_files:
                # Use the first available txt file
                links_file = txt_files[0]
                print(f"Using links file: {links_file}")
            else:
                print(f"Error: No YouTube links files found in {data_dir}")
                print("Please add a .txt file with YouTube URLs (one per line)")
                return
        else:
            # Final fallback to old location
            links_file = Path("src/gong_detector/core/data/tbpn_youtube_links.txt")
            if not links_file.exists():
                print(f"Error: Links file not found at {links_file}")
                print("Please create the file with YouTube URLs (one per line)")
                return

    # Get parameters
    threshold = get_float_input("Confidence threshold", 0.94)
    use_version_one = get_yes_no_input("Use trained classifier (version one)?", False)
    should_save_samples = get_yes_no_input("Save positive samples?", False)
    save_csv = get_yes_no_input("Save results to CSV file?", False)

    print(f"\nProcessing videos from: {links_file}")
    print(f"Threshold: {threshold}")
    print(f"Using trained classifier: {use_version_one}")
    print(f"Save CSV: {save_csv}")

    # Set up sys.argv for bulk processor
    sys.argv = ["bulk_processor"]
    if threshold != 0.94:
        sys.argv.extend(["--threshold", str(threshold)])
    if use_version_one:
        sys.argv.append("--version_one")
    if should_save_samples:
        sys.argv.append("--save_positive_samples")
    if save_csv:
        sys.argv.append("--csv")

    # Run bulk processor
    bulk_processor_main()


def manual_sample_collection() -> None:
    """Run manual sample collection for training data."""
    print("\n=== Manual Sample Collection ===\n")

    youtube_url = get_user_input("Enter YouTube URL")
    timestamp = get_float_input("Enter timestamp in seconds where gong occurs")
    confidence = get_float_input("Confidence value (1.0 for manual detections)", 1.0)

    print(f"\nProcessing: {youtube_url}")
    print(f"Timestamp: {timestamp}s")
    print(f"Confidence: {confidence}")

    success = process_single_sample(
        youtube_url=youtube_url, timestamp=timestamp, confidence=confidence
    )

    if success:
        print("Sample collected successfully!")
    else:
        print("Failed to collect sample.")


def negative_sample_collection() -> None:
    """Run negative sample collection."""
    print("\n=== Negative Sample Collection ===\n")

    youtube_url = get_user_input("Enter YouTube URL")
    num_samples = get_int_input("Number of negative samples to collect", 5)
    threshold = get_float_input("Detection threshold for finding gongs to avoid", 0.4)
    max_threshold = get_float_input(
        "Maximum confidence threshold (optional, press Enter to skip)", None
    )
    keep_audio = get_yes_no_input("Keep temporary audio files?", False)

    print(f"\nProcessing: {youtube_url}")
    print(f"Number of samples: {num_samples}")
    print(f"Threshold: {threshold}")
    if max_threshold:
        print(f"Max threshold: {max_threshold}")

    result = collect_negative_samples(
        youtube_url=youtube_url,
        num_samples=num_samples,
        threshold=threshold,
        max_threshold=max_threshold,
        keep_audio=keep_audio,
    )

    print("\nCollection complete!")
    print(f"Samples collected: {len(result.get('negative_samples', []))}")


def audio_conversion() -> None:
    """Run audio conversion tools."""
    print("\n=== Audio Conversion ===\n")

    input_source = get_user_input("Enter YouTube URL or local file path")
    output_path = get_user_input("Output WAV file path", "converted_audio.wav")

    print(f"\nConverting: {input_source}")
    print(f"Output: {output_path}")

    try:
        result_path = convert_youtube_audio(input_source, output_path)
        print(f"Conversion successful! Saved to: {result_path}")
    except Exception as e:
        print(f"Conversion failed: {e}")


def model_management() -> None:
    """Manage YAMNet model."""
    print("\n=== Model Management ===\n")

    print("Loading YAMNet model...")
    try:
        detector = YAMNetGongDetector()
        detector.load_model()
        print("✓ Model loaded successfully!")

        # Show model info
        perf_info = detector.get_performance_info()
        print("✓ Performance configuration:")
        print(f"  - Inter-op threads: {perf_info['tensorflow_threads']['inter_op']}")
        print(f"  - Intra-op threads: {perf_info['tensorflow_threads']['intra_op']}")
        print(f"  - Batch size: {perf_info['batch_size']}")

        # Try to load trained classifier
        try:
            detector.load_trained_classifier()
            print("✓ Trained classifier loaded successfully!")
        except Exception as e:
            print(f"⚠ Trained classifier not available: {e}")

    except Exception as e:
        print(f"✗ Model loading failed: {e}")


def main() -> None:
    """Run the interactive menu."""
    # Create menu items
    menu_items = [
        MenuItem("Single Video Detection", single_video_detection),
        MenuItem("Bulk Processing", bulk_processing),
        MenuItem("Manual Sample Collection", manual_sample_collection),
        MenuItem("Negative Sample Collection", negative_sample_collection),
        MenuItem("Audio Conversion", audio_conversion),
        MenuItem("Model Management", model_management),
    ]

    # Create and run menu
    menu = InteractiveMenu("GONG DETECTOR MASTER MENU", menu_items)
    menu.run()


if __name__ == "__main__":
    main()
