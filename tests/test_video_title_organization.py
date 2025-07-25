"""Tests for video title organization functionality."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from gong_detector.core.youtube_utils import create_folder_name_from_date, sanitize_title_for_folder


class TestVideoTitleOrganization:
    """Tests for video title organization features."""

    def test_sanitize_title_for_folder_basic(self) -> None:
        """Test basic title sanitization."""
        title = "My Test Video Title"
        result = sanitize_title_for_folder(title)
        assert result == "my_test_video_title"

    def test_sanitize_title_with_special_chars(self) -> None:
        """Test sanitization with special characters."""
        title = "Video with <bad> chars: | ? * / \\"
        result = sanitize_title_for_folder(title)
        assert result == "video_with_bad_chars"

    def test_sanitize_title_with_multiple_spaces(self) -> None:
        """Test sanitization with multiple spaces."""
        title = "Video   with    multiple     spaces"
        result = sanitize_title_for_folder(title)
        assert result == "video_with_multiple_spaces"

    def test_sanitize_title_leading_trailing_underscores(self) -> None:
        """Test removal of leading/trailing underscores."""
        title = "___Video Title___"
        result = sanitize_title_for_folder(title)
        assert result == "video_title"

    def test_sanitize_title_length_limit(self) -> None:
        """Test length limiting of folder names."""
        long_title = "A" * 150
        result = sanitize_title_for_folder(long_title)
        assert len(result) <= 100
        assert result.startswith("a")

    def test_sanitize_title_empty_string(self) -> None:
        """Test handling of empty string."""
        result = sanitize_title_for_folder("")
        assert result == ""

    def test_sanitize_title_unicode(self) -> None:
        """Test handling of unicode characters."""
        title = "Video with émojis 🎵 and accents"
        result = sanitize_title_for_folder(title)
        # Should handle unicode gracefully
        assert "video_with" in result
        assert len(result) > 0

    def test_sanitize_title_remove_commas(self) -> None:
        """Test removal of commas from titles."""
        title = "Video, with, multiple, commas"
        result = sanitize_title_for_folder(title)
        assert result == "video_with_multiple_commas"
        assert "," not in result

    def test_sanitize_title_mixed_case_and_commas(self) -> None:
        """Test combination of lowercase and comma removal."""
        title = "My Video, Title WITH Mixed Case"
        result = sanitize_title_for_folder(title)
        assert result == "my_video_title_with_mixed_case"
        assert "," not in result
        assert result.islower()


class TestCreateFolderNameFromDate:
    """Tests for date-based folder naming functionality."""

    def test_create_folder_name_from_date_valid(self) -> None:
        """Test folder name creation with valid date."""
        upload_date = "20250717"
        result = create_folder_name_from_date(upload_date)
        assert result == "tbpn_07_17_2025"

    def test_create_folder_name_from_date_different_year(self) -> None:
        """Test folder name creation with different year."""
        upload_date = "20231225"
        result = create_folder_name_from_date(upload_date)
        assert result == "tbpn_12_25_2023"

    def test_create_folder_name_from_date_empty_string(self) -> None:
        """Test folder name creation with empty string."""
        result = create_folder_name_from_date("")
        assert result == "tbpn_unknown_date"

    def test_create_folder_name_from_date_invalid_length(self) -> None:
        """Test folder name creation with invalid length."""
        result = create_folder_name_from_date("2025")
        assert result == "tbpn_unknown_date"

    def test_create_folder_name_from_date_none(self) -> None:
        """Test folder name creation with None."""
        result = create_folder_name_from_date(None)  # type: ignore
        assert result == "tbpn_unknown_date"
