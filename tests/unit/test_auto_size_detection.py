"""
Tests for automatic model size detection feature.

Tests verify that:
- Size can be auto-detected for all model types (v8, v9, v11, X)
- Explicit size parameter still works (backward compatibility)
- Error handling works correctly
- Size attribute is set correctly on model instances
"""

import pytest
import torch
from pathlib import Path
from libreyolo.factory import (
    LIBREYOLO,
    detect_yolo8_11_size,
    detect_yolo9_size,
    detect_yolox_size,
    _unwrap_state_dict,
)


class TestUnwrapStateDict:
    """Test state dict unwrapping helper."""

    def test_unwrap_plain_dict(self):
        """Test unwrapping plain state dict."""
        state_dict = {'key1': 'value1', 'key2': 'value2'}
        result = _unwrap_state_dict(state_dict)
        assert result == state_dict

    def test_unwrap_ema_with_module(self):
        """Test unwrapping EMA checkpoint with module."""
        weights = {'key1': 'value1'}
        state_dict = {'ema': {'module': weights}}
        result = _unwrap_state_dict(state_dict)
        assert result == weights

    def test_unwrap_ema_without_module(self):
        """Test unwrapping EMA checkpoint without module."""
        weights = {'key1': 'value1'}
        state_dict = {'ema': weights}
        result = _unwrap_state_dict(state_dict)
        assert result == weights

    def test_unwrap_model_wrapper(self):
        """Test unwrapping model wrapper."""
        weights = {'key1': 'value1'}
        state_dict = {'model': weights}
        result = _unwrap_state_dict(state_dict)
        assert result == weights


class TestYOLO8_11SizeDetection:
    """Test YOLOv8/v11 size detection."""

    def test_detect_nano(self):
        """Test detection of nano model (16 channels)."""
        state_dict = {
            'backbone.p1.cnn.weight': torch.zeros(16, 3, 3, 3)
        }
        assert detect_yolo8_11_size(state_dict) == 'n'

    def test_detect_small(self):
        """Test detection of small model (32 channels)."""
        state_dict = {
            'backbone.p1.cnn.weight': torch.zeros(32, 3, 3, 3)
        }
        assert detect_yolo8_11_size(state_dict) == 's'

    def test_detect_medium(self):
        """Test detection of medium model (48 channels)."""
        state_dict = {
            'backbone.p1.cnn.weight': torch.zeros(48, 3, 3, 3)
        }
        assert detect_yolo8_11_size(state_dict) == 'm'

    def test_detect_large(self):
        """Test detection of large model (64 channels)."""
        state_dict = {
            'backbone.p1.cnn.weight': torch.zeros(64, 3, 3, 3)
        }
        assert detect_yolo8_11_size(state_dict) == 'l'

    def test_detect_xlarge(self):
        """Test detection of xlarge model (80 channels)."""
        state_dict = {
            'backbone.p1.cnn.weight': torch.zeros(80, 3, 3, 3)
        }
        assert detect_yolo8_11_size(state_dict) == 'x'

    def test_detect_missing_key(self):
        """Test detection fails gracefully when key is missing."""
        state_dict = {'other.key': torch.zeros(16, 3, 3, 3)}
        assert detect_yolo8_11_size(state_dict) is None

    def test_detect_unknown_channels(self):
        """Test detection returns None for unknown channel count."""
        state_dict = {
            'backbone.p1.cnn.weight': torch.zeros(99, 3, 3, 3)
        }
        assert detect_yolo8_11_size(state_dict) is None


class TestYOLO9SizeDetection:
    """Test YOLOv9 size detection."""

    def test_detect_tiny(self):
        """Test detection of tiny model (16 channels)."""
        state_dict = {
            'backbone.conv0.conv.weight': torch.zeros(16, 3, 3, 3)
        }
        assert detect_yolo9_size(state_dict) == 't'

    def test_detect_small(self):
        """Test detection of small model (32 channels, 64 elan)."""
        state_dict = {
            'backbone.conv0.conv.weight': torch.zeros(32, 3, 3, 3),
            'backbone.elan1.cv1.conv.weight': torch.zeros(64, 32, 1, 1)
        }
        assert detect_yolo9_size(state_dict) == 's'

    def test_detect_medium(self):
        """Test detection of medium model (32 channels, 128 elan)."""
        state_dict = {
            'backbone.conv0.conv.weight': torch.zeros(32, 3, 3, 3),
            'backbone.elan1.cv1.conv.weight': torch.zeros(128, 32, 1, 1)
        }
        assert detect_yolo9_size(state_dict) == 'm'

    def test_detect_compact(self):
        """Test detection of compact model (64 channels)."""
        state_dict = {
            'backbone.conv0.conv.weight': torch.zeros(64, 3, 3, 3)
        }
        assert detect_yolo9_size(state_dict) == 'c'

    def test_detect_missing_key(self):
        """Test detection fails gracefully when key is missing."""
        state_dict = {'other.key': torch.zeros(16, 3, 3, 3)}
        assert detect_yolo9_size(state_dict) is None

    def test_detect_32_channels_no_secondary_key(self):
        """Test detection returns None when secondary key missing for 32-channel model."""
        state_dict = {
            'backbone.conv0.conv.weight': torch.zeros(32, 3, 3, 3)
            # Missing secondary key
        }
        assert detect_yolo9_size(state_dict) is None


class TestYOLOXSizeDetection:
    """Test YOLOX size detection."""

    def test_detect_nano(self):
        """Test detection of nano model (16 channels)."""
        state_dict = {
            'backbone.backbone.stem.conv.conv.weight': torch.zeros(16, 3, 3, 3)
        }
        assert detect_yolox_size(state_dict) == 'nano'

    def test_detect_tiny(self):
        """Test detection of tiny model (24 channels)."""
        state_dict = {
            'backbone.backbone.stem.conv.conv.weight': torch.zeros(24, 3, 3, 3)
        }
        assert detect_yolox_size(state_dict) == 'tiny'

    def test_detect_small(self):
        """Test detection of small model (32 channels)."""
        state_dict = {
            'backbone.backbone.stem.conv.conv.weight': torch.zeros(32, 3, 3, 3)
        }
        assert detect_yolox_size(state_dict) == 's'

    def test_detect_medium(self):
        """Test detection of medium model (48 channels)."""
        state_dict = {
            'backbone.backbone.stem.conv.conv.weight': torch.zeros(48, 3, 3, 3)
        }
        assert detect_yolox_size(state_dict) == 'm'

    def test_detect_large(self):
        """Test detection of large model (64 channels)."""
        state_dict = {
            'backbone.backbone.stem.conv.conv.weight': torch.zeros(64, 3, 3, 3)
        }
        assert detect_yolox_size(state_dict) == 'l'

    def test_detect_missing_key(self):
        """Test detection fails gracefully when key is missing."""
        state_dict = {'other.key': torch.zeros(16, 3, 3, 3)}
        assert detect_yolox_size(state_dict) is None


class TestAutoDetectionIntegration:
    """Integration tests for auto-detection with real model files."""

    @pytest.mark.skipif(
        not Path("libreyolo8n.pt").exists(),
        reason="Model file not available"
    )
    def test_auto_detect_yolo8n(self):
        """Test auto-detection with YOLOv8n model."""
        model = LIBREYOLO("libreyolo8n.pt")
        assert model.size == "n"
        assert model.version == "8"

    @pytest.mark.skipif(
        not Path("libreyolo11n.pt").exists(),
        reason="Model file not available"
    )
    def test_auto_detect_yolo11n(self):
        """Test auto-detection with YOLOv11n model."""
        model = LIBREYOLO("libreyolo11n.pt")
        assert model.size == "n"
        assert model.version == "11"

    @pytest.mark.skipif(
        not Path("libreyolo9t.pt").exists(),
        reason="Model file not available"
    )
    def test_auto_detect_yolo9t(self):
        """Test auto-detection with YOLOv9t model."""
        model = LIBREYOLO("libreyolo9t.pt")
        assert model.size == "t"
        assert model.version == "9"

    @pytest.mark.skipif(
        not Path("libreyoloXnano.pt").exists(),
        reason="Model file not available"
    )
    def test_auto_detect_yolox_nano(self):
        """Test auto-detection with YOLOXnano model."""
        model = LIBREYOLO("libreyoloXnano.pt")
        assert model.size == "nano"
        assert model.version == "x"


class TestBackwardCompatibility:
    """Test that explicit size parameter still works."""

    @pytest.mark.skipif(
        not Path("libreyolo8n.pt").exists(),
        reason="Model file not available"
    )
    def test_explicit_size_still_works(self):
        """Test that providing explicit size parameter still works."""
        model = LIBREYOLO("libreyolo8n.pt", size="n")
        assert model.size == "n"
        assert model.version == "8"

    @pytest.mark.skipif(
        not Path("libreyolo11s.pt").exists(),
        reason="Model file not available"
    )
    def test_explicit_size_overrides_detection(self):
        """Test that explicit size is used without auto-detection."""
        # This should not print "Auto-detected size" message
        model = LIBREYOLO("libreyolo11s.pt", size="s")
        assert model.size == "s"


class TestErrorHandling:
    """Test error handling for edge cases."""

    def test_file_not_found_without_size(self):
        """Test that missing file without size gives clear error."""
        with pytest.raises(ValueError, match="Model weights file not found"):
            LIBREYOLO("nonexistent_model.pt")

    def test_file_not_found_with_size(self):
        """Test that missing file with size attempts download."""
        # This should attempt download and fail
        with pytest.raises((FileNotFoundError, RuntimeError)):
            LIBREYOLO("definitely_nonexistent_12345.pt", size="n")

    def test_detection_failure_gives_clear_error(self):
        """Test that detection failure provides helpful error message."""
        # Create a temporary file with invalid state dict
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            # Save a state dict that will fail detection
            torch.save({'invalid_key': torch.zeros(10, 10)}, f.name)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Could not automatically detect"):
                LIBREYOLO(temp_path)
        finally:
            Path(temp_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
