"""Tests for CLI error handling."""

import pytest

from libreyolo.cli.errors import CLIError, EXIT_CODES, suggest_key

pytestmark = pytest.mark.unit


class TestCLIError:
    """Test CLIError creation and attributes."""

    def test_basic_error(self):
        err = CLIError("model_not_found", "Weights not found: yolox-s.pt")
        assert err.code == "model_not_found"
        assert err.message == "Weights not found: yolox-s.pt"
        assert err.suggestion is None
        assert err.exit_code == 4

    def test_error_with_suggestion(self):
        err = CLIError(
            "config_unknown_key",
            "Unknown key 'epoch'",
            suggestion="Did you mean 'epochs'?",
        )
        assert err.suggestion == "Did you mean 'epochs'?"
        assert err.exit_code == 2

    def test_unknown_code_defaults_to_exit_1(self):
        err = CLIError("some_future_code", "something went wrong")
        assert err.exit_code == 1

    def test_inherits_from_exception(self):
        err = CLIError("io_error", "disk full")
        assert isinstance(err, Exception)
        assert str(err) == "disk full"


class TestExitCodes:
    """Verify exit code categories match the spec."""

    def test_usage_errors_exit_2(self):
        for code in (
            "config_unknown_key",
            "config_type_error",
            "config_range_error",
            "config_required_key",
            "config_conflict",
        ):
            assert EXIT_CODES[code] == 2, f"{code} should exit 2"

    def test_data_errors_exit_3(self):
        for code in ("source_not_found", "data_not_found", "data_invalid", "data_images_missing"):
            assert EXIT_CODES[code] == 3, f"{code} should exit 3"

    def test_model_errors_exit_4(self):
        for code in (
            "model_not_found",
            "model_load_failed",
            "model_family_mismatch",
            "checkpoint_not_found",
        ):
            assert EXIT_CODES[code] == 4, f"{code} should exit 4"

    def test_export_errors_exit_5(self):
        for code in (
            "export_format_unknown",
            "export_dep_missing",
            "format_precision_unsupported",
        ):
            assert EXIT_CODES[code] == 5, f"{code} should exit 5"

    def test_runtime_errors_exit_1(self):
        for code in (
            "device_not_available",
            "cuda_oom",
            "training_diverged",
            "download_failed",
            "io_error",
        ):
            assert EXIT_CODES[code] == 1, f"{code} should exit 1"


class TestSuggestKey:
    """Test fuzzy key matching."""

    def test_close_match(self):
        valid = ["epochs", "batch", "imgsz", "lr0", "momentum"]
        assert suggest_key("epoch", valid) == "epochs"

    def test_close_match_batch(self):
        valid = ["epochs", "batch", "imgsz"]
        assert suggest_key("batcg", valid) == "batch"

    def test_no_match(self):
        valid = ["epochs", "batch", "imgsz"]
        assert suggest_key("zzzzz", valid) is None

    def test_exact_match(self):
        valid = ["epochs", "batch"]
        assert suggest_key("epochs", valid) == "epochs"
