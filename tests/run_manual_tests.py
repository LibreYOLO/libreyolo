#!/usr/bin/env python3
"""
Manual test runner for RF-DETR integration tests.
Runs tests without pytest to verify basic functionality.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set protobuf workaround
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = []

    def test(self, name, func):
        """Run a single test function."""
        try:
            func()
            print(f"✓ {name}")
            self.passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            self.failed += 1
            self.errors.append((name, str(e)))
        except ImportError as e:
            print(f"⊘ {name}: SKIPPED - {e}")
            self.skipped += 1
        except Exception as e:
            print(f"✗ {name}: ERROR - {e}")
            self.failed += 1
            self.errors.append((name, str(e)))

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*70}")
        print(f"Test Results: {self.passed}/{total} passed")
        print(f"  Passed: {self.passed}")
        print(f"  Failed: {self.failed}")
        print(f"  Skipped: {self.skipped}")
        print(f"{'='*70}")

        if self.errors:
            print("\nFailed Tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")

        return self.failed == 0


def test_main_imports():
    """Test that main LibreYOLO imports work."""
    from libreyolo import LIBREYOLO, LIBREYOLO8, LIBREYOLO9, LIBREYOLO11, LIBREYOLOX
    assert LIBREYOLO is not None
    assert LIBREYOLO8 is not None
    assert LIBREYOLO9 is not None
    assert LIBREYOLO11 is not None
    assert LIBREYOLOX is not None


def test_rfdetr_lazy_import():
    """Test that LIBREYOLORFDETR can be imported via lazy mechanism."""
    from libreyolo import LIBREYOLORFDETR
    assert LIBREYOLORFDETR is not None
    assert LIBREYOLORFDETR.__name__ == "LIBREYOLORFDETR"


def test_rfdetr_direct_import():
    """Test that direct import from rfdetr module works."""
    from libreyolo.rfdetr import LIBREYOLORFDETR
    assert LIBREYOLORFDETR is not None


def test_rfdetr_valid_sizes():
    """Test that RF-DETR has correct valid sizes."""
    from libreyolo.rfdetr import LIBREYOLORFDETR
    model = LIBREYOLORFDETR(model_path={}, size="b")
    sizes = model._get_valid_sizes()
    assert sizes == ["n", "s", "b", "m", "l"]


def test_rfdetr_model_name():
    """Test that RF-DETR model name is correct."""
    from libreyolo.rfdetr import LIBREYOLORFDETR
    model = LIBREYOLORFDETR(model_path={}, size="b")
    assert model._get_model_name() == "LIBREYOLORFDETR"


def test_rfdetr_input_sizes():
    """Test that RF-DETR input sizes are correct."""
    from libreyolo.rfdetr import LIBREYOLORFDETR

    # Actual resolutions from RF-DETR configs
    sizes_resolutions = [
        ("n", 384),   # Nano
        ("s", 512),   # Small
        ("b", 560),   # Base
        ("m", 576),   # Medium
        ("l", 704),   # Large (DINOv2-base has larger resolution)
    ]

    for size, expected_res in sizes_resolutions:
        model = LIBREYOLORFDETR(model_path={}, size=size)
        actual_res = model._get_input_size()
        assert actual_res == expected_res, f"{size}: expected {expected_res}, got {actual_res}"


def test_rfdetr_configs():
    """Test that RFDETR_CONFIGS has all sizes."""
    from libreyolo.rfdetr.nn import RFDETR_CONFIGS

    assert 'n' in RFDETR_CONFIGS
    assert 's' in RFDETR_CONFIGS
    assert 'b' in RFDETR_CONFIGS
    assert 'm' in RFDETR_CONFIGS
    assert 'l' in RFDETR_CONFIGS


def test_rfdetr_trainers():
    """Test that RFDETR_TRAINERS has all sizes."""
    from libreyolo.rfdetr.train import RFDETR_TRAINERS

    assert 'n' in RFDETR_TRAINERS
    assert 's' in RFDETR_TRAINERS
    assert 'b' in RFDETR_TRAINERS
    assert 'm' in RFDETR_TRAINERS
    assert 'l' in RFDETR_TRAINERS


def test_box_conversion():
    """Test box format conversion."""
    import torch
    from libreyolo.rfdetr.utils import box_cxcywh_to_xyxy

    # Center (0.5, 0.5), size (0.2, 0.2)
    boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    xyxy = box_cxcywh_to_xyxy(boxes)
    expected = torch.tensor([[0.4, 0.4, 0.6, 0.6]])

    assert torch.allclose(xyxy, expected), f"Expected {expected}, got {xyxy}"


def test_postprocess_output_format():
    """Test that postprocess returns correct format."""
    import torch
    from libreyolo.rfdetr.utils import postprocess

    batch_size = 2
    num_queries = 300
    num_classes = 80

    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries, num_classes),
        'pred_boxes': torch.rand(batch_size, num_queries, 4),
    }
    target_sizes = torch.tensor([[480, 640], [720, 1280]])

    results = postprocess(outputs, target_sizes, num_select=100)

    assert len(results) == batch_size
    for result in results:
        assert 'scores' in result
        assert 'labels' in result
        assert 'boxes' in result
        assert result['scores'].shape[0] == 100
        assert result['labels'].shape[0] == 100
        assert result['boxes'].shape == (100, 4)


def test_factory_detect_rfdetr_size():
    """Test that detect_rfdetr_size function exists."""
    from libreyolo.factory import detect_rfdetr_size

    keys = ['model.weight', 'model.bias']
    size = detect_rfdetr_size(keys)
    assert size in ['n', 's', 'b', 'm', 'l']


def test_factory_rfdetr_detection():
    """Test that DETR keys trigger RF-DETR detection."""
    keys_lower = [
        'query_embed.weight',
        'class_embed.weight',
        'bbox_embed.layers.0.weight',
    ]

    is_rfdetr = any(
        'detr' in k or 'dinov2' in k or 'transformer' in k or
        'query_embed' in k or 'class_embed' in k or 'bbox_embed' in k
        for k in keys_lower
    )

    assert is_rfdetr


def test_validation_config_import():
    """Test that ValidationConfig still works."""
    from libreyolo import ValidationConfig

    config = ValidationConfig(
        data="dummy.yaml",
        batch_size=8,
        imgsz=640,
    )
    assert config.data == "dummy.yaml"
    assert config.batch_size == 8


def test_all_exports_in_init():
    """Test that all models are in __all__ export."""
    import libreyolo

    assert 'LIBREYOLO' in libreyolo.__all__
    assert 'LIBREYOLO8' in libreyolo.__all__
    assert 'LIBREYOLO9' in libreyolo.__all__
    assert 'LIBREYOLO11' in libreyolo.__all__
    assert 'LIBREYOLOX' in libreyolo.__all__
    assert 'LIBREYOLORFDETR' in libreyolo.__all__


def main():
    """Run all tests."""
    runner = TestRunner()

    print("="*70)
    print("RF-DETR Integration Test Suite")
    print("="*70)

    print("\n--- Main Import Tests ---")
    runner.test("test_main_imports", test_main_imports)
    runner.test("test_all_exports_in_init", test_all_exports_in_init)

    print("\n--- RF-DETR Import Tests ---")
    runner.test("test_rfdetr_lazy_import", test_rfdetr_lazy_import)
    runner.test("test_rfdetr_direct_import", test_rfdetr_direct_import)

    print("\n--- RF-DETR Model Tests ---")
    runner.test("test_rfdetr_valid_sizes", test_rfdetr_valid_sizes)
    runner.test("test_rfdetr_model_name", test_rfdetr_model_name)
    runner.test("test_rfdetr_input_sizes", test_rfdetr_input_sizes)
    runner.test("test_rfdetr_configs", test_rfdetr_configs)
    runner.test("test_rfdetr_trainers", test_rfdetr_trainers)

    print("\n--- RF-DETR Utilities Tests ---")
    runner.test("test_box_conversion", test_box_conversion)
    runner.test("test_postprocess_output_format", test_postprocess_output_format)

    print("\n--- Factory Integration Tests ---")
    runner.test("test_factory_detect_rfdetr_size", test_factory_detect_rfdetr_size)
    runner.test("test_factory_rfdetr_detection", test_factory_rfdetr_detection)

    print("\n--- Regression Tests ---")
    runner.test("test_validation_config_import", test_validation_config_import)

    runner.summary()

    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
