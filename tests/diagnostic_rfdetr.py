#!/usr/bin/env python3
"""
Diagnostic script for RF-DETR integration in LibreYOLO.

Tests:
1. Upstream rfdetr baseline (direct prediction)
2. LibreYOLO wrapper prediction
3. Resolution correctness per variant
4. Validation on COCO val2017 (nano only, for speed)
"""

import sys
import torch

COCO_IMAGE = "/tmp/datasets/coco/images/val2017/000000000139.jpg"
COCO_YAML = "libreyolo/cfg/datasets/coco-val-only.yaml"

EXPECTED_RESOLUTIONS = {
    "n": 384,
    "s": 512,
    "b": 560,
    "m": 576,
    "l": 560,
}

WEIGHTS = {
    "n": "librerfdetrnano.pth",
    "s": "librerfdetrsmall.pth",
    "b": "librerfdetrbase.pth",
    "m": "librerfdetrmedium.pth",
    "l": "librerfdetrlarge.pth",
}


def test_upstream_baseline():
    """Test that upstream rfdetr package works directly."""
    print("\n" + "=" * 60)
    print("TEST 1: Upstream rfdetr baseline")
    print("=" * 60)

    from rfdetr import RFDETRBase
    model = RFDETRBase()
    result = model.predict(COCO_IMAGE, threshold=0.3)
    n_detections = len(result)
    print(f"  Detections (threshold=0.3): {n_detections}")
    if n_detections > 0:
        print(f"  Top scores: {result.confidence[:5]}")
        print(f"  Top classes: {result.class_id[:5]}")
    assert n_detections > 0, "Upstream rfdetr produced zero detections!"
    print("  PASSED")


def test_resolution_per_variant():
    """Test that each variant reports the correct native resolution."""
    print("\n" + "=" * 60)
    print("TEST 2: Resolution correctness per variant")
    print("=" * 60)

    from libreyolo import LIBREYOLO

    all_ok = True
    for size_code, expected_res in EXPECTED_RESOLUTIONS.items():
        weight_file = WEIGHTS[size_code]
        model = LIBREYOLO(model_path=weight_file, size=size_code, device="cuda")
        actual_res = model._get_input_size()
        status = "OK" if actual_res == expected_res else "FAIL"
        if status == "FAIL":
            all_ok = False
        print(f"  {size_code} ({weight_file}): expected={expected_res}, actual={actual_res} [{status}]")
        del model
        torch.cuda.empty_cache()

    assert all_ok, "Resolution mismatch detected!"
    print("  PASSED")


def test_wrapper_predict():
    """Test LibreYOLO wrapper prediction on a COCO image."""
    print("\n" + "=" * 60)
    print("TEST 3: LibreYOLO wrapper prediction (base)")
    print("=" * 60)

    from libreyolo import LIBREYOLO

    model = LIBREYOLO(model_path="librerfdetrbase.pth", size="b", device="cuda")
    result = model.predict(COCO_IMAGE, conf=0.3)

    if hasattr(result, '__len__'):
        n_detections = len(result)
    else:
        n_detections = len(result.boxes) if hasattr(result, 'boxes') else 0

    print(f"  Detections (conf=0.3): {n_detections}")
    assert n_detections > 0, "LibreYOLO wrapper produced zero detections!"
    print("  PASSED")

    del model
    torch.cuda.empty_cache()


def test_validation_nano():
    """Test validation on COCO val2017 with rfdetr-nano."""
    print("\n" + "=" * 60)
    print("TEST 4: Validation on COCO val2017 (nano)")
    print("=" * 60)

    from libreyolo import LIBREYOLO

    model = LIBREYOLO(model_path="librerfdetrnano.pth", size="n", device="cuda")
    print(f"  Native resolution: {model._get_input_size()}")
    print("  Running validation (this takes ~1 minute)...")

    results = model.val(
        data=COCO_YAML,
        batch=16,
        conf=0.001,
        iou=0.6,
        verbose=False,
        plots=False,
    )

    map50_95 = results.get("metrics/mAP50-95", 0.0)
    map50 = results.get("metrics/mAP50", 0.0)
    precision = results.get("metrics/precision", 0.0)
    recall = results.get("metrics/recall", 0.0)

    print(f"  mAP50-95: {map50_95:.4f}")
    print(f"  mAP50:    {map50:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")

    # RF-DETR nano should get ~33 mAP50-95 on COCO
    assert map50_95 > 0.25, f"mAP50-95 too low: {map50_95:.4f} (expected > 0.25)"
    print("  PASSED")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    passed = 0
    failed = 0
    tests = [
        test_upstream_baseline,
        test_resolution_per_variant,
        test_wrapper_predict,
        test_validation_nano,
    ]

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)
