"""
YOLOX inference examples for LibreYOLO.

This example demonstrates how to use LIBREYOLOX for object detection.
YOLOX is an anchor-free detector with a simpler design but better performance.

Available model sizes:
    - nano: 0.91M params, 416x416 input (lightweight)
    - tiny: 2.01M params, 416x416 input (lightweight)
    - s: 8.97M params, 640x640 input (small)
    - m: 25.33M params, 640x640 input (medium)
    - l: 54.21M params, 640x640 input (large)
    - x: 99.07M params, 640x640 input (extra large)
"""

import os
from libreyolo import LIBREYOLO, LIBREYOLOX


def run_basic_inference():
    """Basic YOLOX inference example."""
    weights_path = "weights/libreyoloXs.pt"
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        print("Download YOLOX weights from: https://github.com/Megvii-BaseDetection/YOLOX/releases")
        return
    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("--- Running Basic YOLOX-S Inference ---")
    try:
        # Option 1: Use unified factory (auto-detects YOLOX from weights)
        model = LIBREYOLO(weights_path, size="s")

        # Option 2: Use LIBREYOLOX directly
        # model = LIBREYOLOX(weights_path, size="s")

        # Run inference
        results = model(image=test_image, save=True)

        print(f"Detected {results['num_detections']} objects.")
        for i in range(min(5, results['num_detections'])):
            box = results['boxes'][i]
            score = results['scores'][i]
            cls = results['classes'][i]
            print(f"  - Class {cls}: {score:.2f} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

        if 'saved_path' in results:
            print(f"Result saved to: {results['saved_path']}")

    except Exception as e:
        print(f"Error during YOLOX inference: {e}")
        raise


def run_inference_with_options():
    """YOLOX inference with custom thresholds."""
    weights_path = "weights/libreyoloXs.pt"
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(weights_path):
        print(f"Skipping: {weights_path} not found.")
        return
    if not os.path.exists(test_image):
        print(f"Skipping: {test_image} not found.")
        return

    print("\n--- Running YOLOX with Custom Thresholds ---")
    try:
        model = LIBREYOLOX(weights_path, size="s")

        # Run with custom confidence and IoU thresholds
        results = model(
            image=test_image,
            conf_thres=0.5,   # Higher confidence threshold
            iou_thres=0.4,    # Lower IoU threshold for NMS
            save=True
        )

        print(f"Detected {results['num_detections']} objects (conf > 0.5).")

    except Exception as e:
        print(f"Error: {e}")


def run_tiled_inference():
    """YOLOX tiled inference for large images."""
    weights_path = "weights/libreyoloXs.pt"
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(weights_path):
        print(f"Skipping: {weights_path} not found.")
        return
    if not os.path.exists(test_image):
        print(f"Skipping: {test_image} not found.")
        return

    print("\n--- Running YOLOX with Tiled Inference ---")
    try:
        # Enable tiling for large/high-resolution images
        model = LIBREYOLOX(weights_path, size="s", tiling=True)

        results = model(image=test_image, save=True)

        print(f"Detected {results['num_detections']} objects.")
        if results.get('tiled'):
            print(f"Image was processed in {results['num_tiles']} tiles.")

    except Exception as e:
        print(f"Error: {e}")


def run_directory_inference():
    """YOLOX inference on a directory of images."""
    weights_path = "weights/libreyoloXs.pt"
    test_dir = "media/"

    if not os.path.exists(weights_path):
        print(f"Skipping: {weights_path} not found.")
        return
    if not os.path.exists(test_dir):
        print(f"Skipping: {test_dir} not found.")
        return

    print("\n--- Running YOLOX on Directory ---")
    try:
        model = LIBREYOLOX(weights_path, size="s")

        # Process all images in directory
        results = model(image=test_dir, save=True)

        print(f"Processed {len(results)} images.")
        for i, result in enumerate(results):
            print(f"  Image {i+1}: {result['num_detections']} detections")

    except Exception as e:
        print(f"Error: {e}")


def compare_model_sizes():
    """Compare different YOLOX model sizes."""
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("\n--- Comparing YOLOX Model Sizes ---")

    # Model sizes and their weight files
    models = [
        ("nano", "weights/libreyoloXnano.pt"),
        ("tiny", "weights/libreyoloXtiny.pt"),
        ("s", "weights/libreyoloXs.pt"),
        ("m", "weights/libreyoloXm.pt"),
        ("l", "weights/libreyoloXl.pt"),
        ("x", "weights/libreyoloXx.pt"),
    ]

    for size, weights_path in models:
        if not os.path.exists(weights_path):
            print(f"  {size}: weights not found at {weights_path}")
            continue

        try:
            model = LIBREYOLOX(weights_path, size=size)
            results = model(image=test_image, conf_thres=0.25)
            print(f"  {size}: {results['num_detections']} detections")
        except Exception as e:
            print(f"  {size}: Error - {e}")


def demo_without_weights():
    """Demonstrate model architecture without actual weights."""
    print("\n--- YOLOX Architecture Demo (No Weights Required) ---")

    try:
        from libreyolo.yolox.nn import YOLOXModel
        import torch

        # Create models of different sizes
        for size in ['nano', 'tiny', 's', 'm', 'l', 'x']:
            model = YOLOXModel(config=size, nb_classes=80)
            params = sum(p.numel() for p in model.parameters())
            input_size = 416 if size in ['nano', 'tiny'] else 640

            # Test forward pass
            x = torch.randn(1, 3, input_size, input_size)
            with torch.no_grad():
                outputs = model(x)

            total_anchors = sum(out.shape[2] * out.shape[3] for out in outputs)
            print(f"  YOLOX-{size}: {params/1e6:.2f}M params, {input_size}x{input_size} input, {total_anchors} anchors")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Run demo that doesn't require weights
    demo_without_weights()

    # Run examples that require weights
    run_basic_inference()
    run_inference_with_options()
    run_tiled_inference()
    run_directory_inference()
    compare_model_sizes()
