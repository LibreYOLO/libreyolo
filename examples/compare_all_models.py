"""
Example: Compare All LibreYOLO Model Versions

This script demonstrates how to run inference with all supported YOLO
model versions and compare their detection results.

Supported model versions:
- YOLOv7: Anchor-based detection with implicit knowledge
- YOLOv8: Anchor-free detection with C2f blocks
- YOLOv9: Anchor-free detection with RepNCSPELAN and DFL
- YOLOv11: Latest anchor-free with C2PSA attention
- YOLOX: Anchor-free with decoupled head
- YOLO-RD: YOLOv9-c + Regional Diversity (DConv/PONO)
"""
import os
import time
from libreyolo import LIBREYOLO


# Define all available models with their configurations
ALL_MODELS = [
    # (display_name, weights_path, size)
    ("YOLOv7-tiny", "weights/yolov7-tiny.pt", "tiny"),
    ("YOLOv7", "weights/yolov7.pt", "base"),
    ("YOLOv8n", "weights/libreyolo8n.pt", "n"),
    ("YOLOv9t", "weights/yolov9t.pt", "t"),
    ("YOLOv9c", "weights/yolov9c.pt", "c"),
    ("YOLOv11n", "weights/libreyolo11n.pt", "n"),
    ("YOLO-RD", "weights/yolo_rd_c.pt", "c"),
]


def compare_all_models():
    """Run inference with all available models and compare results."""
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("=" * 60)
    print("LibreYOLO Model Comparison")
    print("=" * 60)
    print(f"\nTest image: {test_image}\n")

    results_table = []

    for name, weights_path, size in ALL_MODELS:
        print(f"Loading {name}...", end=" ", flush=True)
        try:
            start_time = time.time()
            model = LIBREYOLO(weights_path, size=size)
            load_time = time.time() - start_time

            start_time = time.time()
            results = model(image=test_image, save=False, conf_thres=0.25)
            inference_time = time.time() - start_time

            results_table.append({
                'name': name,
                'version': model.version,
                'detections': results['num_detections'],
                'load_time': load_time,
                'inference_time': inference_time,
            })
            print(f"OK ({results['num_detections']} detections)")

        except FileNotFoundError:
            print(f"SKIP (weights not found)")
        except Exception as e:
            print(f"ERROR: {e}")

    # Print summary table
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"{'Model':<15} {'Version':<8} {'Detections':>10} {'Load (s)':>10} {'Infer (s)':>10}")
    print("-" * 60)

    for r in results_table:
        print(f"{r['name']:<15} {r['version']:<8} {r['detections']:>10} "
              f"{r['load_time']:>10.3f} {r['inference_time']:>10.3f}")

    print("-" * 60)


def quick_comparison():
    """Quick comparison with just the smallest models."""
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    # Smallest/fastest models from each version
    quick_models = [
        ("YOLOv7-tiny", "weights/yolov7-tiny.pt", "tiny"),
        ("YOLOv8n", "weights/libreyolo8n.pt", "n"),
        ("YOLOv9t", "weights/yolov9t.pt", "t"),
        ("YOLOv11n", "weights/libreyolo11n.pt", "n"),
    ]

    print("Quick Model Comparison (smallest variants)")
    print("-" * 40)

    for name, weights_path, size in quick_models:
        try:
            model = LIBREYOLO(weights_path, size=size)
            results = model(image=test_image, save=False)
            print(f"{name}: {results['num_detections']} detections")
        except FileNotFoundError:
            print(f"{name}: weights not found")
        except Exception as e:
            print(f"{name}: error - {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_comparison()
    else:
        compare_all_models()
