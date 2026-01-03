"""
Example: YOLOv9 Inference with LibreYOLO

YOLOv9 uses anchor-free detection with DFL (Distribution Focal Loss)
for improved bounding box regression. It features RepNCSPELAN blocks
for efficient feature extraction.

Available variants:
- yolov9t (tiny)
- yolov9s (small)
- yolov9m (medium)
- yolov9c (compact - largest)
"""
import os
from libreyolo import LIBREYOLO


def run_yolov9_inference():
    # Path to YOLOv9 weights (will auto-download if not present)
    weights_path = "weights/yolov9t.pt"
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("--- Running Inference with YOLOv9-t (Tiny) ---")
    try:
        # Load model (auto-downloads weights if not found)
        model = LIBREYOLO(weights_path, size="t")
        print(f"Model version: {model.version}")

        # Run inference
        results = model(image=test_image, save=True, conf_thres=0.25, iou_thres=0.45)

        print(f"Detected {results['num_detections']} objects.")
        if results['num_detections'] > 0:
            for i, (box, score, cls) in enumerate(zip(
                results['boxes'][:5],  # Show first 5
                results['scores'][:5],
                results['classes'][:5]
            )):
                print(f"  {i+1}. Class {cls}: {score:.2%} at {box}")

        if 'saved_path' in results:
            print(f"Result saved to: {results['saved_path']}")

    except Exception as e:
        print(f"Error during YOLOv9 inference: {e}")
        raise


def run_all_v9_variants():
    """Run inference on all YOLOv9 variants."""
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    variants = [
        ("t", "weights/yolov9t.pt"),
        ("s", "weights/yolov9s.pt"),
        ("m", "weights/yolov9m.pt"),
        ("c", "weights/yolov9c.pt"),
    ]

    for size, weights_path in variants:
        print(f"\n--- YOLOv9-{size} ---")
        try:
            model = LIBREYOLO(weights_path, size=size)
            results = model(image=test_image, save=False)
            print(f"  Detections: {results['num_detections']}")
        except FileNotFoundError:
            print(f"  Weights not found: {weights_path}")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    run_yolov9_inference()
