"""
Example: YOLO-RD Inference with LibreYOLO

YOLO-RD (Regional Diversity) extends YOLOv9-c with Dynamic Convolution (DConv)
and Position-wise Channel Normalization (PONO) for enhanced regional feature
diversity. This helps the model better distinguish objects in different
regions of the image.

Key features:
- DConv: Dynamic convolution that adapts to regional content
- PONO: Position-aware normalization for spatial diversity
- Based on YOLOv9-c architecture

Available variant:
- yolo_rd_c (based on v9-c)
"""
import os
from libreyolo import LIBREYOLO


def run_yolo_rd_inference():
    # Path to YOLO-RD weights (will auto-download if not present)
    weights_path = "weights/yolo_rd_c.pt"
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("--- Running Inference with YOLO-RD (Regional Diversity) ---")
    try:
        # Load model (auto-downloads weights if not found)
        model = LIBREYOLO(weights_path, size="c")
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
        print(f"Error during YOLO-RD inference: {e}")
        raise


def compare_rd_vs_v9():
    """Compare YOLO-RD vs YOLOv9-c on the same image."""
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("\n=== Comparing YOLO-RD vs YOLOv9-c ===\n")

    models = [
        ("YOLOv9-c", "weights/yolov9c.pt", "c"),
        ("YOLO-RD", "weights/yolo_rd_c.pt", "c"),
    ]

    for name, weights_path, size in models:
        print(f"--- {name} ---")
        try:
            model = LIBREYOLO(weights_path, size=size)
            results = model(image=test_image, save=False)
            print(f"  Detections: {results['num_detections']}")
            print(f"  Model version: {model.version}")
        except FileNotFoundError:
            print(f"  Weights not found: {weights_path}")
        except Exception as e:
            print(f"  Error: {e}")
        print()


if __name__ == "__main__":
    run_yolo_rd_inference()
