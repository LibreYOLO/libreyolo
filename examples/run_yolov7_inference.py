"""
Example: YOLOv7 Inference with LibreYOLO

YOLOv7 uses anchor-based detection with implicit knowledge
through ImplicitA and ImplicitM layers. It features ELAN
(Efficient Layer Aggregation Network) blocks.

Available variants:
- yolov7 (base)
- yolov7-tiny
"""
import os
from libreyolo import LIBREYOLO


def run_yolov7_inference():
    # Path to YOLOv7 weights (will auto-download if not present)
    weights_path = "weights/yolov7.pt"
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("--- Running Inference with YOLOv7 (Base) ---")
    try:
        # Load model (auto-downloads weights if not found)
        model = LIBREYOLO(weights_path, size="base")
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
        print(f"Error during YOLOv7 inference: {e}")
        raise


def run_yolov7_tiny():
    """Run inference with YOLOv7-tiny."""
    weights_path = "weights/yolov7-tiny.pt"
    test_image = "media/test_image_1_creative_commons.jpg"

    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print("\n--- Running Inference with YOLOv7-Tiny ---")
    try:
        model = LIBREYOLO(weights_path, size="tiny")
        print(f"Model version: {model.version}")

        results = model(image=test_image, save=True)
        print(f"Detected {results['num_detections']} objects.")

        if 'saved_path' in results:
            print(f"Result saved to: {results['saved_path']}")

    except Exception as e:
        print(f"Error during YOLOv7-tiny inference: {e}")


if __name__ == "__main__":
    run_yolov7_inference()
