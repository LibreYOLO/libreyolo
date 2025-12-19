"""
Example usage of Libre YOLO8

Note: You must provide your own model weights file.
Convert weights using: weights/convert_yolo8_weights.py
"""

from libreyolo import LIBREYOLO8

# Initialize model with model_path and size
# model = LIBREYOLO8(model_path="path/to/your/weights.pt", size="x")

# Run inference on an image
# detections = model(image="path/to/image.jpg", save=True)

# The save=True parameter saves the image with detections
# Results are returned as a dictionary with:
# - boxes: List of bounding boxes in xyxy format
# - scores: List of confidence scores
# - classes: List of class IDs
# - num_detections: Number of detections
# - saved_path: Path to saved image (if save=True)
# print(detections)

print("Libre YOLO8 package is ready to use!")
print("Example: model = LIBREYOLO8(model_path='weights.pt', size='x')")

