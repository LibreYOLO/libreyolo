"""
Example usage of Libre YOLO8
"""

from libreyolo import LIBREYOLO8

# Initialize model with size
model = LIBREYOLO8(size="x")

# Or without size
# model = LIBREYOLO8()

# Run inference on an image (dummy implementation)
# detections = model(image="path/to/image.jpg", save=True)

# The save=True parameter saves the image with detections
# Results are returned as a dictionary
# print(detections)

print("Libre YOLO8 package is ready to use!")
print(f"Model initialized with size: {model.size}")

