from libreyolo import LIBREYOLO8

# Initialize model
model = LIBREYOLO8(model_path="weights/libreyolo8n.pt", size="n")

# Run inference
detections = model(image="media/test_image_1_creative_commons.jpg", save=True)

print(f"Found {detections['num_detections']} detections")
print(f"Saved to: {detections.get('saved_path', 'N/A')}")

