import onnxruntime as ort
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_class_color(class_id):
    import colorsys
    hue = (class_id * 137.508) % 360 / 360.0
    saturation = 0.7 + (class_id % 3) * 0.1
    value = 0.8 + (class_id % 2) * 0.15
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

def preprocess(image_path, input_size=640):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h0, w0 = img.shape[:2]
    
    # Resize keeping aspect ratio
    r = min(input_size / h0, input_size / w0)
    img_resized = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    h, w = img_resized.shape[:2]
    dw, dh = input_size - w, input_size - h
    dw /= 2; dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # HWC to CHW, normalize
    blob = cv2.dnn.blobFromImage(img_padded, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
    return blob, (h0, w0), (dw, dh), r

def nms(boxes, scores, iou_threshold=0.45):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= iou_threshold)[0] + 1]
    return keep

def run_inference(onnx_path, image_path):
    print(f"Running end-to-end inference on {image_path}...")
    session = ort.InferenceSession(onnx_path)
    
    # 1. Preprocess
    INPUT_SIZE = 640
    blob, (h0, w0), (dw, dh), ratio = preprocess(image_path, INPUT_SIZE)
    
    # 2. Run Inference - Now returns [Batch, N, 84] (boxes + scores)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    
    # 3. Postprocess (only NMS and scaling)
    boxes = outputs[:, :4]
    scores = outputs[:, 4:]
    
    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    mask = max_scores > 0.25
    boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]
    
    if len(boxes) == 0:
        print("No objects detected.")
        return

    # Scale boxes back to original image
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
    
    # NMS
    keep = nms(boxes, max_scores)
    boxes, scores, classes = boxes[keep], max_scores[keep], class_ids[keep]
    
    # 4. Draw and Save
    print(f"Detected {len(boxes)} objects.")
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try: font = ImageFont.truetype("arial.ttf", 15)
    except: font = ImageFont.load_default()

    for box, score, cls_id in zip(boxes, scores, classes):
        color = get_class_color(int(cls_id))
        draw.rectangle(box.tolist(), outline=color, width=3)
        label = f"{COCO_CLASSES[int(cls_id)]}: {score:.2f}"
        draw.text((box[0], box[1] - 15), label, fill=color, font=font)
    
    img.save("onnx_result.jpg")
    print("Saved results to onnx_result.jpg")

if __name__ == "__main__":
    ONNX_MODEL = "libreyolo11n.onnx"
    TEST_IMAGE = "media/test_image_1_creative_commons.jpg"
    if not os.path.exists(ONNX_MODEL):
        print(f"Error: {ONNX_MODEL} not found. Run export_onnx.py first.")
    else:
        run_inference(ONNX_MODEL, TEST_IMAGE)
