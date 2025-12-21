import torch
from libreyolo import LIBREYOLO
import os

def export_example():
    # Example for YOLO11
    print("Exporting LibreYOLO11n...")
    try:
        model11 = LIBREYOLO("weights/libreyolo11n.pt", size="n")
        onnx_path11 = model11.export("libreyolo11n.onnx")
        print(f"LibreYOLO11n exported to: {onnx_path11}")
    except Exception as e:
        print(f"Error exporting LibreYOLO11n: {e}")

    # Example for YOLO8
    print("\nExporting LibreYOLO8n...")
    try:
        model8 = LIBREYOLO("weights/libreyolo8n.pt", size="n")
        onnx_path8 = model8.export("libreyolo8n.onnx")
        print(f"LibreYOLO8n exported to: {onnx_path8}")
    except Exception as e:
        print(f"Error exporting LibreYOLO8n: {e}")

if __name__ == "__main__":
    export_example()

