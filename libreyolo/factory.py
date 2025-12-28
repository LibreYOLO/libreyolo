import torch
import re
import requests
from pathlib import Path
from .v8.nn import LibreYOLO8Model
from .v11.nn import LibreYOLO11Model
from .v8.model import LIBREYOLO8
from .v11.model import LIBREYOLO11
from .yolox.nn import YOLOXModel
from .yolox.model import LIBREYOLOX

# Registry for model classes
MODELS = {
    '8': LibreYOLO8Model,
    '11': LibreYOLO11Model,
    'x': YOLOXModel
}

def download_weights(model_path: str, size: str):
    """
    Download weights from Hugging Face if not found locally.
    
    Args:
        model_path: Path where weights should be saved
        size: Model size variant ('n', 's', 'm', 'l', 'x')
    """
    path = Path(model_path)
    if path.exists():
        return

    filename = path.name
    # Try to infer version from filename (e.g. libreyolo8n.pt -> 8, yolov11x.pt -> 11)
    match = re.search(r'(?:yolo|libreyolo)?(8|11)', filename.lower())
    if not match:
        raise ValueError(f"Could not determine model version (8 or 11) from filename '{filename}' for auto-download.")
    
    version = match.group(1)
    
    # Construct Hugging Face URL
    # Repo format: Libre-YOLO/yolov{version}{size}
    repo = f"Libre-YOLO/yolov{version}{size}"
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    
    print(f"Model weights not found at {model_path}. Attempting download from {url}...")
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int(100 * downloaded / total_size)
                        print(f"\rDownloading: {percent}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)", end="", flush=True)
            print("\nDownload complete.")
    except Exception as e:
        # If download fails, remove the empty/partial file
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Failed to download weights from {url}: {e}") from e

def create_model(version: str, config: str, reg_max: int = 16, nb_classes: int = 80, img_size: int = 640):
    """
    Create a fresh model instance for training.
    
    Args:
        version: "8", "11", etc.
        config: Model size ("n", "s", "m", "l", "x")
        reg_max: Regression max
        nb_classes: Number of classes
        img_size: Input image size (default: 640)
    
    Returns:
        Model instance (nn.Module)
    """
    if version not in MODELS:
        raise ValueError(f"Unsupported model version: {version}. Available: {list(MODELS.keys())}")
        
    model_cls = MODELS[version]
    return model_cls(config=config, reg_max=reg_max, nb_classes=nb_classes, img_size=img_size)

def LIBREYOLO(
    model_path: str,
    size: str = None,
    reg_max: int = 16,
    nb_classes: int = 80,
    save_feature_maps: bool = False,
    save_eigen_cam: bool = False,
    cam_method: str = "eigencam",
    cam_layer: str = None,
    device: str = "auto",
    tiling: bool = False
):
    """
    Unified Libre YOLO factory that automatically detects model version (8, 11, or X)
    from the weights file and returns the appropriate model instance.

    Args:
        model_path: Path to model weights file (.pt) or ONNX file (.onnx)
        size: Model size variant. Required for .pt files.
              - For YOLOv8/v11: "n", "s", "m", "l", "x"
              - For YOLOX: "nano", "tiny", "s", "m", "l", "x"
        reg_max: Regression max value for DFL (default: 16). Only used for v8/v11.
        nb_classes: Number of classes (default: 80 for COCO)
        save_feature_maps: If True, saves backbone feature map visualizations (v8/v11 only)
        save_eigen_cam: If True, saves EigenCAM heatmap visualizations (v8/v11 only)
        cam_method: Default CAM method for explain() (v8/v11 only)
        cam_layer: Target layer for CAM computation (v8/v11 only)
        device: Device for inference. "auto" (default) uses CUDA if available, else MPS, else CPU.
        tiling: Enable tiling for large images (default: False).

    Returns:
        Instance of LIBREYOLO8, LIBREYOLO11, LIBREYOLOX, or LIBREYOLOOnnx

    Example:
        >>> model = LIBREYOLO("yolo11n.pt", size="n")
        >>> detections = model("image.jpg", save=True)
        >>>
        >>> # For YOLOX
        >>> model = LIBREYOLO("yolox_s.pt", size="s")
        >>> detections = model("image.jpg", save=True)
    """
    # Handle ONNX models
    if model_path.endswith('.onnx'):
        from .common.onnx import LIBREYOLOOnnx
        return LIBREYOLOOnnx(model_path, nb_classes=nb_classes, device=device)
    
    # For .pt files, size is required
    if size is None:
        raise ValueError("size is required for .pt weights. Must be one of: 'n', 's', 'm', 'l', 'x'")
    
    if not Path(model_path).exists():
        try:
            download_weights(model_path, size)
        except Exception as e:
            # If download fails or is not possible, fall through to FileNotFoundError
            print(f"Auto-download failed: {e}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    # Load state dict once to avoid double loading
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e

    # Detect model version from state dict keys
    keys = list(state_dict.keys())

    # Check for YOLOX-specific layer names (e.g., 'backbone.backbone.stem' or Focus module)
    is_yolox = any('backbone.backbone' in key or 'head.stems' in key for key in keys)

    # Check for YOLO11-specific layer names (e.g., 'c2psa')
    is_yolo11 = any('c2psa' in key for key in keys)

    if is_yolox:
        # YOLOX detected - use LIBREYOLOX
        model = LIBREYOLOX(
            state_dict, size, nb_classes,
            device=device,
            tiling=tiling
        )
        model.version = "x"
        model.model_path = model_path
    elif is_yolo11:
        model = LIBREYOLO11(
            state_dict, size, reg_max, nb_classes,
            save_feature_maps=save_feature_maps,
            save_eigen_cam=save_eigen_cam,
            cam_method=cam_method,
            cam_layer=cam_layer,
            device=device,
            tiling=tiling
        )
        model.version = "11"
        model.model_path = model_path
    else:
        model = LIBREYOLO8(
            state_dict, size, reg_max, nb_classes,
            save_feature_maps=save_feature_maps,
            save_eigen_cam=save_eigen_cam,
            cam_method=cam_method,
            cam_layer=cam_layer,
            device=device,
            tiling=tiling
        )
        model.version = "8"
        model.model_path = model_path

    return model
