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
from .v9.nn import LibreYOLO9Model
from .v9.model import LIBREYOLO9
# LIBREYOLORFDETR is imported lazily when needed to avoid dependency issues

# Registry for model classes
MODELS = {
    '8': LibreYOLO8Model,
    '11': LibreYOLO11Model,
    'x': YOLOXModel,
    '9': LibreYOLO9Model,
}

def download_weights(model_path: str, size: str):
    """
    Download weights from Hugging Face if not found locally.

    Args:
        model_path: Path where weights should be saved
        size: Model size variant:
              - For v8/v11: 'n', 's', 'm', 'l', 'x'
              - For YOLOX: 'nano', 'tiny', 's', 'm', 'l', 'x'
              - For v9: 't', 's', 'm', 'c'
    """
    path = Path(model_path)
    if path.exists():
        return

    filename = path.name

    # Check for YOLOX (e.g., libreyoloXs.pt, libreyoloXnano.pt)
    if re.search(r'libreyolox(nano|tiny|s|m|l|x)', filename.lower()):
        yolox_match = re.search(r'libreyolox(nano|tiny|s|m|l|x)', filename.lower())
        yolox_size = yolox_match.group(1)
        repo = f"Libre-YOLO/libreyoloX{yolox_size}"
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    # Check for YOLOv9 (e.g., libreyolo9c.pt)
    elif re.search(r'libreyolo9|yolov?9', filename.lower()):
        repo = f"Libre-YOLO/libreyolo9{size}"
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    else:
        # Try to infer version from filename (e.g. libreyolo8n.pt -> 8, libreyolo11x.pt -> 11)
        match = re.search(r'(?:yolo|libreyolo)?(8|11)', filename.lower())
        if not match:
            raise ValueError(f"Could not determine model version from filename '{filename}' for auto-download.")

        version = match.group(1)

        # Construct Hugging Face URL
        # Repo format: Libre-YOLO/libreyolo{version}{size}
        repo = f"Libre-YOLO/libreyolo{version}{size}"
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

def detect_rfdetr_size(keys):
    """
    Detect RF-DETR model size from state dict keys.

    RF-DETR sizes have different hidden dimensions and query counts:
    - nano (n): hidden_dim=128, num_queries=100
    - small (s): hidden_dim=256, num_queries=200
    - base (b): hidden_dim=256, num_queries=300
    - medium (m): hidden_dim=384, num_queries=300
    - large (l): hidden_dim=384, num_queries=300

    Args:
        keys: List of state dict keys

    Returns:
        Size code ('n', 's', 'b', 'm', 'l'), defaults to 'b' if cannot determine
    """
    # Look for embedding layers to determine hidden_dim
    for key in keys:
        if 'query_embed' in key.lower() or 'input_proj' in key.lower():
            # Try to infer from shape if available
            pass

    # If we can't determine from keys, default to base
    # User should provide size explicitly for RF-DETR models
    return 'b'

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
    device: str = "auto"
):
    """
    Unified Libre YOLO factory that automatically detects model version (8, 9, 11, or X)
    from the weights file and returns the appropriate model instance.

    Args:
        model_path: Path to model weights file (.pt/.pth) or ONNX file (.onnx)
        size: Model size variant. Required for .pt/.pth files.
              - For YOLOv8/v11: "n", "s", "m", "l", "x"
              - For YOLOX: "nano", "tiny", "s", "m", "l", "x"
              - For YOLOv9: "t", "s", "m", "c"
        reg_max: Regression max value for DFL (default: 16). Only used for v8/v11/v9.
        nb_classes: Number of classes (default: 80 for COCO)
        save_feature_maps: If True, saves backbone feature map visualizations (YOLO8 only)
        save_eigen_cam: If True, saves EigenCAM heatmap visualizations (YOLO8 only)
        cam_method: Default CAM method for explain() (YOLO8 only)
        cam_layer: Target layer for CAM computation (YOLO8 only)
        device: Device for inference. "auto" (default) uses CUDA if available, else MPS, else CPU.
    
    Note:
        XAI features (save_feature_maps, save_eigen_cam, cam_method, cam_layer, explain())
        are currently only supported by LIBREYOLO8. For other model versions, use
        the model class directly (e.g., LIBREYOLO11) for future XAI support.

    Returns:
        Instance of LIBREYOLO8, LIBREYOLO9, LIBREYOLO11, LIBREYOLOX, or LIBREYOLOOnnx

    Example:
        >>> model = LIBREYOLO("yolo11n.pt", size="n")
        >>> detections = model("image.jpg", save=True)
        >>> # Use tiling for large images
        >>> detections = model("large_image.jpg", save=True, tiling=True)
        >>>
        >>> # For YOLOX
        >>> model = LIBREYOLO("libreyoloXs.pt", size="s")
        >>> detections = model("image.jpg", save=True)
        >>>
        >>> # For YOLOv9
        >>> model = LIBREYOLO("yolov9c.pt", size="c")
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

    # Handle checkpoint format (e.g., YOLOX checkpoints with 'model' or 'ema' key)
    # Extract actual weights from nested checkpoint formats
    weights_dict = state_dict
    if 'ema' in state_dict and isinstance(state_dict.get('ema'), dict):
        ema_data = state_dict['ema']
        if 'module' in ema_data:
            weights_dict = ema_data['module']
        else:
            weights_dict = ema_data
    elif 'model' in state_dict and isinstance(state_dict.get('model'), dict):
        weights_dict = state_dict['model']

    # Detect model version from state dict keys
    keys = list(weights_dict.keys())
    keys_lower = [k.lower() for k in keys]

    # Check for RF-DETR (DINOv2 backbone or DETR architecture)
    # Look for transformer, encoder, decoder, dinov2, detr patterns
    is_rfdetr = any(
        'detr' in k or 'dinov2' in k or 'transformer' in k or
        ('encoder' in k and 'decoder' in k) or 'query_embed' in k or
        'class_embed' in k or 'bbox_embed' in k
        for k in keys_lower
    )

    # Check for YOLOX-specific layer names (e.g., 'backbone.backbone.stem' or Focus module)
    is_yolox = any('backbone.backbone' in key or 'head.stems' in key for key in keys)

    # Check for YOLOv9-specific layer names (RepNCSPELAN, ADown, SPPELAN, or LibreYOLO9 elan patterns)
    is_yolo9 = any('repncspelan' in k or 'adown' in k or 'sppelan' in k for k in keys_lower) or \
               (any('backbone.elan' in k or 'neck.elan' in k for k in keys))

    # Check for YOLO11-specific layer names (e.g., 'c2psa')
    is_yolo11 = any('c2psa' in key for key in keys)

    if is_rfdetr:
        # RF-DETR detected - use LIBREYOLORFDETR (lazy import)
        from .rfdetr.model import LIBREYOLORFDETR
        # Detect size from state dict if not provided
        if size is None:
            size = detect_rfdetr_size(keys)
        model = LIBREYOLORFDETR(
            model_path=weights_dict, size=size, nb_classes=nb_classes, device=device
        )
        model.version = "rfdetr"
        model.model_path = model_path
    elif is_yolox:
        # YOLOX detected - use LIBREYOLOX
        model = LIBREYOLOX(
            model_path=weights_dict, size=size, nb_classes=nb_classes, device=device
        )
        model.version = "x"
        model.model_path = model_path
    elif is_yolo9:
        # YOLOv9 detected - use LIBREYOLO9
        model = LIBREYOLO9(
            model_path=weights_dict, size=size, reg_max=reg_max, nb_classes=nb_classes,
            device=device
        )
        model.version = "9"
        model.model_path = model_path
    elif is_yolo11:
        # YOLOv11 detected - use LIBREYOLO11
        model = LIBREYOLO11(
            model_path=weights_dict, size=size, reg_max=reg_max, nb_classes=nb_classes,
            device=device
        )
        model.version = "11"
        model.model_path = model_path
    else:
        # Default to YOLOv8
        # Note: XAI features (save_feature_maps, save_eigen_cam, cam_method, cam_layer)
        # are only supported by LIBREYOLO8. Other model versions do not support
        # these parameters and will ignore them if passed through the factory.
        model = LIBREYOLO8(
            weights_dict, size, reg_max, nb_classes,
            save_feature_maps=save_feature_maps,
            save_eigen_cam=save_eigen_cam,
            cam_method=cam_method,
            cam_layer=cam_layer,
            device=device
        )
        model.version = "8"
        model.model_path = model_path

    return model
