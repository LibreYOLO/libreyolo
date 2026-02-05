import torch
import re
import requests
from pathlib import Path
from .yolox.nn import YOLOXModel
from .yolox.model import LIBREYOLOX
from .v9.nn import LibreYOLO9Model
from .v9.model import LIBREYOLO9
# LIBREYOLORFDETR is imported lazily when needed to avoid dependency issues

# Registry for model classes
MODELS = {
    'x': YOLOXModel,
    '9': LibreYOLO9Model,
}

def download_weights(model_path: str, size: str):
    """
    Download weights from Hugging Face if not found locally.

    Args:
        model_path: Path where weights should be saved
        size: Model size variant:
              - For YOLOX: 'nano', 'tiny', 's', 'm', 'l', 'x'
              - For v9: 't', 's', 'm', 'c'
    """
    path = Path(model_path)
    if path.exists():
        return

    filename = path.name

    # Check for RF-DETR (e.g., librerfdetrnano.pth, librerfdetrsmall.pth)
    if re.search(r'librerfdetr(nano|small|base|medium|large)', filename.lower()):
        rfdetr_match = re.search(r'librerfdetr(nano|small|base|medium|large)', filename.lower())
        rfdetr_size = rfdetr_match.group(1)
        repo = f"Libre-YOLO/librerfdetr{rfdetr_size}"
        # Actual filename in repo is rf-detr-{size}.pth
        actual_filename = f"rf-detr-{rfdetr_size}.pth"
        url = f"https://huggingface.co/{repo}/resolve/main/{actual_filename}"
    # Check for YOLOX (e.g., libreyoloXs.pt, libreyoloXnano.pt)
    elif re.search(r'libreyolox(nano|tiny|s|m|l|x)', filename.lower()):
        yolox_match = re.search(r'libreyolox(nano|tiny|s|m|l|x)', filename.lower())
        yolox_size = yolox_match.group(1)
        repo = f"Libre-YOLO/libreyoloX{yolox_size}"
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    # Check for YOLOv9 (e.g., libreyolo9c.pt)
    elif re.search(r'libreyolo9|yolov?9', filename.lower()):
        repo = f"Libre-YOLO/libreyolo9{size}"
        url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    else:
        raise ValueError(f"Could not determine model version from filename '{filename}' for auto-download.")
    
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

def detect_size_from_filename(filename: str) -> tuple:
    """
    Extract model type and size from filename.

    Supports patterns like:
    - libreyoloXs.pt, libreyoloXnano.pt -> (x, nano/tiny/s/m/l/x)
    - libreyolo9c.pt -> (9, t/s/m/c)
    - librerfdetrnano.pth -> (rfdetr, n/s/b/m/l)

    Args:
        filename: The model filename

    Returns:
        Tuple of (model_type, size) or (None, None) if not detected
    """
    filename_lower = filename.lower()

    # YOLOX: libreyoloXs.pt, libreyoloXnano.pt, etc.
    yolox_match = re.search(r'libreyolox(nano|tiny|s|m|l|x)', filename_lower)
    if yolox_match:
        return ('x', yolox_match.group(1))

    # RF-DETR: librerfdetrnano.pth, librerfdetrsmall.pth, etc.
    rfdetr_match = re.search(r'librerfdetr(nano|small|base|medium|large)', filename_lower)
    if rfdetr_match:
        size_map = {'nano': 'n', 'small': 's', 'base': 'b', 'medium': 'm', 'large': 'l'}
        return ('rfdetr', size_map[rfdetr_match.group(1)])

    # YOLOv9: libreyolo9c.pt, yolov9t.pt, etc.
    yolo9_match = re.search(r'(?:libreyolo|yolov?)9([tsmc])', filename_lower)
    if yolo9_match:
        return ('9', yolo9_match.group(1))

    return (None, None)


def detect_rfdetr_size(state_dict):
    """
    Detect RF-DETR model size from checkpoint.

    Uses args.resolution stored in the checkpoint:
    - 384 -> nano (n)
    - 512 -> small (s)
    - 560 -> large (l)
    - 576 -> medium (m)

    Falls back to backbone_dim (768 = large, 384 = others) if args not present.

    Args:
        state_dict: Full checkpoint dict (not just keys)

    Returns:
        Size code ('n', 's', 'b', 'm', 'l') or None if cannot determine
    """
    RESOLUTION_TO_SIZE = {384: 'n', 512: 's', 560: 'l', 576: 'm'}

    # Primary: read resolution from checkpoint args
    args = state_dict.get('args')
    if args is not None:
        resolution = getattr(args, 'resolution', None) if hasattr(args, 'resolution') else args.get('resolution') if isinstance(args, dict) else None
        if resolution in RESOLUTION_TO_SIZE:
            return RESOLUTION_TO_SIZE[resolution]

    # Fallback: infer from backbone position_embeddings shape
    weights = state_dict.get('model', state_dict)
    pos_key = 'backbone.0.encoder.encoder.embeddings.position_embeddings'
    if pos_key in weights:
        backbone_dim = weights[pos_key].shape[2]
        pos_tokens = weights[pos_key].shape[1]
        if backbone_dim == 768:
            return 'l'
        # backbone_dim=384: distinguish by pos_tokens
        if pos_tokens == 577:
            return 'n'
        if pos_tokens == 1025:
            return 's'
        if pos_tokens == 1297:
            return 'm'

    return None

def _unwrap_state_dict(state_dict: dict) -> dict:
    """
    Extract weights from nested checkpoint formats (EMA, model wrappers).

    Args:
        state_dict: Raw checkpoint dict that may contain nested formats

    Returns:
        Unwrapped weights dict
    """
    weights_dict = state_dict
    if 'ema' in state_dict and isinstance(state_dict.get('ema'), dict):
        ema_data = state_dict['ema']
        weights_dict = ema_data.get('module', ema_data)
    elif 'model' in state_dict and isinstance(state_dict.get('model'), dict):
        weights_dict = state_dict['model']
    return weights_dict

def detect_yolo9_size(weights_dict: dict) -> str:
    """
    Detect YOLOv9 model size from state dict.

    Uses primary check on backbone.conv0.conv.weight shape[0]:
    - 't': 16 channels (unique)
    - 's', 'm': both 32 channels (need secondary check)
    - 'c': 64 channels (unique)

    For 32-channel models, uses secondary check on backbone.elan1.cv1.conv.weight:
    - 's': 64 channels
    - 'm': 128 channels

    Args:
        weights_dict: Model state dict (already unwrapped)

    Returns:
        Size code ('t', 's', 'm', 'c') or None if detection fails
    """
    key = 'backbone.conv0.conv.weight'
    if key not in weights_dict:
        return None

    first_channel = weights_dict[key].shape[0]

    # Tiny is unique (16 channels)
    if first_channel == 16:
        return 't'

    # Compact is unique (64 channels)
    if first_channel == 64:
        return 'c'

    # For 32-channel models (s and m), need secondary check
    if first_channel == 32:
        secondary_key = 'backbone.elan1.cv1.conv.weight'
        if secondary_key in weights_dict:
            mid_channel = weights_dict[secondary_key].shape[0]
            if mid_channel == 64:
                return 's'
            elif mid_channel == 128:
                return 'm'

    return None

def detect_yolox_size(weights_dict: dict) -> str:
    """
    Detect YOLOX model size from state dict.

    Checks backbone.backbone.stem.conv.conv.weight shape[0] to determine size:
    - 'nano': 16 channels
    - 'tiny': 24 channels
    - 's': 32 channels
    - 'm': 48 channels
    - 'l': 64 channels

    Args:
        weights_dict: Model state dict (already unwrapped)

    Returns:
        Size code ('nano', 'tiny', 's', 'm', 'l') or None if detection fails
    """
    key = 'backbone.backbone.stem.conv.conv.weight'
    if key not in weights_dict:
        return None

    first_channel = weights_dict[key].shape[0]

    size_map = {
        16: 'nano',
        24: 'tiny',
        32: 's',
        48: 'm',
        64: 'l',
        80: 'x',
    }

    return size_map.get(first_channel)

def detect_yolox_nb_classes(weights_dict: dict):
    """Detect number of classes from YOLOX state dict.

    Checks head.cls_preds.0.weight shape[0] (output channels = nb_classes).
    """
    key = 'head.cls_preds.0.weight'
    if key in weights_dict:
        return weights_dict[key].shape[0]
    return None

def detect_yolo9_nb_classes(weights_dict: dict):
    """Detect number of classes from YOLOv9 state dict.

    Checks the final Conv2d in the class branch (cv3.*.2.weight).
    """
    for key, tensor in weights_dict.items():
        if re.match(r'(?:head|detect)\.cv3\.\d+\.2\.weight', key):
            return tensor.shape[0]
    return None

def create_model(version: str, config: str, reg_max: int = 16, nb_classes: int = 80, img_size: int = 640):
    """
    Create a fresh model instance for training.

    Args:
        version: "x", "9", etc.
        config: Model size ("n", "s", "m", "l", "x" for YOLOX; "t", "s", "m", "c" for v9)
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
    nb_classes: int = None,
    device: str = "auto"
):
    """
    Unified Libre YOLO factory that automatically detects model version (9 or X),
    size, and number of classes from the weights file and returns the appropriate model instance.

    Args:
        model_path: Path to model weights file (.pt/.pth) or ONNX file (.onnx)
        size: Model size variant. Optional for .pt/.pth files (auto-detected if omitted).
              - For YOLOX: "nano", "tiny", "s", "m", "l", "x"
              - For YOLOv9: "t", "s", "m", "c"
        reg_max: Regression max value for DFL (default: 16). Only used for v9.
        nb_classes: Number of classes. Auto-detected from weights if not provided.
        device: Device for inference. "auto" (default) uses CUDA if available, else MPS, else CPU.

    Returns:
        Instance of LIBREYOLO9, LIBREYOLOX, or LIBREYOLOOnnx

    Example:
        >>> # For YOLOX
        >>> model = LIBREYOLO("libreyoloXs.pt")  # Auto-detect size
        >>> detections = model("image.jpg", save=True)
        >>>
        >>> # For YOLOv9
        >>> model = LIBREYOLO("yolov9c.pt")  # Auto-detect size
        >>> detections = model("image.jpg", save=True)
        >>>
        >>> # Use tiling for large images
        >>> detections = model("large_image.jpg", save=True, tiling=True)
    """
    # Handle ONNX models
    if model_path.endswith('.onnx'):
        from .common.onnx import LIBREYOLOOnnx
        return LIBREYOLOOnnx(model_path, nb_classes=nb_classes or 80, device=device)

    # For .pt files, handle file download if needed
    if not Path(model_path).exists():
        # Try to extract size from filename if not provided
        if size is None:
            _, detected_size = detect_size_from_filename(Path(model_path).name)
            if detected_size:
                size = detected_size
                print(f"Detected size '{size}' from filename")
            else:
                raise ValueError(
                    f"Model weights file not found: {model_path}\n"
                    f"Cannot auto-download: unable to determine size from filename.\n"
                    f"Please specify size explicitly (e.g., size='n', size='s') or provide a valid weights file path."
                )

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

    # Auto-detect size if not provided
    if size is None:
        # Unwrap state dict for consistent detection
        unwrapped_weights = _unwrap_state_dict(state_dict)

        # Detect size based on model type
        if is_yolox:
            size = detect_yolox_size(unwrapped_weights)
            model_type = "YOLOX"
        elif is_yolo9:
            size = detect_yolo9_size(unwrapped_weights)
            model_type = "YOLOv9"
        elif is_rfdetr:
            size = detect_rfdetr_size(state_dict)
            model_type = "RF-DETR"
        else:
            model_type = "Unknown"

        if size is None:
            raise ValueError(
                f"Could not automatically detect {model_type} model size from state dict.\n"
                f"Please specify size explicitly: LIBREYOLO('{model_path}', size='n')"
            )

        print(f"Auto-detected size: {size}")

    # Auto-detect nb_classes if not provided
    if nb_classes is None:
        if is_yolox:
            nb_classes = detect_yolox_nb_classes(weights_dict)
        elif is_yolo9:
            nb_classes = detect_yolo9_nb_classes(weights_dict)

        if nb_classes is None:
            nb_classes = 80

    if is_rfdetr:
        # RF-DETR detected - use LIBREYOLORFDETR (lazy import)
        # RF-DETR needs the path string, not the loaded weights dict
        try:
            from .rfdetr.model import LIBREYOLORFDETR
        except ImportError:
            raise ModuleNotFoundError(
                "RF-DETR support requires extra dependencies.\n"
                "Install with: pip install libreyolo[rfdetr]"
            )
        model = LIBREYOLORFDETR(
            model_path=model_path, size=size, nb_classes=nb_classes, device=device
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
    else:
        raise ValueError(
            f"Could not detect model architecture from state dict keys.\n"
            f"Supported architectures: YOLOX, YOLOv9, RF-DETR."
        )

    return model
