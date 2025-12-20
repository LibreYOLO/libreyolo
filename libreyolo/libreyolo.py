import torch
from pathlib import Path
from .libreyolo8 import LIBREYOLO8
from .libreyolo11 import LIBREYOLO11

def LIBREYOLO(model_path: str, size: str, reg_max: int = 16, nb_classes: int = 80):
    """
    Unified Libre YOLO factory that automatically detects model version (8 or 11)
    from the weights file and returns the appropriate model instance.
    
    Args:
        model_path: Path to model weights file (required)
        size: Model size variant (required). Must be one of: "n", "s", "m", "l", "x"
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
    
    Returns:
        Instance of LIBREYOLO8 or LIBREYOLO11
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    # Load state dict once to avoid double loading
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model weights from {model_path}: {e}") from e

    # Check for YOLO11-specific layer names (e.g., 'c2psa')
    is_yolo11 = any('c2psa' in key for key in state_dict.keys())

    if is_yolo11:
        model = LIBREYOLO11(state_dict, size, reg_max, nb_classes)
        model.version = "11"
        model.model_path = model_path # Restore path for reference
    else:
        model = LIBREYOLO8(state_dict, size, reg_max, nb_classes)
        model.version = "8"
        model.model_path = model_path # Restore path for reference
        
    return model
