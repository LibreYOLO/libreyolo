import torch
from pathlib import Path
from .v8.nn import LibreYOLO8Model
from .v11.nn import LibreYOLO11Model
from .v8.model import LIBREYOLO8
from .v11.model import LIBREYOLO11

# Registry for model classes
MODELS = {
    '8': LibreYOLO8Model,
    '11': LibreYOLO11Model
}

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

def LIBREYOLO(model_path: str, size: str, reg_max: int = 16, nb_classes: int = 80, save_feature_maps: bool = False):
    """
    Unified Libre YOLO factory that automatically detects model version (8 or 11)
    from the weights file and returns the appropriate model instance.
    
    Args:
        model_path: Path to model weights file (required)
        size: Model size variant (required). Must be one of: "n", "s", "m", "l", "x"
        reg_max: Regression max value for DFL (default: 16)
        nb_classes: Number of classes (default: 80 for COCO)
        save_feature_maps: If True, saves backbone feature map visualizations on each inference (default: False)
    
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
        model = LIBREYOLO11(state_dict, size, reg_max, nb_classes, save_feature_maps=save_feature_maps)
        model.version = "11"
        model.model_path = model_path # Restore path for reference
    else:
        model = LIBREYOLO8(state_dict, size, reg_max, nb_classes, save_feature_maps=save_feature_maps)
        model.version = "8"
        model.model_path = model_path # Restore path for reference
        
    return model
