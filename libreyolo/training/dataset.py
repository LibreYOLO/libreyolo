import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class YOLODataset(Dataset):
    """
    Dataset for loading YOLO format data (images and labels).
    """
    def __init__(self, img_path, label_path=None, img_size=640, transform=None):
        """
        Args:
            img_path: Path to images directory
            label_path: Path to labels directory. If None, assumes adjacent 'labels' dir.
            img_size: Target image size
            transform: Optional transforms
        """
        self.img_path = Path(img_path)
        
        # Helper to find images
        IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
        self.img_files = sorted([x for x in self.img_path.rglob('*.*') if x.suffix.lower().strip('.') in IMG_FORMATS])
        
        if label_path:
             self.label_path = Path(label_path)
        else:
             # Assume /images/ -> /labels/ structure
             self.label_path = Path(str(self.img_path).replace('images', 'labels'))
             
        self.label_files = []
        for img_file in self.img_files:
            # Map image/file.jpg -> labels/file.txt
            # Handle different directory structures if needed, for now assume simple mapping
            # Replace images with labels in path? No, we have explicit label_path or inferred it.
            # We look for file.txt in label_path
            lbl_file = self.label_path / img_file.with_suffix('.txt').name
            self.label_files.append(lbl_file)
            
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Load Image
        img_path = self.img_files[index]
        label_path = self.label_files[index]
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # Load Labels
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    # class x_center y_center w h (normalized)
                    data = [float(x) for x in line.strip().split()]
                    if len(data) == 5:
                        boxes.append(data)
        
        boxes = np.array(boxes, dtype=np.float32)
        if len(boxes) == 0:
            boxes = np.zeros((0, 5), dtype=np.float32)
            
        # Resize Image (Simple resize to match utils.py logic)
        # Note: Standard YOLO uses Letterbox (pad), but utils.py uses simple resize.
        # We stick to simple resize for consistency with provided inference code.
        img_resized = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        
        # Labels are normalized (0-1), so they remain valid after simple resize 
        # as long as we simply stretched the image.
        
        # Convert to Tensor
        img_tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
        
        # Targets: [cls, x, y, w, h]
        targets = torch.from_numpy(boxes)
        
        return img_tensor, targets, str(img_path)

def yolov8_collate_fn(batch):
    """
    Collate function to handle variable number of boxes.
    Returns:
        imgs: (B, C, H, W)
        targets: (N, 6) -> [batch_idx, cls, x, y, w, h]
        paths: List of paths
    """
    imgs, targets, paths = zip(*batch)
    
    # Stack images
    imgs = torch.stack(imgs, 0)
    
    # Concatenate targets with batch index
    new_targets = []
    for i, t in enumerate(targets):
        if t.shape[0] > 0:
            # Add batch index column
            batch_idx = torch.full((t.shape[0], 1), i)
            new_t = torch.cat([batch_idx, t], dim=1)
            new_targets.append(new_t)
            
    if new_targets:
        targets = torch.cat(new_targets, 0)
    else:
        targets = torch.zeros((0, 6))
        
    return imgs, targets, paths


