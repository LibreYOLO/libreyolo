import argparse
import torch
import torch.optim as optim
from pathlib import Path
import time
import yaml
import types

from libreyolo.factory import create_model
from libreyolo.training.loss import ComputeLoss
from libreyolo.training.dataset import YOLODataset, yolox_collate_fn as yolo_collate_fn
from torch.utils.data import DataLoader

def train(args):
    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    
    # 1. Initialize Model
    # 80 classes by default
    version = getattr(args, 'version', '8')
    print(f"Initializing LibreYOLO {version} model...")
    
    try:
        model = create_model(version, config=args.size, reg_max=16, nb_classes=args.num_classes).to(device)
    except ValueError as e:
        print(e)
        return
    
    if args.pretrained_weights:
        if Path(args.pretrained_weights).exists():
            try:
                state_dict = torch.load(args.pretrained_weights, map_location=device)
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights from {args.pretrained_weights}")
            except Exception as e:
                print(f"Error loading weights: {e}")
    
    model.train()
    
    # 2. Dataset and Dataloader
    # Get image size from model attributes
    img_size = getattr(model, 'img_size', 640)
    
    dataset = YOLODataset(
        img_path=args.data_path,
        img_size=img_size,
        transform=None # Add augmentations here if needed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=yolo_collate_fn,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # 3. Optimizer
    # Separate parameters for weight decay
    g_bn = []    # BatchNorm (no decay)
    g_w = []     # Weights (decay)
    g_b = []     # Biases (no decay)
    
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            g_bn.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            g_w.append(v.weight)
            
    optimizer = optim.AdamW([
        {'params': g_bn, 'weight_decay': 0.0},
        {'params': g_b, 'weight_decay': 0.0},
        {'params': g_w, 'weight_decay': args.weight_decay}
    ], lr=args.lr)
    
    # 4. Loss Function
    compute_loss = ComputeLoss(model, reg_max=16)
    
    # 5. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            imgs, targets, paths = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward
            preds = model(imgs)
            
            # Loss
            loss, loss_items = compute_loss(preds, {'targets': targets})
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} (Box: {loss_items[0]:.4f} Cls: {loss_items[1]:.4f} DFL: {loss_items[2]:.4f})")
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{args.epochs}] Completed. Avg Loss: {avg_loss:.4f}. Time: {time.time()-start_time:.2f}s")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = Path(args.output_dir) / f"libreyolo{version}{args.size}_epoch_{epoch+1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LibreYOLO with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    
    cli_args = parser.parse_args()
    
    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    # Convert dict to SimpleNamespace to keep dot notation in train function
    args = types.SimpleNamespace(**config_dict)

    if not hasattr(args, 'version'):
        print("Warning: 'version' not found in config, defaulting to '8'")
        args.version = '8'
    
    # Verify required arguments that might be missing from yaml if not validated
    if not hasattr(args, 'data_path'):
        raise ValueError("Configuration file must contain 'data_path'")
        
    train(args)
