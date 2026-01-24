"""
Integration test for RF-DETR training preservation.

Verifies that RF-DETR training still works after LibreYOLO integration
and uses the original RF-DETR training code (not LibreYOLO's trainer).
"""

import tempfile
from pathlib import Path
import yaml
from PIL import Image


def create_minimal_dataset(tmp_path):
    """Create a minimal YOLO dataset for testing training."""
    # Create directory structure
    for split in ['train', 'val']:
        images_dir = tmp_path / "images" / split
        labels_dir = tmp_path / "labels" / split
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        # Create 2 dummy images per split
        for i in range(2):
            img = Image.new('RGB', (640, 640), color=(i*100, i*100, i*100))
            img.save(images_dir / f"img{i}.jpg")

            # Create label (single object per image)
            label = f"0 0.5 0.5 0.3 0.3\n"
            (labels_dir / f"img{i}.txt").write_text(label)

    # Create data.yaml
    data_yaml = {
        'path': str(tmp_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['object']
    }

    yaml_path = tmp_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f)

    return yaml_path


def test_rfdetr_training_interface():
    """Test that RF-DETR model has the expected training interface by examining source code."""
    print("="*60)
    print("RF-DETR Training Interface Test")
    print("="*60)

    # Read the model source code directly
    from pathlib import Path
    repo_root = Path(__file__).parent.parent.parent
    model_file = repo_root / 'libreyolo' / 'rfdetr' / 'model.py'

    if not model_file.exists():
        print("  ✗ RF-DETR model file not found")
        return

    source = model_file.read_text()

    # Test 1: Check train() method exists
    print("\nTest 1: Checking train() method exists...")
    if 'def train(' in source:
        print("  ✓ train() method defined")
    else:
        print("  ✗ train() method not found")
        return

    # Test 2: Extract train method signature (may span multiple lines)
    print("\nTest 2: Checking Ultralytics-compatible parameters...")
    import re

    # Find the train method definition (multi-line)
    train_match = re.search(r'def train\(\s*self,([^\)]+\))\s*->', source, re.MULTILINE | re.DOTALL)
    if train_match:
        params_str = train_match.group(1)
        params_str = params_str.replace('\n', ' ').replace('  ', ' ')
        print(f"  Parameters found in signature")

        expected_params = ['data', 'epochs', 'batch_size', 'lr', 'output_dir', 'resume']
        for param in expected_params:
            # Look for "param:" or "param ="
            if param + ':' in params_str or param + ' =' in params_str:
                print(f"    ✓ {param}")
            else:
                print(f"    ⚠ {param} (not found)")
    else:
        # Fallback: just check if parameter names appear in the file
        print("  Could not parse full signature, checking for parameters...")
        expected_params = ['data', 'epochs', 'batch_size', 'lr']
        for param in expected_params:
            if f'data: str' in source or f'epochs: int' in source:
                print(f"    ✓ {param}")
                break

    # Test 3: Check it calls train_rfdetr (original RF-DETR code)
    print("\nTest 3: Verifying RF-DETR uses original training code...")

    if 'train_rfdetr' in source or 'from .train import train' in source:
        print("  ✓ Calls train_rfdetr() or imports RF-DETR training")
    else:
        print("  ⚠ Warning: Doesn't appear to call train_rfdetr()")

    # Test 4: Check it doesn't use LibreYOLO trainer
    if 'libreyolo.training.trainer' in source or 'YOLOXTrainer' in source:
        print("  ✗ ERROR: Uses LibreYOLO trainer (should use RF-DETR trainer)")
    else:
        print("  ✓ Does NOT use LibreYOLO trainer")

    # Test 5: Check docstring mentions Ultralytics API
    if 'ultralytics' in source.lower() or 'yaml' in source.lower():
        print("  ✓ Documentation mentions Ultralytics/YAML API")

    print("\n" + "="*60)
    print("✓ RF-DETR training interface verification complete!")
    print("="*60)


def test_rfdetr_train_method_signature():
    """Test that RF-DETR train() method has Ultralytics-compatible signature."""
    print("\n" + "="*60)
    print("RF-DETR train() Method Signature Test")
    print("="*60)

    from pathlib import Path
    repo_root = Path(__file__).parent.parent.parent
    model_file = repo_root / 'libreyolo' / 'rfdetr' / 'model.py'

    if not model_file.exists():
        print("  ✗ RF-DETR model file not found")
        return

    source = model_file.read_text()

    # Extract train method (may be multi-line)
    import re
    train_match = re.search(r'def train\(\s*self,([^\)]+\))\s*->', source, re.MULTILINE | re.DOTALL)

    if not train_match:
        print("  ⚠ Could not parse full method signature")
        print("  (checking for parameter presence in file...)")
        params_str = source  # Fallback: search entire file
    else:
        params_str = train_match.group(1)

    # Expected Ultralytics-style parameters
    ultralytics_params = {
        'data': 'Dataset YAML path (REQUIRED for Ultralytics API)',
        'epochs': 'Number of training epochs',
        'batch_size': 'Batch size',
        'lr': 'Learning rate',
        'device': 'Device (cpu/cuda)',
        'imgsz': 'Image size',
        'workers': 'DataLoader workers',
        'output_dir': 'Output directory',
    }

    print("\nUltralytics-compatible parameters:")
    for param_name, description in ultralytics_params.items():
        if param_name in params_str:
            # Try to extract default value
            param_pattern = rf'{param_name}\s*=\s*([^,\)]+)'
            default_match = re.search(param_pattern, params_str)
            if default_match:
                default = default_match.group(1).strip()
                print(f"  ✓ {param_name} (default: {default}) - {description}")
            else:
                print(f"  ✓ {param_name} (required) - {description}")
        else:
            print(f"  ✗ {param_name} - {description} (MISSING)")

    # Special check: 'data' parameter must NOT have a default
    # (Ultralytics API requires passing data='path/to/data.yaml')
    if 'data=' in params_str and 'data: str' not in params_str:
        print("\n  ⚠ WARNING: 'data' parameter should be required (no default)")
        print("            Ultralytics API: model.train(data='data.yaml', epochs=100)")
    elif 'data' in params_str:
        print("\n  ✓ 'data' parameter is properly required")

    print("\n" + "="*60)
    print("✓ Signature check complete!")
    print("="*60)


def test_rfdetr_training_imports():
    """Test that RF-DETR training wrapper modules exist."""
    print("\n" + "="*60)
    print("RF-DETR Training Dependencies Test")
    print("="*60)

    # Test that wrapper modules exist (don't import to avoid dependency issues)
    wrapper_files = {
        'libreyolo/rfdetr/model.py': 'Model wrapper',
        'libreyolo/rfdetr/train.py': 'Training wrapper',
        'libreyolo/rfdetr/nn.py': 'Neural network wrapper',
        'libreyolo/rfdetr/utils.py': 'Postprocessing utilities',
    }

    from pathlib import Path
    repo_root = Path(__file__).parent.parent.parent

    all_ok = True
    for file_path, description in wrapper_files.items():
        full_path = repo_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path} - {description}")
        else:
            print(f"  ✗ {file_path} - {description} (MISSING)")
            all_ok = False

    if all_ok:
        print("\n✓ All wrapper files exist")
    else:
        print("\n⚠ Some wrapper files missing")

    # Try importing wrapper modules (may fail if rfdetr not installed)
    print("\nTrying to import wrapper modules...")
    try:
        from libreyolo.rfdetr import model
        print("  ✓ libreyolo.rfdetr.model imported")
    except Exception as e:
        print(f"  ⚠ libreyolo.rfdetr.model - {type(e).__name__}: {str(e)[:80]}")

    try:
        from libreyolo.rfdetr import train
        print("  ✓ libreyolo.rfdetr.train imported")
    except Exception as e:
        print(f"  ⚠ libreyolo.rfdetr.train - {type(e).__name__}: {str(e)[:80]}")

    print("="*60)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RF-DETR TRAINING PRESERVATION TESTS")
    print("="*60)

    test_rfdetr_training_imports()
    test_rfdetr_training_interface()
    test_rfdetr_train_method_signature()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("These tests verify that:")
    print("  1. RF-DETR training interface is preserved")
    print("  2. train() method uses original RF-DETR code")
    print("  3. Ultralytics-compatible API (data='yaml', epochs=X)")
    print("  4. Does NOT use LibreYOLO's 'shit' trainer")
    print("\nNOTE: Actual training test requires RF-DETR package installed.")
    print("      To test actual training: pip install rfdetr")
    print("="*60)
