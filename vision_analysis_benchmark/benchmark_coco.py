#!/usr/bin/env python3
"""
Vision Analysis Benchmark for LibreYOLO Models

Comprehensive benchmarking following Vision Analysis Benchmark Specification v1.0.
Measures accuracy, timing breakdown, throughput, and model statistics on COCO val2017.

Usage:
    python benchmark_coco.py --coco-yaml path/to/coco.yaml --output-dir ./results
"""

import argparse
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch

from libreyolo import LIBREYOLO


# ============================================================================
# Model Registry
# ============================================================================

LIBREYOLO_MODELS = {
    'yolov9': {
        'variants': ['t', 's', 'm', 'c'],
        'weights_pattern': 'libreyolo9{variant}.pt',
        'input_size': 640,
    },
    'yolox': {
        'variants': ['nano', 'tiny', 's', 'm', 'l', 'x'],
        'weights_pattern': 'libreyoloX{variant}.pt',
        'input_size': 640,
    },
}


# ============================================================================
# Hardware/Software Metadata Collection
# ============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """Collect GPU information."""
    gpu_name = "Unknown"
    memory_gb = 0
    driver = "Unknown"

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        parts = result.stdout.strip().split(', ')
        if len(parts) >= 3:
            gpu_name = parts[0]
            # Handle memory format like "40960 MiB" or "Insufficient Permissions"
            mem_str = parts[1]
            if 'MiB' in mem_str or 'MB' in mem_str:
                memory_gb = float(mem_str.split()[0]) / 1024
            driver = parts[2]
    except Exception as e:
        print(f"Warning: Could not get GPU info: {e}")

    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"

    return {
        'gpu': gpu_name,
        'gpu_memory_gb': round(memory_gb, 1),
        'driver_version': driver,
        'cuda_version': cuda_version,
    }


def get_cpu_info():
    """Get CPU model and core count."""
    try:
        if platform.system() == 'Linux':
            with open('/proc/cpuinfo', 'r') as f:
                lines = f.readlines()
            cpu_model = [l for l in lines if 'model name' in l][0].split(':')[1].strip()
            cpu_cores = len([l for l in lines if 'processor' in l])
        else:
            cpu_model = platform.processor()
            cpu_cores = platform.os.cpu_count() or 0
    except Exception:
        cpu_model = "Unknown"
        cpu_cores = 0

    return cpu_model, cpu_cores


def get_system_memory_gb() -> int:
    """Get total system RAM in GB."""
    try:
        if platform.system() == 'Linux':
            with open('/proc/meminfo', 'r') as f:
                mem_kb = int([l for l in f.readlines() if 'MemTotal' in l][0].split()[1])
            return mem_kb // (1024 ** 2)
        else:
            import psutil
            return psutil.virtual_memory().total // (1024 ** 3)
    except Exception:
        return 0


def get_software_info() -> Dict[str, str]:
    """Get Python and package versions."""
    import libreyolo

    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'libreyolo': getattr(libreyolo, '__version__', 'dev'),
    }


# ============================================================================
# Model Statistics
# ============================================================================

def count_parameters(model: torch.nn.Module) -> float:
    """Count model parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def estimate_gflops(model: torch.nn.Module, input_size: int = 640) -> float:
    """Estimate GFLOPs for a given input size."""
    try:
        from thop import profile
        dummy_input = torch.randn(1, 3, input_size, input_size).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        return 0.0
    except Exception as e:
        print(f"Warning: Could not estimate GFLOPs: {e}")
        return 0.0


# ============================================================================
# Main Benchmark Function
# ============================================================================

def benchmark_model(
    model_name: str,
    weights_path: str,
    size: str,
    coco_yaml: str,
    batch_size: int = 1,
    device: str = 'auto',
) -> Dict[str, Any]:
    """
    Benchmark a single model on COCO val2017 using proper validation API.

    Returns comprehensive benchmark results including accuracy, timing, throughput.
    """

    print(f"\n{'='*80}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*80}")

    # Determine family and variant
    for family, info in LIBREYOLO_MODELS.items():
        if size in info['variants']:
            model_family = family
            break
    else:
        model_family = "unknown"

    input_size = LIBREYOLO_MODELS.get(model_family, {}).get('input_size', 640)

    # Load model
    print(f"Loading model from {weights_path}...")
    model = LIBREYOLO(model_path=weights_path, size=size, device=device)

    # Model statistics
    print("Computing model statistics...")
    params_millions = count_parameters(model.model)
    gflops = estimate_gflops(model.model, input_size)

    print(f"  Parameters: {params_millions:.2f}M")
    print(f"  GFLOPs: {gflops:.2f}")

    # Run validation using proper API
    print(f"\nRunning COCO validation (batch_size={batch_size})...")
    val_results = model.val(
        data=coco_yaml,
        batch=batch_size,
        imgsz=input_size,
        conf=0.001,
        iou=0.6,
        verbose=True,
        plots=False,
    )

    # Extract metrics from validation results
    print("\nExtracting metrics...")

    # Accuracy metrics
    accuracy_metrics = {
        'mAP_50': val_results.get('metrics/mAP50', 0.0),
        'mAP_50_95': val_results.get('metrics/mAP50-95', 0.0),
        'precision': val_results.get('metrics/precision', 0.0),
        'recall': val_results.get('metrics/recall', 0.0),
        # Note: Size-specific mAP not available from .val() API currently
        'mAP_small': 0.0,
        'mAP_medium': 0.0,
        'mAP_large': 0.0,
    }

    # Timing metrics via separate pass
    # .val() doesn't return timing, so we run a quick timing benchmark
    print("\nRunning separate timing benchmark (100 images)...")

    from pycocotools.coco import COCO
    from tqdm import tqdm
    import numpy as np

    # Load COCO for timing pass
    coco_yaml_path = Path(coco_yaml)
    timing_successful = False

    if not coco_yaml_path.exists():
        print(f"  Warning: COCO yaml not found at {coco_yaml_path}")
    else:
        try:
            import yaml
            with open(coco_yaml_path) as f:
                coco_config = yaml.safe_load(f)

            coco_root = Path(coco_config['path'])
            ann_file = coco_root / 'annotations' / 'instances_val2017.json'
            img_dir = coco_root / 'images' / 'val2017'

            print(f"  Looking for annotations at: {ann_file}")
            print(f"  Looking for images at: {img_dir}")

            if not ann_file.exists():
                print(f"  Warning: Annotation file not found at {ann_file}")
            elif not img_dir.exists():
                print(f"  Warning: Image directory not found at {img_dir}")
            else:
                coco_gt = COCO(str(ann_file))
                img_ids = sorted(coco_gt.getImgIds())[:100]  # Sample 100 images

                timings = []
                use_cuda_events = torch.cuda.is_available()

                if not use_cuda_events:
                    print("  Warning: CUDA not available, using CPU timing")

                for img_id in tqdm(img_ids, desc="Timing"):
                    img_info = coco_gt.loadImgs(img_id)[0]
                    img_path = img_dir / img_info['file_name']

                    if use_cuda_events:
                        # Time with CUDA events for GPU
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        torch.cuda.synchronize()
                        start_event.record()

                        _ = model.predict(str(img_path), save=False, conf_thres=0.001, iou_thres=0.6)

                        end_event.record()
                        torch.cuda.synchronize()

                        elapsed_ms = start_event.elapsed_time(end_event)
                    else:
                        # Time with CPU timer
                        import time
                        start_time = time.perf_counter()

                        _ = model.predict(str(img_path), save=False, conf_thres=0.001, iou_thres=0.6)

                        end_time = time.perf_counter()
                        elapsed_ms = (end_time - start_time) * 1000.0

                    timings.append(elapsed_ms)

                # Compute statistics
                timings = np.array(timings)
                total_stats = {
                    'mean': round(float(np.mean(timings)), 4),
                    'std': round(float(np.std(timings)), 4),
                    'p50': round(float(np.percentile(timings, 50)), 4),
                    'p95': round(float(np.percentile(timings, 95)), 4),
                    'p99': round(float(np.percentile(timings, 99)), 4),
                }

                fps_mean = 1000.0 / total_stats['mean'] if total_stats['mean'] > 0 else 0
                fps_p50 = 1000.0 / total_stats['p50'] if total_stats['p50'] > 0 else 0

                print(f"  Mean latency: {total_stats['mean']:.2f}ms")
                print(f"  p50 latency: {total_stats['p50']:.2f}ms")
                print(f"  FPS (mean): {fps_mean:.1f}")
                timing_successful = True

        except Exception as e:
            print(f"  Error during timing pass: {e}")
            import traceback
            traceback.print_exc()

    if not timing_successful:
        print("  Setting timing metrics to 0 due to timing pass failure")
        total_stats = {'mean': 0.0, 'std': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0}
        fps_mean = 0.0
        fps_p50 = 0.0

    # Collect hardware/software info
    gpu_info = get_gpu_info()
    cpu_model, cpu_cores = get_cpu_info()
    ram_gb = get_system_memory_gb()
    software_info = get_software_info()

    # Assemble results
    results = {
        'model': {
            'name': model_name,
            'family': model_family,
            'variant': size,
            'source': 'libreyolo',
            'weights': Path(weights_path).name,
            'input_size': input_size,
        },
        'hardware': {
            'gpu': gpu_info['gpu'],
            'gpu_memory_gb': gpu_info['gpu_memory_gb'],
            'driver_version': gpu_info['driver_version'],
            'cuda_version': gpu_info['cuda_version'],
            'cpu': cpu_model,
            'cpu_cores': cpu_cores,
            'ram_gb': ram_gb,
        },
        'software': software_info,
        'accuracy': accuracy_metrics,
        'timing': {
            'batch_size': batch_size,
            'num_images': 5000,
            'total_ms': total_stats,
        },
        'throughput': {
            'fps_mean': round(fps_mean, 2),
            'fps_p50': round(fps_p50, 2),
        },
        'model_stats': {
            'params_millions': round(params_millions, 2),
            'gflops': round(gflops, 2),
        },
        'metadata': {
            'benchmark_date': datetime.now().strftime('%Y-%m-%d'),
            'benchmark_version': '1.0',
        }
    }

    # Print summary
    print(f"\nResults for {model_name}:")
    print(f"  mAP@50-95: {accuracy_metrics['mAP_50_95']:.4f}")
    print(f"  mAP@50:    {accuracy_metrics['mAP_50']:.4f}")
    print(f"  Precision: {accuracy_metrics['precision']:.4f}")
    print(f"  Recall:    {accuracy_metrics['recall']:.4f}")

    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Vision Analysis Benchmark for LibreYOLO models'
    )
    parser.add_argument(
        '--coco-yaml',
        type=str,
        required=True,
        help='Path to COCO dataset YAML configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('./results'),
        help='Output directory for benchmark results'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to benchmark (e.g., yolov8n yolov11s). Default: all'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference (default: 1 for accurate latency measurement)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Device to use (default: auto)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to benchmark
    if args.models:
        models_to_benchmark = []
        for model_spec in args.models:
            # Parse model spec (e.g., "yolov8n")
            for family, info in LIBREYOLO_MODELS.items():
                for variant in info['variants']:
                    # family is like 'yolov8', so just add variant
                    model_name = f"{family}{variant}"
                    if model_spec == model_name:
                        weights = info['weights_pattern'].format(variant=variant)
                        models_to_benchmark.append((model_name, weights, variant))
                        break
    else:
        # Benchmark all models
        models_to_benchmark = []
        for family, info in LIBREYOLO_MODELS.items():
            for variant in info['variants']:
                model_name = f"{family}{variant}"
                weights = info['weights_pattern'].format(variant=variant)
                models_to_benchmark.append((model_name, weights, variant))

    print(f"Will benchmark {len(models_to_benchmark)} models")

    # Run benchmarks
    all_results = []

    for model_name, weights_file, size in models_to_benchmark:
        try:
            results = benchmark_model(
                model_name=model_name,
                weights_path=weights_file,
                size=size,
                coco_yaml=args.coco_yaml,
                batch_size=args.batch_size,
                device=args.device,
            )

            # Save individual JSON
            output_file = args.output_dir / f"{model_name}_benchmark.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {output_file}")

            all_results.append(results)

        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate summary CSV
    if all_results:
        print("\nGenerating summary CSV...")
        csv_rows = []
        for r in all_results:
            csv_rows.append({
                'model': r['model']['name'],
                'family': r['model']['family'],
                'variant': r['model']['variant'],
                'mAP_50_95': r['accuracy']['mAP_50_95'],
                'mAP_50': r['accuracy']['mAP_50'],
                'precision': r['accuracy']['precision'],
                'recall': r['accuracy']['recall'],
                'fps_mean': r['throughput']['fps_mean'],
                'fps_p50': r['throughput']['fps_p50'],
                'latency_mean_ms': r['timing']['total_ms']['mean'],
                'latency_p50_ms': r['timing']['total_ms']['p50'],
                'latency_p95_ms': r['timing']['total_ms']['p95'],
                'latency_p99_ms': r['timing']['total_ms']['p99'],
                'params_M': r['model_stats']['params_millions'],
                'gflops': r['model_stats']['gflops'],
            })

        df = pd.DataFrame(csv_rows)
        csv_file = args.output_dir / 'benchmark_summary.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved summary to {csv_file}")

        # Print summary table
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(df.to_string(index=False))

    print(f"\nBenchmark complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
