#!/usr/bin/env python3
"""
Advanced Image Quality Grader for YOLO Datasets
Evaluates image quality using multiple metrics:
- NIQE (Naturalness Image Quality Evaluator)
- BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator)
- PIQE (Perception-based Image Quality Evaluator)
- MUSIQ (Multi-Scale Image Quality Transformer)
- CONTRIQUE (Contrastive Learning for Image Quality Assessment)
- CLIPIQA (CLIP-based Image Quality Assessment)
"""

import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Check for required libraries
MISSING_LIBS = []

try:
    import cv2
    # Check if cv2.quality is available (needs opencv-contrib-python)
    try:
        cv2.quality.QualityBRISQUE_create
        OPENCV_QUALITY_AVAILABLE = True
    except AttributeError:
        OPENCV_QUALITY_AVAILABLE = False
        print("WARNING: cv2.quality not available. Install opencv-contrib-python for BRISQUE/NIQE from OpenCV.")
except ImportError:
    MISSING_LIBS.append("opencv-python")
    OPENCV_QUALITY_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    MISSING_LIBS.append("torch")
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'

try:
    import pyiqa
    PYIQA_AVAILABLE = True
except ImportError:
    MISSING_LIBS.append("pyiqa")
    PYIQA_AVAILABLE = False

# Warn about missing libraries but don't exit
# pyiqa is optional - script can work with OpenCV metrics only
if MISSING_LIBS:
    print("WARNING: Missing optional libraries:")
    for lib in MISSING_LIBS:
        print(f"  - {lib}")
    
    if 'pyiqa' in MISSING_LIBS:
        print("\nNOTE: pyiqa is optional. The script can still work with OpenCV quality metrics.")
        print("      However, some advanced metrics (MUSIQ, CONTRIQUE, CLIPIQA) require pyiqa.")
        print("\nTo install pyiqa (may require Windows Long Path support):")
        print("  pip install pyiqa")
        print("\nFor Windows Long Path support, see:")
        print("  https://pip.pypa.io/warnings/enable-long-paths")
    
    if 'opencv-python' in MISSING_LIBS:
        print("\nWARNING: opencv-python is required for basic image processing.")
        print("Install with: pip install opencv-python")
    
    print("\nFor OpenCV quality metrics (BRISQUE, NIQE), use:")
    print("  pip install opencv-contrib-python")
    print("\nContinuing with available metrics...\n")


def _evaluate_image_worker(args):
    """Worker function for multiprocessing - evaluates a single image"""
    image_path, metric_names, device, split_name, use_opencv = args
    
    # Initialize metrics in this worker process
    metrics = {}
    for metric_name in metric_names:
        try:
            # Try pyiqa first
            if PYIQA_AVAILABLE:
                metric = pyiqa.create_metric(metric_name, device=device)
                metrics[metric_name] = metric
                continue
            
            # Fallback to OpenCV for BRISQUE/NIQE (only if multiprocessing supports it)
            if use_opencv and metric_name.lower() in ['brisque', 'niqe'] and OPENCV_QUALITY_AVAILABLE:
                import cv2
                if metric_name.lower() == 'brisque':
                    metric = cv2.quality.QualityBRISQUE_create()
                elif metric_name.lower() == 'niqe':
                    metric = cv2.quality.QualityNIQE_create()
                metrics[metric_name] = metric
        except Exception as e:
            # If initialization fails, we'll handle it per-image
            pass
    
    results = {'filename': Path(image_path).name, 'split': split_name}
    errors = []
    
    try:
        # Convert path to string and use absolute path
        image_path_str = str(Path(image_path).absolute())
        
        # Verify image can be loaded
        from PIL import Image
        img = Image.open(image_path_str).convert('RGB')
        
        # Evaluate with each metric
        for metric_name in metric_names:
            if metric_name not in metrics:
                results[metric_name] = 'N/A'
                errors.append(f"{Path(image_path).name} - {metric_name}: Metric not initialized")
                continue
            
            try:
                metric = metrics[metric_name]
                # Check if this is an OpenCV metric
                if OPENCV_QUALITY_AVAILABLE and hasattr(metric, 'compute'):
                    import cv2
                    img_array = cv2.imread(image_path_str)
                    if img_array is None:
                        raise ValueError(f"Could not load image: {image_path_str}")
                    score, _ = metric.compute(img_array)
                    results[metric_name] = round(float(score[0]), 4)
                else:
                    # pyiqa metrics use path string
                    score = metric(image_path_str).item()
                    results[metric_name] = round(score, 4)
            except Exception as e:
                results[metric_name] = 'N/A'
                errors.append(f"{Path(image_path).name} - {metric_name}: {str(e)}")
    
    except Exception as e:
        # If image can't be loaded, fill all metrics with N/A
        for metric_name in metric_names:
            results[metric_name] = 'N/A'
        errors.append(f"{Path(image_path).name} - Failed to load: {str(e)}")
    
    return results, errors


class ImageQualityGrader:
    def __init__(self):
        self.device = DEVICE
        self.metrics = {}
        self.selected_metrics = []
        self.errors = []
        self.available_metrics = {
            'niqe': 'NIQE (Naturalness-based, no-reference)',
            'brisque': 'BRISQUE (Blind spatial quality, no-reference)',
            'piqe': 'PIQE (Perception-based, no-reference)',
            'musiq': 'MUSIQ (Multi-scale transformer, requires GPU for speed)',
            'clipiqa': 'CLIPIQA (CLIP-based quality assessment)',
            # Note: 'contrique' is not available in pyiqa 0.1.14.1
        }
        
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def initialize_metrics(self, selected_metrics):
        """Initialize the selected quality metrics"""
        print("\nInitializing metrics (downloading models if needed)...")
        
        self.selected_metrics = selected_metrics
        
        for metric_name in selected_metrics:
            try:
                print(f"  Loading {metric_name.upper()}...", end=' ')
                
                # Try pyiqa first (preferred)
                if PYIQA_AVAILABLE:
                    metric = pyiqa.create_metric(metric_name, device=self.device)
                    self.metrics[metric_name] = metric
                    print("✓ (pyiqa)")
                    continue
                
                # Fallback to OpenCV for BRISQUE and NIQE
                if metric_name.lower() in ['brisque', 'niqe'] and OPENCV_QUALITY_AVAILABLE:
                    try:
                        import cv2
                        if metric_name.lower() == 'brisque':
                            metric = cv2.quality.QualityBRISQUE_create()
                        elif metric_name.lower() == 'niqe':
                            metric = cv2.quality.QualityNIQE_create()
                        self.metrics[metric_name] = metric
                        print("✓ (OpenCV)")
                        continue
                    except Exception as e:
                        print(f"FAILED - OpenCV fallback error: {e}")
                        continue
                
                # No fallback available
                print("FAILED - pyiqa not available and no OpenCV fallback")
                self.errors.append(f"{metric_name}: Requires pyiqa (or opencv-contrib-python for BRISQUE/NIQE)")
                
            except Exception as e:
                print(f"FAILED - {e}")
                self.errors.append(f"Failed to initialize {metric_name}: {e}")
        
        if not self.metrics:
            print("\nERROR: No metrics were successfully initialized!")
            print("       Install pyiqa for full functionality, or opencv-contrib-python for BRISQUE/NIQE only.")
            return False
        
        print(f"\nSuccessfully initialized {len(self.metrics)} metric(s)")
        return True
    
    def evaluate_image(self, image_path):
        """Evaluate a single image with all initialized metrics"""
        results = {'filename': Path(image_path).name}
        
        try:
            # Convert path to string and use absolute path
            image_path_str = str(Path(image_path).absolute())
            
            # Verify image can be loaded
            from PIL import Image
            img = Image.open(image_path_str).convert('RGB')
            
            # Evaluate with each metric
            for metric_name, metric in self.metrics.items():
                try:
                    # Check if this is an OpenCV metric (different API)
                    if OPENCV_QUALITY_AVAILABLE and hasattr(metric, 'compute'):
                        # OpenCV quality metrics need image array
                        import cv2
                        img_array = cv2.imread(image_path_str)
                        if img_array is None:
                            raise ValueError(f"Could not load image: {image_path_str}")
                        score, _ = metric.compute(img_array)
                        results[metric_name] = round(float(score[0]), 4)
                    else:
                        # pyiqa metrics use path string
                        score = metric(image_path_str).item()
                        results[metric_name] = round(score, 4)
                except Exception as e:
                    results[metric_name] = 'N/A'
                    error_msg = f"{Path(image_path).name} - {metric_name}: {str(e)}"
                    if error_msg not in self.errors:
                        self.errors.append(error_msg)
        
        except Exception as e:
            # If image can't be loaded, fill all metrics with N/A
            for metric_name in self.metrics.keys():
                results[metric_name] = 'N/A'
            error_msg = f"{Path(image_path).name} - Failed to load: {str(e)}"
            if error_msg not in self.errors:
                self.errors.append(error_msg)
        
        return results
    
    def process_split(self, split_path, split_name, num_workers=1):
        """Process all images in a split directory"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Collect image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(split_path.glob(f'*{ext}'))
            image_files.extend(split_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"WARNING: No images found in {split_path}")
            return []
        
        print(f"\nProcessing {split_name} split: {len(image_files)} images")

        results = []
        
        if num_workers <= 1:
            # Single-threaded processing with tqdm
            for image_file in tqdm(image_files, desc=f"{split_name}", unit="img"):
                result = self.evaluate_image(image_file)
                result['split'] = split_name
                results.append(result)
        else:
            # Multi-process processing
            print(f"Using {num_workers} worker processes...")
            
            # Prepare arguments for worker processes
            # Include OpenCV flag if OpenCV metrics are available and no pyiqa
            use_opencv = not PYIQA_AVAILABLE and OPENCV_QUALITY_AVAILABLE
            work_args = [
                (str(image_file), self.selected_metrics, self.device, split_name, use_opencv)
                for image_file in image_files
            ]
            
            # Process with progress counter
            completed = 0
            total = len(image_files)
            print(f"Progress: 0/{total} (0.0%)", end='\r')
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                futures = [executor.submit(_evaluate_image_worker, args) for args in work_args]
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        result, errors = future.result()
                        results.append(result)
                        self.errors.extend(errors)
                    except Exception as e:
                        # Handle any unexpected errors
                        self.errors.append(f"Worker error: {str(e)}")
                    
                    completed += 1
                    percentage = (completed / total) * 100
                    print(f"Progress: {completed}/{total} ({percentage:.1f}%)", end='\r')
            
            print(f"Progress: {total}/{total} (100.0%) - Complete!")
        
        return results
    
    def save_results(self, results, output_path):
        """Save results to CSV file"""
        if not results:
            print("ERROR: No results to save!")
            return False
        
        df = pd.DataFrame(results)
        
        # Reorder columns: split, filename, then metrics
        metric_columns = [col for col in df.columns if col not in ['split', 'filename']]
        df = df[['split', 'filename'] + metric_columns]
        
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        for split in df['split'].unique():
            split_df = df[df['split'] == split]
            print(f"\n{split.upper()} Split ({len(split_df)} images):")
            
            for metric in metric_columns:
                # Convert to numeric, treating N/A as NaN
                numeric_values = pd.to_numeric(split_df[metric], errors='coerce')
                valid_count = numeric_values.notna().sum()
                
                if valid_count > 0:
                    print(f"  {metric.upper()}:")
                    print(f"    Mean:   {numeric_values.mean():.4f}")
                    print(f"    Median: {numeric_values.median():.4f}")
                    print(f"    Min:    {numeric_values.min():.4f}")
                    print(f"    Max:    {numeric_values.max():.4f}")
                    print(f"    Valid:  {valid_count}/{len(split_df)}")
                else:
                    print(f"  {metric.upper()}: No valid scores")
        
        return True
    
    def print_errors(self):
        """Print all errors encountered during processing"""
        if self.errors:
            print("\n" + "="*60)
            print(f"ERRORS ENCOUNTERED ({len(self.errors)})")
            print("="*60)
            for error in self.errors[:20]:  # Show first 20 errors
                print(f"  • {error}")
            if len(self.errors) > 20:
                print(f"  ... and {len(self.errors) - 20} more errors")


def is_valid_yolo_dataset(path):
    """Check if a folder is a valid YOLO dataset"""
    path = Path(path)
    required = ['images', 'labels', 'dataset.yaml']
    return all((path / item).exists() for item in required)


def get_parent_directory():
    """Prompt user for parent directory containing YOLO datasets"""
    print("\n" + "="*60)
    print("IMAGE QUALITY GRADER - Parent Directory Selection")
    print("="*60)
    
    while True:
        parent_path = input("Enter parent directory containing YOLO datasets: ").strip().strip('"')
        if not parent_path:
            print("ERROR: Please provide a valid path.")
            continue
        
        parent_path = Path(parent_path)
        
        # Check if path exists
        if not parent_path.exists():
            print(f"ERROR: Path does not exist: {parent_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None
            continue
        
        if not parent_path.is_dir():
            print(f"ERROR: Path is not a directory: {parent_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None
            continue
        
        return parent_path


def detect_yolo_datasets(parent_path):
    """Detect all valid YOLO datasets in parent directory"""
    datasets = []
    
    for item in parent_path.iterdir():
        if item.is_dir() and is_valid_yolo_dataset(item):
            datasets.append(item)
    
    return sorted(datasets, key=lambda x: x.name)


def get_dataset_selection(parent_path):
    """Prompt user for which datasets to process"""
    print("\n" + "="*60)
    print("Dataset Detection")
    print("="*60)
    
    datasets = detect_yolo_datasets(parent_path)
    
    if not datasets:
        print(f"ERROR: No valid YOLO datasets found in {parent_path}")
        print("Valid YOLO datasets must contain: images/, labels/, dataset.yaml")
        return None
    
    print(f"Found {len(datasets)} valid YOLO dataset(s):\n")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset.name}")
    
    print("\nOptions:")
    print("  • Enter 'all' to process all datasets")
    print("  • Enter numbers separated by commas (e.g., 1,2,3)")
    print("  • Enter dataset names separated by commas")
    
    while True:
        try:
            selection = input("\nSelect datasets: ").strip()
            
            if not selection:
                print("ERROR: Please select at least one dataset.")
                continue
            
            if selection.lower() == 'all':
                print(f"✓ Selected: All {len(datasets)} datasets")
                return datasets
            
            # Parse selection
            selected_datasets = []
            for item in selection.split(','):
                item = item.strip()
                
                # Try as number
                if item.isdigit():
                    idx = int(item) - 1
                    if 0 <= idx < len(datasets):
                        if datasets[idx] not in selected_datasets:
                            selected_datasets.append(datasets[idx])
                    else:
                        print(f"ERROR: Invalid number: {item}. Please use 1-{len(datasets)}")
                        break
                # Try as dataset name
                else:
                    matched = False
                    for dataset in datasets:
                        if dataset.name == item:
                            if dataset not in selected_datasets:
                                selected_datasets.append(dataset)
                            matched = True
                            break
                    if not matched:
                        print(f"ERROR: Unknown dataset: {item}")
                        break
            else:
                if selected_datasets:
                    print(f"✓ Selected: {', '.join([d.name for d in selected_datasets])}")
                    return selected_datasets
        
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None


def get_output_directory():
    """Prompt user for output directory"""
    print("\n" + "="*60)
    print("Output Directory Selection")
    print("="*60)
    
    while True:
        output_path = input("Enter output directory path (for CSV files): ").strip().strip('"')
        if not output_path:
            print("ERROR: Please provide a valid path.")
            continue
        
        output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        if not output_path.exists():
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created output directory: {output_path}")
            except Exception as e:
                print(f"ERROR: Could not create directory: {e}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
                continue
        
        if not output_path.is_dir():
            print(f"ERROR: Path exists but is not a directory: {output_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return None
            continue
        
        return output_path


def get_split_selection(dataset_path):
    """Prompt user for which splits to process"""
    print("\n" + "="*60)
    print("Split Selection")
    print("="*60)
    
    images_dir = dataset_path / 'images'
    available_splits = []
    
    # Check which splits exist
    for split in ['train', 'val', 'test']:
        split_path = images_dir / split
        if split_path.exists() and split_path.is_dir():
            # Count images
            image_count = len(list(split_path.glob('*.jpg'))) + \
                         len(list(split_path.glob('*.jpeg'))) + \
                         len(list(split_path.glob('*.png')))
            if image_count > 0:
                available_splits.append(split)
                print(f"  • {split}: {image_count} images")
    
    if not available_splits:
        print("ERROR: No valid splits found with images!")
        return None
    
    print("\nOptions:")
    print("  1. Process all splits")
    for i, split in enumerate(available_splits, 2):
        print(f"  {i}. Process only '{split}' split")
    
    while True:
        try:
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                print(f"✓ Selected: All splits ({', '.join(available_splits)})")
                return available_splits
            
            choice_idx = int(choice) - 2
            if 0 <= choice_idx < len(available_splits):
                selected = available_splits[choice_idx]
                print(f"✓ Selected: {selected} split")
                return [selected]
            else:
                print(f"ERROR: Please choose 1-{len(available_splits) + 1}")
        
        except ValueError:
            print("ERROR: Please enter a valid number")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None


def get_metric_selection(available_metrics):
    """Prompt user for which metrics to compute"""
    print("\n" + "="*60)
    print("Metric Selection")
    print("="*60)
    print("Available quality metrics:")
    
    for i, (metric_name, description) in enumerate(available_metrics.items(), 1):
        print(f"  {i}. {metric_name.upper()}: {description}")
    
    print("\nOptions:")
    print("  • Enter 'all' to compute all metrics")
    print("  • Enter numbers separated by commas (e.g., 1,2,3)")
    print("  • Enter metric names separated by commas (e.g., niqe,brisque)")
    
    metric_list = list(available_metrics.keys())
    
    while True:
        try:
            selection = input("\nSelect metrics: ").strip().lower()
            
            if not selection:
                print("ERROR: Please select at least one metric.")
                continue
            
            if selection == 'all':
                print(f"✓ Selected: All {len(metric_list)} metrics")
                return metric_list
            
            # Parse selection
            selected_metrics = []
            for item in selection.split(','):
                item = item.strip()
                
                # Try as number
                if item.isdigit():
                    idx = int(item) - 1
                    if 0 <= idx < len(metric_list):
                        selected_metrics.append(metric_list[idx])
                    else:
                        print(f"ERROR: Invalid number: {item}. Please use 1-{len(metric_list)}")
                        break
                # Try as metric name
                elif item in metric_list:
                    selected_metrics.append(item)
                else:
                    print(f"ERROR: Unknown metric: {item}")
                    break
            else:
                # Remove duplicates while preserving order
                unique_metrics = []
                for metric in selected_metrics:
                    if metric not in unique_metrics:
                        unique_metrics.append(metric)
                
                if unique_metrics:
                    print(f"✓ Selected: {', '.join(unique_metrics)}")
                    return unique_metrics
        
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None


def get_worker_count():
    """Prompt user for number of worker processes"""
    print("\n" + "="*60)
    print("Multiprocessing Configuration")
    print("="*60)
    
    cpu_count = multiprocessing.cpu_count()
    recommended = max(1, cpu_count - 1)
    
    print(f"CPU cores detected: {cpu_count}")
    print(f"Recommended workers: {recommended} (CPU cores - 1)")
    print("\nNOTE: Each worker process loads a copy of the models into memory.")
    print("⚠️  WARNING: Using too many workers may cause out-of-memory errors!")
    print("            Deep learning models (MUSIQ, CONTRIQUE, CLIPIQA) use significant RAM.")
    
    print("\nOptions:")
    print(f"  • Press Enter to use recommended ({recommended} workers)")
    print("  • Enter '1' for single-threaded (no parallelization)")
    print(f"  • Enter a number (1-{cpu_count}) to specify worker count")
    
    while True:
        try:
            choice = input("\nNumber of workers: ").strip()
            
            # Default to recommended
            if not choice:
                print(f"✓ Using {recommended} worker(s)")
                return recommended
            
            # Parse number
            num_workers = int(choice)
            
            if num_workers < 1:
                print("ERROR: Number of workers must be at least 1")
                continue
            
            if num_workers > cpu_count:
                print(f"WARNING: You have {cpu_count} CPU cores but requested {num_workers} workers")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            
            if num_workers > 4:
                print(f"WARNING: Using {num_workers} workers may consume significant memory!")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            
            print(f"✓ Using {num_workers} worker(s)")
            return num_workers
        
        except ValueError:
            print("ERROR: Please enter a valid number")
        except KeyboardInterrupt:
            print("\nUsing recommended value...")
            return recommended


def main():
    """Main function"""
    print("="*60)
    print("ADVANCED IMAGE QUALITY GRADER")
    print("="*60)
    print("Metrics: NIQE, BRISQUE, PIQE, MUSIQ, CONTRIQUE, CLIPIQA")
    print("="*60)
    
    try:
        # Get parent directory
        parent_path = get_parent_directory()
        if parent_path is None:
            print("Operation cancelled.")
            return
        
        # Get dataset selection
        selected_datasets = get_dataset_selection(parent_path)
        if selected_datasets is None:
            print("Operation cancelled.")
            return
        
        # Get output directory
        output_dir = get_output_directory()
        if output_dir is None:
            print("Operation cancelled.")
            return
        
        # Get split selection (will apply to all datasets)
        print("\n" + "="*60)
        print("Split Selection (applies to all selected datasets)")
        print("="*60)
        # Use first dataset to show available splits
        selected_splits = get_split_selection(selected_datasets[0])
        if selected_splits is None:
            print("Operation cancelled.")
            return
        
        # Initialize grader
        grader = ImageQualityGrader()
        
        # Get metric selection (will apply to all datasets)
        selected_metrics = get_metric_selection(grader.available_metrics)
        if selected_metrics is None:
            print("Operation cancelled.")
            return
        
        # Initialize metrics
        if not grader.initialize_metrics(selected_metrics):
            print("Failed to initialize metrics. Exiting.")
            return
        
        # Get worker count for multiprocessing
        num_workers = get_worker_count()
        
        # Process each dataset
        total_images_processed = 0
        output_files = []
        
        print("\n" + "="*60)
        print(f"PROCESSING {len(selected_datasets)} DATASET(S)")
        print("="*60)
        
        for dataset_idx, dataset_path in enumerate(selected_datasets, 1):
            print(f"\n{'='*60}")
            print(f"DATASET {dataset_idx}/{len(selected_datasets)}: {dataset_path.name}")
            print("="*60)
            
            # Reset errors for each dataset
            grader.errors = []
            
            # Process selected splits
            all_results = []
            images_dir = dataset_path / 'images'
            
            for split in selected_splits:
                split_path = images_dir / split
                if split_path.exists():
                    results = grader.process_split(split_path, split, num_workers)
                    all_results.extend(results)
                else:
                    print(f"WARNING: Split '{split}' not found in {dataset_path.name}, skipping...")
            
            if not all_results:
                print(f"\nWARNING: No images were processed for {dataset_path.name}!")
                continue
            
            # Save results for this dataset
            output_filename = f"{dataset_path.name}_quality_report.csv"
            output_path = output_dir / output_filename
            grader.save_results(all_results, output_path)
            
            # Print errors if any
            grader.print_errors()
            
            total_images_processed += len(all_results)
            output_files.append(output_path)
        
        # Final summary
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        print(f"Datasets processed: {len(output_files)}/{len(selected_datasets)}")
        print(f"Total images processed: {total_images_processed}")
        print(f"\nOutput files:")
        for output_file in output_files:
            print(f"  • {output_file.name}")
        print(f"\nOutput directory: {output_dir.absolute()}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
