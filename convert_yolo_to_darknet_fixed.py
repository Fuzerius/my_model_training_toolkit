#!/usr/bin/env python3
"""
Convert YOLO format datasets to Darknet format.
Fixed version based on rolodex_dataset example and Darknet documentation.

This script converts YOLO format datasets into the proper Darknet training format.
You can choose to convert a specific dataset or all datasets.

Usage:
    python convert_yolo_to_darknet_fixed.py                    # Interactive mode
    python convert_yolo_to_darknet_fixed.py dataset_1          # Convert specific dataset
    python convert_yolo_to_darknet_fixed.py all                # Convert all datasets
    python convert_yolo_to_darknet_fixed.py --list             # List available datasets

Author: Assistant
Date: 2025-09-17
"""

import os
import sys
import shutil
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


def find_available_datasets(source_base_dir: str = "dataset_yolo_format") -> List[str]:
    """Find all available YOLO datasets in the source directory."""
    if not Path(source_base_dir).exists():
        # Try relative paths from different locations
        possible_paths = [
            "dataset_yolo_format",
            "Annotated Data/dataset_yolo_format", 
            "../dataset_yolo_format",
            "./Annotated Data/dataset_yolo_format"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                source_base_dir = path
                break
        else:
            return []
    
    base_path = Path(source_base_dir)
    datasets = []
    
    # Look for dataset_X directories
    for item in base_path.iterdir():
        if item.is_dir() and item.name.startswith("dataset_"):
            # Check if it has the required structure
            if (item / "dataset.yaml").exists() and (item / "images").exists() and (item / "labels").exists():
                datasets.append(item.name)
    
    return sorted(datasets)


def load_dataset_yaml(yaml_path: str) -> Dict:
    """Load and parse the dataset.yaml file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def get_source_directory() -> Optional[str]:
    """Find the correct source directory for YOLO datasets."""
    possible_paths = [
        "dataset_yolo_format",
        "Annotated Data/dataset_yolo_format", 
        "../dataset_yolo_format",
        "./Annotated Data/dataset_yolo_format"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            print(f"📁 Found YOLO datasets in: {Path(path).resolve()}")
            return path
    
    print("❌ Could not find dataset_yolo_format directory")
    print("   Looked in:", possible_paths)
    return None


def create_names_file(class_names: List[str], output_path: str) -> None:
    """Create .names file with class names, one per line."""
    with open(output_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")


def create_data_file(dataset_name: str, num_classes: int, base_path: str, output_path: str) -> None:
    """Create .data configuration file for Darknet training."""
    # Use absolute paths like in the rolodex example
    abs_base_path = Path(base_path).resolve()
    data_content = f"""classes = {num_classes}
train = {abs_base_path}/train.txt
valid = {abs_base_path}/valid.txt
names = {abs_base_path}/{dataset_name}.names
backup = {abs_base_path}/backup
"""
    with open(output_path, 'w') as f:
        f.write(data_content)


def create_image_list_file(dataset_dir: Path, split_name: str, output_file: Path) -> int:
    """Create text file listing all image paths for train/valid splits."""
    split_dir = dataset_dir / split_name
    
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist, skipping {split_name} split")
        return 0
    
    # Get all images in the split directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(split_dir.glob(ext))
    
    # Filter to only include images that have corresponding annotation files in the same directory
    valid_images = []
    for img_file in image_files:
        # Check if corresponding .txt annotation exists in the same directory
        annotation_file = split_dir / (img_file.stem + '.txt')
        if annotation_file.exists():
            valid_images.append(img_file)
        else:
            print(f"Warning: No annotation found for {img_file.name}, skipping")
    
    # Write the image paths to the output file
    with open(output_file, 'w') as f:
        for img_file in sorted(valid_images):
            # Use absolute paths as required by Darknet
            abs_path = img_file.resolve()
            f.write(f"{abs_path}\n")
    
    print(f"Created {output_file} with {len(valid_images)} images")
    return len(valid_images)


def copy_and_flatten_dataset(source_dir: Path, dest_dir: Path) -> Tuple[int, int, int]:
    """
    Copy dataset to Darknet format: images and annotations in the same directories.
    Following rolodex_dataset example structure.
    Returns tuple of (train_count, val_count, test_count)
    """
    images_src = source_dir / "images"
    labels_src = source_dir / "labels"
    
    counts = {'train': 0, 'val': 0, 'test': 0}
    
    # Copy each split (train, val, test) - images and labels together
    for split in ['train', 'val', 'test']:
        src_images_split = images_src / split
        src_labels_split = labels_src / split
        dst_split = dest_dir / split
        
        if src_images_split.exists() and src_labels_split.exists():
            # Create destination directory for this split
            if dst_split.exists():
                shutil.rmtree(dst_split)
            dst_split.mkdir(parents=True, exist_ok=True)
            
            # Copy images and their corresponding annotation files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(src_images_split.glob(ext))
            
            copied_count = 0
            for img_file in image_files:
                # Copy image
                dst_img = dst_split / img_file.name
                shutil.copy2(img_file, dst_img)
                
                # Copy corresponding annotation file
                annotation_file = src_labels_split / (img_file.stem + '.txt')
                if annotation_file.exists():
                    dst_annotation = dst_split / (img_file.stem + '.txt')
                    shutil.copy2(annotation_file, dst_annotation)
                    copied_count += 1
                else:
                    print(f"Warning: No annotation found for {img_file.name}")
            
            counts[split] = copied_count
            print(f"Copied {split}: {copied_count} image-annotation pairs")
    
    return counts['train'], counts['val'], counts['test']


def create_cfg_file(dataset_name: str, num_classes: int, output_dir: Path) -> None:
    """Create a sample .cfg file based on YOLOv4-tiny template."""
    # Calculate values as per Darknet documentation
    max_batches = 2000 * max(num_classes, 6)  # At least 6000 for stability
    steps_80 = int(0.8 * max_batches)
    steps_90 = int(0.9 * max_batches)
    filters = (num_classes + 5) * 3
    
    cfg_content = f"""# Darknet configuration for {dataset_name}
# Based on YOLOv4-tiny template
# Classes: {num_classes}

[net]
# Training
batch=64
subdivisions=16
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches={max_batches}
policy=steps
steps={steps_80},{steps_90}
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1
groups = 2
group_id = 1

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1
groups = 2
group_id = 1

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

##################################

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={filters}
activation=linear

[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes={num_classes}
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 8

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters={filters}
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes={num_classes}
num=6
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
"""
    
    cfg_file = output_dir / f"{dataset_name}.cfg"
    with open(cfg_file, 'w') as f:
        f.write(cfg_content)
    
    print(f"Created configuration file: {cfg_file}")


def list_datasets_info(source_base_dir: str) -> None:
    """List all available datasets with their information."""
    datasets = find_available_datasets(source_base_dir)
    
    if not datasets:
        print("❌ No valid YOLO datasets found")
        return
    
    print(f"\n📋 Available datasets in {Path(source_base_dir).resolve()}:")
    print("=" * 60)
    
    for dataset_name in datasets:
        dataset_path = Path(source_base_dir) / dataset_name
        yaml_path = dataset_path / "dataset.yaml"
        
        try:
            config = load_dataset_yaml(str(yaml_path))
            classes = config.get('names', [])
            num_classes = len(classes)
            
            # Count images (from source YOLO format)
            train_imgs = len(list((dataset_path / "images" / "train").glob("*.jpg"))) if (dataset_path / "images" / "train").exists() else 0
            val_imgs = len(list((dataset_path / "images" / "val").glob("*.jpg"))) if (dataset_path / "images" / "val").exists() else 0
            test_imgs = len(list((dataset_path / "images" / "test").glob("*.jpg"))) if (dataset_path / "images" / "test").exists() else 0
            
            print(f"📦 {dataset_name}:")
            print(f"   Classes: {num_classes} ({', '.join(classes)})")
            print(f"   Images: {train_imgs} train, {val_imgs} val, {test_imgs} test")
            print()
            
        except Exception as e:
            print(f"📦 {dataset_name}: ❌ Error reading config - {e}")
            print()


def interactive_dataset_selection(source_base_dir: str) -> List[str]:
    """Interactive dataset selection."""
    datasets = find_available_datasets(source_base_dir)
    
    if not datasets:
        print("❌ No valid YOLO datasets found")
        return []
    
    print(f"\n🎯 Available datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"   {i}. {dataset}")
    
    print("\n📝 Options:")
    print("   • Enter numbers (e.g., '1,3,5' or '1-3' or '1 3 5')")
    print("   • Enter 'all' to convert all datasets")
    print("   • Enter 'q' to quit")
    
    while True:
        try:
            selection = input("\n👉 Your choice: ").strip().lower()
            
            if selection == 'q':
                return []
            
            if selection == 'all':
                return datasets
            
            # Parse selection
            selected_indices = set()
            
            # Handle comma-separated, space-separated, and ranges
            parts = selection.replace(',', ' ').split()
            
            for part in parts:
                if '-' in part:
                    # Handle range like "1-3"
                    start, end = map(int, part.split('-'))
                    selected_indices.update(range(start, end + 1))
                else:
                    # Handle single number
                    selected_indices.add(int(part))
            
            # Convert to dataset names
            selected_datasets = []
            for idx in selected_indices:
                if 1 <= idx <= len(datasets):
                    selected_datasets.append(datasets[idx - 1])
                else:
                    print(f"⚠️  Invalid selection: {idx} (valid range: 1-{len(datasets)})")
                    continue
            
            if selected_datasets:
                print(f"✅ Selected: {', '.join(selected_datasets)}")
                return selected_datasets
            else:
                print("❌ No valid datasets selected")
                
        except ValueError:
            print("❌ Invalid input. Please enter numbers, ranges, or 'all'")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return []


def convert_single_dataset(dataset_name: str, source_base_dir: str, dest_base_dir: str) -> bool:
    """Convert a single YOLO dataset to Darknet format."""
    print(f"\n=== Converting {dataset_name} ===")
    
    source_dir = Path(source_base_dir) / dataset_name
    dest_dir = Path(dest_base_dir) / dataset_name
    
    if not source_dir.exists():
        print(f"❌ Source directory {source_dir} does not exist")
        return False
    
    # Load dataset configuration
    yaml_path = source_dir / "dataset.yaml"
    if not yaml_path.exists():
        print(f"❌ dataset.yaml not found in {source_dir}")
        return False
    
    try:
        config = load_dataset_yaml(str(yaml_path))
        class_names = config['names']
        num_classes = len(class_names)
        
        print(f"📊 Classes: {class_names}")
        print(f"📊 Number of classes: {num_classes}")
        
        # Remove existing destination directory
        if dest_dir.exists():
            print(f"🗑️  Removing existing directory: {dest_dir}")
            shutil.rmtree(dest_dir, ignore_errors=True)
        
        # Create destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy dataset structure maintaining YOLO organization
        print("📁 Copying dataset structure...")
        train_count, val_count, test_count = copy_and_flatten_dataset(source_dir, dest_dir)
        
        # Create .names file
        names_file = dest_dir / f"{dataset_name}.names"
        create_names_file(class_names, str(names_file))
        print(f"📝 Created {names_file}")
        
        # Create .data file
        data_file = dest_dir / f"{dataset_name}.data"
        create_data_file(dataset_name, num_classes, str(dest_dir), str(data_file))
        print(f"📝 Created {data_file}")
        
        # Create train.txt and valid.txt files
        train_file = dest_dir / "train.txt"
        valid_file = dest_dir / "valid.txt"
        
        actual_train_count = create_image_list_file(dest_dir, "train", train_file)
        actual_val_count = create_image_list_file(dest_dir, "val", valid_file)
        
        # Create test.txt if test images exist
        test_file = dest_dir / "test.txt"
        actual_test_count = 0
        if (dest_dir / "test").exists():
            actual_test_count = create_image_list_file(dest_dir, "test", test_file)
        
        # Create backup directory
        backup_dir = dest_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        print(f"📁 Created backup directory: {backup_dir}")
        
        # Create .cfg file
        create_cfg_file(dataset_name, num_classes, dest_dir)
        
        # Copy original dataset.yaml for reference
        shutil.copy2(str(yaml_path), str(dest_dir / "original_dataset.yaml"))
        
        print(f"✅ Successfully converted {dataset_name}")
        print(f"📊 Final counts: {actual_train_count} train, {actual_val_count} val, {actual_test_count} test images")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main conversion function with command line argument support."""
    print("🚀 YOLO to Darknet Converter (Interactive Version)")
    print("Based on rolodex_dataset example and Darknet documentation")
    print("=" * 60)
    
    # Find source directory
    source_base_dir = get_source_directory()
    if not source_base_dir:
        return
    
    dest_base_dir = "dataset_darknet_format"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == '--list' or arg == '-l':
            list_datasets_info(source_base_dir)
            return
        elif arg == 'all':
            datasets_to_convert = find_available_datasets(source_base_dir)
            if not datasets_to_convert:
                print("❌ No datasets found to convert")
                return
        elif arg.startswith('dataset_'):
            # Single dataset specified
            if arg in find_available_datasets(source_base_dir):
                datasets_to_convert = [arg]
            else:
                print(f"❌ Dataset '{arg}' not found")
                print("Available datasets:")
                for dataset in find_available_datasets(source_base_dir):
                    print(f"   • {dataset}")
                return
        else:
            print(f"❌ Unknown argument: {arg}")
            print("Usage:")
            print("   python convert_yolo_to_darknet_fixed.py              # Interactive mode")
            print("   python convert_yolo_to_darknet_fixed.py dataset_1    # Convert specific dataset")
            print("   python convert_yolo_to_darknet_fixed.py all          # Convert all datasets")
            print("   python convert_yolo_to_darknet_fixed.py --list       # List available datasets")
            return
    else:
        # Interactive mode
        print("🎯 Interactive Mode")
        list_datasets_info(source_base_dir)
        datasets_to_convert = interactive_dataset_selection(source_base_dir)
        
        if not datasets_to_convert:
            print("👋 No datasets selected. Goodbye!")
            return
    
    # Create output directory
    Path(dest_base_dir).mkdir(exist_ok=True)
    
    # Convert selected datasets
    print(f"\n🔄 Converting {len(datasets_to_convert)} dataset(s)...")
    successful_conversions = []
    failed_conversions = []
    
    for dataset_name in datasets_to_convert:
        success = convert_single_dataset(dataset_name, source_base_dir, dest_base_dir)
        if success:
            successful_conversions.append(dataset_name)
        else:
            failed_conversions.append(dataset_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 CONVERSION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully converted: {len(successful_conversions)} datasets")
    if successful_conversions:
        print(f"   Datasets: {', '.join(successful_conversions)}")
    
    if failed_conversions:
        print(f"❌ Failed conversions: {len(failed_conversions)} datasets")
        print(f"   Datasets: {', '.join(failed_conversions)}")
    
    if successful_conversions:
        print(f"\n📁 Output directory: {Path(dest_base_dir).resolve()}")
        print("\n📋 Training Instructions:")
        example_dataset = successful_conversions[0]
        print("1. Navigate to a dataset directory:")
        print(f"   cd {Path(dest_base_dir).resolve()}/{example_dataset}")
        print("2. Start training with:")
        print(f"   darknet detector train {example_dataset}.data {example_dataset}.cfg")
        print("3. For multiple GPUs:")
        print(f"   darknet detector -gpus 0,1,2,3 train {example_dataset}.data {example_dataset}.cfg")
        print("4. Monitor progress by viewing chart.png")
        print(f"5. Best weights will be saved as {example_dataset}_best.weights")
    
    # Create a summary file
    summary = {
        "conversion_date": "2025-09-17",
        "conversion_type": "Interactive Darknet Format Converter",
        "successful_datasets": successful_conversions,
        "failed_datasets": failed_conversions,
        "total_datasets": len(successful_conversions) + len(failed_conversions),
        "output_directory": str(Path(dest_base_dir).resolve()),
        "source_directory": str(Path(source_base_dir).resolve()),
        "features": [
            "Interactive dataset selection",
            "Command line argument support",
            "Proper .names files created",
            "Absolute paths in .data files", 
            "Train/valid text files with image paths",
            "Complete .cfg files with correct filter values",
            "Backup directories for weights",
            "Maintains YOLO annotation format compatibility"
        ],
        "training_parameters": {
            "max_batches": "2000 * max(num_classes, 6) for stability",
            "steps": "80% and 90% of max_batches", 
            "filters": "(num_classes + 5) * 3",
            "batch_size": 64,
            "subdivisions": 16,
            "input_size": "416x416"
        }
    }
    
    summary_file = Path(dest_base_dir) / "conversion_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Conversion summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
