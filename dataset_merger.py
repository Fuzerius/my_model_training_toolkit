#!/usr/bin/env python3
"""
Dataset Merger for YOLO Format Datasets
Merges multiple YOLO format datasets into a single consolidated dataset.

Features:
- User-specified output location
- Option to merge all datasets or select specific ones
- Maintains reference to original dataset in filenames
- Verifies class consistency across datasets
- Handles different image formats safely
- Generates statistics and new dataset.yaml
- Ignores unnecessary cache files
"""

import os
import shutil
import yaml
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import random

class DatasetMerger:
    def __init__(self):
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.stats = {
            'total_datasets': 0,
            'total_images': 0,
            'train': 0,
            'val': 0,
            'test': 0,
            'skipped_files': 0,
            'converted_files': 0
        }
        self.class_info = None
        self.image_counter = 1  # Start from 1 for better readability
        
    def get_user_input(self, prompt):
        """Get user input with error handling"""
        while True:
            try:
                user_input = input(prompt).strip()
                if user_input:
                    return user_input
                print("ERROR Please provide a valid input.")
            except KeyboardInterrupt:
                print("\nERROR Operation cancelled by user.")
                return None
    
    def get_output_location(self):
        """Get output location from user"""
        print("\n" + "="*60)
        print("DATASET MERGER - Output Location")
        print("="*60)
        
        while True:
            output_path = self.get_user_input("Enter the output path for merged dataset: ")
            if output_path is None:
                return None
                
            output_path = Path(output_path)
            
            # Check if path exists and is not empty
            if output_path.exists():
                if any(output_path.iterdir()):
                    overwrite = self.get_user_input(f"WARNING Directory '{output_path}' exists and is not empty. Overwrite? (yes/no): ")
                    if overwrite and overwrite.lower() in ['yes', 'y']:
                        try:
                            shutil.rmtree(output_path)
                            print(f"SUCCESS Cleared existing directory: {output_path}")
                        except Exception as e:
                            print(f"ERROR Could not clear directory: {e}")
                            continue
                    else:
                        continue
            
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                print(f"SUCCESS Output directory ready: {output_path.absolute()}")
                return output_path
            except Exception as e:
                print(f"ERROR Could not create directory: {e}")
    
    def get_available_datasets(self, base_path):
        """Get list of available datasets in the base path"""
        base_path = Path(base_path)
        if not base_path.exists():
            print(f"ERROR Base path does not exist: {base_path}")
            return []
        
        datasets = []
        for item in base_path.iterdir():
            if item.is_dir():
                # Check if it looks like a YOLO dataset (has dataset.yaml)
                yaml_file = item / "dataset.yaml"
                if yaml_file.exists():
                    datasets.append(item.name)
                else:
                    print(f"WARNING Skipping '{item.name}' - no dataset.yaml found")
        
        return sorted(datasets)
    
    def get_merge_mode(self):
        """Let user select merge mode (sequential or randomized)"""
        print("\n" + "="*60)
        print("DATASET MERGER - Merge Mode Selection")
        print("="*60)
        print("Choose how to merge datasets:")
        print("  1. Sequential - Copy all frames from dataset_1, then dataset_2, etc.")
        print("  2. Randomized - Randomly shuffle frames from all datasets (maintains train/val/test splits)")
        
        while True:
            try:
                choice = self.get_user_input("\nSelect merge mode (1 or 2): ")
                if choice is None:
                    return None
                
                if choice in ['1', 'sequential', 'seq']:
                    print("SUCCESS Selected sequential merge mode")
                    return 'sequential'
                elif choice in ['2', 'randomized', 'random', 'rand']:
                    print("SUCCESS Selected randomized merge mode")
                    return 'randomized'
                else:
                    print("ERROR Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\nERROR Operation cancelled by user.")
                return None
    
    def select_datasets(self, base_path):
        """Let user select which datasets to merge"""
        print("\n" + "="*60)
        print("DATASET MERGER - Dataset Selection")
        print("="*60)
        
        available_datasets = self.get_available_datasets(base_path)
        if not available_datasets:
            print("ERROR No valid YOLO datasets found in the specified directory!")
            return None
        
        print(f"Found {len(available_datasets)} valid YOLO datasets:")
        for i, dataset in enumerate(available_datasets, 1):
            print(f"  {i:2d}. {dataset}")
        
        print("\nSelection options:")
        print("  • Enter 'all' to merge all datasets")
        print("  • Enter numbers separated by commas (e.g., 1,3,5)")
        print("  • Enter dataset names separated by commas")
        
        while True:
            selection = self.get_user_input("\nSelect datasets to merge: ")
            if selection is None:
                return None
            
            if selection.lower() == 'all':
                print(f"SUCCESS Selected all {len(available_datasets)} datasets")
                return available_datasets
            
            # Parse selection
            selected_datasets = []
            try:
                for item in selection.split(','):
                    item = item.strip()
                    if item.isdigit():
                        idx = int(item) - 1
                        if 0 <= idx < len(available_datasets):
                            selected_datasets.append(available_datasets[idx])
                        else:
                            print(f"ERROR Invalid number: {item}. Please use 1-{len(available_datasets)}")
                            break
                    else:
                        if item in available_datasets:
                            selected_datasets.append(item)
                        else:
                            print(f"ERROR Dataset '{item}' not found")
                            break
                else:
                    # Remove duplicates while preserving order
                    unique_datasets = []
                    for dataset in selected_datasets:
                        if dataset not in unique_datasets:
                            unique_datasets.append(dataset)
                    
                    if unique_datasets:
                        print(f"SUCCESS Selected {len(unique_datasets)} datasets: {', '.join(unique_datasets)}")
                        return unique_datasets
                
                print("ERROR Please provide valid dataset selections.")
                
            except ValueError:
                print("ERROR Invalid input format. Please use numbers or dataset names.")
    
    def load_dataset_yaml(self, dataset_path):
        """Load and validate dataset.yaml file"""
        yaml_file = dataset_path / "dataset.yaml"
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # Validate required fields
            required_fields = ['nc', 'names']
            for field in required_fields:
                if field not in data:
                    print(f"ERROR Missing required field '{field}' in {yaml_file}")
                    return None
            
            return data
        except Exception as e:
            print(f"ERROR Could not load {yaml_file}: {e}")
            return None
    
    def verify_class_consistency(self, dataset_yamls):
        """Verify that all datasets have consistent class definitions"""
        print("\nVERIFYING class consistency across datasets...")
        
        reference_classes = None
        reference_dataset = None
        
        for dataset_name, yaml_data in dataset_yamls.items():
            if reference_classes is None:
                reference_classes = yaml_data['names']
                reference_dataset = dataset_name
                print(f"REFERENCE Using {dataset_name} as reference: {reference_classes}")
            else:
                if yaml_data['names'] != reference_classes:
                    print(f"ERROR Class mismatch in {dataset_name}:")
                    print(f"  Reference ({reference_dataset}): {reference_classes}")
                    print(f"  Current ({dataset_name}): {yaml_data['names']}")
                    return False
        
        print(f"SUCCESS All datasets have consistent classes: {reference_classes}")
        self.class_info = {
            'nc': len(reference_classes),
            'names': reference_classes
        }
        return True
    
    def convert_image_format(self, image_path, target_format='.jpg'):
        """Convert image to target format if needed"""
        image_path = Path(image_path)
        if image_path.suffix.lower() == target_format.lower():
            return image_path  # Already in target format
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (for JPEG)
                if target_format.lower() in ['.jpg', '.jpeg'] and img.mode != 'RGB':
                    img = img.convert('RGB')
                
                new_path = image_path.with_suffix(target_format)
                img.save(new_path, quality=95)  # High quality for JPEG
                
                # Remove original file
                image_path.unlink()
                self.stats['converted_files'] += 1
                return new_path
                
        except Exception as e:
            print(f"ERROR Could not convert {image_path}: {e}")
            return None
    
    def copy_dataset_files(self, source_dataset_path, dataset_name, output_path):
        """Copy files from a single dataset to the merged dataset (sequential mode)"""
        print(f"\nPROCESSING dataset: {dataset_name}")
        
        dataset_stats = {'train': 0, 'val': 0, 'test': 0}
        
        for split in ['train', 'val', 'test']:
            # Create output directories
            output_images_dir = output_path / 'images' / split
            output_labels_dir = output_path / 'labels' / split
            output_images_dir.mkdir(parents=True, exist_ok=True)
            output_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Source directories
            source_images_dir = source_dataset_path / 'images' / split
            source_labels_dir = source_dataset_path / 'labels' / split
            
            if not source_images_dir.exists():
                print(f"WARNING No {split} images directory found in {dataset_name}")
                continue
            
            # Process images
            image_files = []
            for ext in self.supported_image_formats:
                image_files.extend(source_images_dir.glob(f'*{ext}'))
            
            for image_file in image_files:
                # Generate new filename with dataset reference
                new_filename = f"{dataset_name}_{self.image_counter:06d}.jpg"
                
                # Convert image format if needed
                converted_image = self.convert_image_format(image_file, '.jpg')
                if converted_image is None:
                    self.stats['skipped_files'] += 1
                    continue
                
                # Copy image
                output_image_path = output_images_dir / new_filename
                shutil.copy2(converted_image, output_image_path)
                
                # Copy corresponding label file
                label_file = source_labels_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    output_label_path = output_labels_dir / f"{Path(new_filename).stem}.txt"
                    shutil.copy2(label_file, output_label_path)
                else:
                    # Create empty label file for images without annotations
                    output_label_path = output_labels_dir / f"{Path(new_filename).stem}.txt"
                    output_label_path.touch()
                
                self.image_counter += 1
                dataset_stats[split] += 1
                self.stats[split] += 1
                self.stats['total_images'] += 1
        
        print(f"SUCCESS Processed {dataset_name}: Train={dataset_stats['train']}, Val={dataset_stats['val']}, Test={dataset_stats['test']}")
        return dataset_stats
    
    def build_file_list(self, selected_datasets, base_path):
        """Build a list of all files from all datasets, organized by split"""
        print("\nBUILDING file list from all datasets...")
        
        file_lists = {
            'train': [],
            'val': [],
            'test': []
        }
        
        for dataset_name in selected_datasets:
            dataset_path = base_path / dataset_name
            
            for split in ['train', 'val', 'test']:
                source_images_dir = dataset_path / 'images' / split
                source_labels_dir = dataset_path / 'labels' / split
                
                if not source_images_dir.exists():
                    continue
                
                # Collect all image files
                image_files = []
                for ext in self.supported_image_formats:
                    image_files.extend(source_images_dir.glob(f'*{ext}'))
                
                # Add to file list with metadata
                for image_file in image_files:
                    label_file = source_labels_dir / f"{image_file.stem}.txt"
                    file_lists[split].append({
                        'dataset_name': dataset_name,
                        'image_path': image_file,
                        'label_path': label_file if label_file.exists() else None,
                        'split': split
                    })
        
        # Print statistics
        total_files = sum(len(file_lists[split]) for split in ['train', 'val', 'test'])
        print(f"SUCCESS Collected {total_files} files:")
        print(f"  • Train: {len(file_lists['train'])}")
        print(f"  • Val: {len(file_lists['val'])}")
        print(f"  • Test: {len(file_lists['test'])}")
        
        return file_lists
    
    def copy_files_randomized(self, file_lists, output_path):
        """Copy files in randomized order (within each split)"""
        print("\nCOPYING files in randomized order...")
        
        for split in ['train', 'val', 'test']:
            if not file_lists[split]:
                continue
                
            # Shuffle the file list for this split
            random.shuffle(file_lists[split])
            
            # Create output directories
            output_images_dir = output_path / 'images' / split
            output_labels_dir = output_path / 'labels' / split
            output_images_dir.mkdir(parents=True, exist_ok=True)
            output_labels_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\nPROCESSING {split} split ({len(file_lists[split])} files)...")
            
            for file_info in file_lists[split]:
                dataset_name = file_info['dataset_name']
                image_file = file_info['image_path']
                label_file = file_info['label_path']
                
                # Generate new filename with dataset reference and 6-digit counter
                new_filename = f"{dataset_name}_{self.image_counter:06d}.jpg"
                
                # Convert image format if needed
                converted_image = self.convert_image_format(image_file, '.jpg')
                if converted_image is None:
                    self.stats['skipped_files'] += 1
                    continue
                
                # Copy image
                output_image_path = output_images_dir / new_filename
                shutil.copy2(converted_image, output_image_path)
                
                # Copy corresponding label file
                if label_file and label_file.exists():
                    output_label_path = output_labels_dir / f"{Path(new_filename).stem}.txt"
                    shutil.copy2(label_file, output_label_path)
                else:
                    # Create empty label file for images without annotations
                    output_label_path = output_labels_dir / f"{Path(new_filename).stem}.txt"
                    output_label_path.touch()
                
                self.image_counter += 1
                self.stats[split] += 1
                self.stats['total_images'] += 1
            
            print(f"SUCCESS Processed {split} split: {self.stats[split]} files")
    
    def create_merged_yaml(self, output_path):
        """Create dataset.yaml for the merged dataset"""
        yaml_content = {
            'path': str(output_path.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': self.class_info['nc'],
            'names': self.class_info['names']
        }
        
        # Add statistics as comments
        yaml_content['# Dataset Statistics'] = None
        yaml_content[f'# Total images'] = self.stats['total_images']
        yaml_content[f'# Train'] = self.stats['train']
        yaml_content[f'# Val'] = self.stats['val']
        yaml_content[f'# Test'] = self.stats['test']
        yaml_content[f'# Merged from {self.stats["total_datasets"]} datasets'] = None
        
        yaml_file = output_path / 'dataset.yaml'
        try:
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
            print(f"SUCCESS Created dataset.yaml: {yaml_file}")
        except Exception as e:
            print(f"ERROR Could not create dataset.yaml: {e}")
    
    def print_final_statistics(self, output_path):
        """Print final merge statistics"""
        print("\n" + "="*60)
        print("MERGE COMPLETED - Final Statistics")
        print("="*60)
        print(f"Output location: {output_path.absolute()}")
        print(f"Total datasets merged: {self.stats['total_datasets']}")
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"  • Train: {self.stats['train']}")
        print(f"  • Validation: {self.stats['val']}")
        print(f"  • Test: {self.stats['test']}")
        print(f"Files converted: {self.stats['converted_files']}")
        print(f"Files skipped: {self.stats['skipped_files']}")
        print(f"Classes: {self.class_info['names']}")
        print("="*60)
    
    def merge_datasets(self):
        """Main method to merge datasets"""
        print("="*60)
        print("YOLO DATASET MERGER")
        print("="*60)
        
        # Get base path containing datasets
        base_path = self.get_user_input("Enter path to folder containing YOLO datasets: ")
        if base_path is None:
            return
        
        base_path = Path(base_path)
        if not base_path.exists():
            print(f"ERROR Path does not exist: {base_path}")
            return
        
        # Get output location
        output_path = self.get_output_location()
        if output_path is None:
            return
        
        # Select datasets to merge
        selected_datasets = self.select_datasets(base_path)
        if selected_datasets is None:
            return
        
        # Get merge mode
        merge_mode = self.get_merge_mode()
        if merge_mode is None:
            return
        
        # Load and verify dataset yamls
        print("\nLOADING dataset configurations...")
        dataset_yamls = {}
        for dataset_name in selected_datasets:
            dataset_path = base_path / dataset_name
            yaml_data = self.load_dataset_yaml(dataset_path)
            if yaml_data is None:
                print(f"ERROR Failed to load {dataset_name}. Aborting merge.")
                return
            dataset_yamls[dataset_name] = yaml_data
        
        # Verify class consistency
        if not self.verify_class_consistency(dataset_yamls):
            print("ERROR Class definitions are not consistent across datasets. Aborting merge.")
            return
        
        # Merge datasets based on selected mode
        self.stats['total_datasets'] = len(selected_datasets)
        
        if merge_mode == 'sequential':
            print(f"\nMERGING {len(selected_datasets)} datasets (sequential mode)...")
            for dataset_name in selected_datasets:
                dataset_path = base_path / dataset_name
                self.copy_dataset_files(dataset_path, dataset_name, output_path)
        else:  # randomized mode
            print(f"\nMERGING {len(selected_datasets)} datasets (randomized mode)...")
            file_lists = self.build_file_list(selected_datasets, base_path)
            self.copy_files_randomized(file_lists, output_path)
        
        # Create merged dataset.yaml
        self.create_merged_yaml(output_path)
        
        # Print final statistics
        self.print_final_statistics(output_path)
        
        print("SUCCESS Dataset merge completed successfully!")

def main():
    """Main function"""
    try:
        merger = DatasetMerger()
        merger.merge_datasets()
    except KeyboardInterrupt:
        print("\nERROR Operation cancelled by user.")
    except Exception as e:
        print(f"ERROR Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
