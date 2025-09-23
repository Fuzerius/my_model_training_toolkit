#!/usr/bin/env python3
"""
Create Individual YOLO Datasets from Supervisely Format
Convert selected folders to individual YOLO datasets with 70/10/20 random split
"""

import json
import os
import cv2
import shutil
from pathlib import Path
import glob
from typing import Dict, List, Tuple
import argparse
import random
import math

class IndividualDatasetCreator:
    def __init__(self, input_dir: str, output_base_dir: str):
        self.input_dir = Path(input_dir)
        self.output_base_dir = Path(output_base_dir)
        
        # Class mapping - only classes that actually exist in your dataset
        self.class_mapping = {
            "Human": 0,
            "Vehicle": 1
        }
        
        # Classes to ignore (defined in meta.json but not used)
        self.ignored_classes = {"Tank", "Human_Group"}
        
        # Split ratios
        self.train_ratio = 0.7
        self.val_ratio = 0.1
        self.test_ratio = 0.2
        
        # Statistics for each dataset
        self.all_stats = {}
    
    def setup_dataset_dirs(self, dataset_name: str):
        """Create YOLO dataset directory structure for one dataset"""
        dataset_path = self.output_base_dir / dataset_name
        dirs = ['images/train', 'images/val', 'images/test', 
                'labels/train', 'labels/val', 'labels/test']
        
        for dir_name in dirs:
            (dataset_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        return dataset_path
    
    def convert_bbox_to_yolo(self, bbox: List[List[int]], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert Supervisely bbox to YOLO format (normalized)"""
        # Supervisely format: [[x1, y1], [x2, y2]]
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        
        # Calculate center coordinates and dimensions
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Normalize to [0, 1]
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height
        
        return center_x, center_y, width, height
    
    def extract_frame_from_video(self, video_path: str, frame_index: int, output_path: str) -> bool:
        """Extract a specific frame from video"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        
        if ret:
            cv2.imwrite(output_path, frame)
            cap.release()
            return True
        else:
            print(f"Error: Could not extract frame {frame_index} from {video_path}")
            cap.release()
            return False
    
    def get_all_frames_from_annotation(self, ann_file: str) -> List[Dict]:
        """Get all frames (annotated and unannotated) from annotation file"""
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        if not frames:
            return []
        
        # Get annotated frame indices
        annotated_indices = {frame['index']: frame for frame in frames}
        
        # Determine video frame range
        min_index = min(annotated_indices.keys())
        max_index = max(annotated_indices.keys())
        
        # Create complete frame list (annotated + unannotated)
        all_frames = []
        for i in range(min_index, max_index + 1):
            if i in annotated_indices:
                # Annotated frame
                all_frames.append({
                    'index': i,
                    'annotated': True,
                    'data': annotated_indices[i],
                    'img_size': data['size']
                })
            else:
                # Unannotated frame (negative example)
                all_frames.append({
                    'index': i,
                    'annotated': False,
                    'data': None,
                    'img_size': data['size']
                })
        
        return all_frames
    
    def process_frame(self, frame_info: Dict, video_file: str, video_name: str, split: str, dataset_path: Path) -> bool:
        """Process a single frame (annotated or unannotated)"""
        frame_index = frame_info['index']
        is_annotated = frame_info['annotated']
        
        # Create filenames
        suffix = "" if is_annotated else "_negative"
        img_filename = f"{video_name}_frame_{frame_index:06d}{suffix}.jpg"
        label_filename = f"{video_name}_frame_{frame_index:06d}{suffix}.txt"
        
        # Paths
        img_path = dataset_path / f"images/{split}" / img_filename
        label_path = dataset_path / f"labels/{split}" / label_filename
        
        # Extract frame from video
        if not self.extract_frame_from_video(video_file, frame_index, str(img_path)):
            return False
        
        # Create label file
        annotations = []
        if is_annotated and frame_info['data']:
            frame_data = frame_info['data']
            figures = frame_data.get('figures', [])
            img_width = frame_info['img_size']['width']
            img_height = frame_info['img_size']['height']
            
            # Process each annotation in the frame
            for figure in figures:
                # Get object class
                object_key = figure['objectKey']
                
                # Find class title from annotation data  
                ann_file_path = frame_info['ann_file']
                with open(ann_file_path, 'r') as f:
                    ann_data = json.load(f)
                
                class_title = None
                for obj in ann_data.get('objects', []):
                    if obj['key'] == object_key:
                        class_title = obj['classTitle']
                        break
                
                if class_title in self.ignored_classes:
                    continue
                elif class_title not in self.class_mapping:
                    print(f"Warning: Unknown class '{class_title}' in frame {frame_index}")
                    continue
                
                class_id = self.class_mapping[class_title]
                
                # Convert bbox
                if figure['geometryType'] == 'rectangle':
                    bbox = figure['geometry']['points']['exterior']
                    center_x, center_y, width, height = self.convert_bbox_to_yolo(
                        bbox, img_width, img_height
                    )
                    
                    # YOLO format: class_id center_x center_y width height
                    annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
        
        # Write label file (empty for unannotated frames)
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        
        return True
    
    def create_dataset_yaml(self, dataset_path: Path, dataset_name: str, stats: Dict):
        """Create YOLO dataset configuration file"""
        yaml_content = f"""# Military Object Detection Dataset - {dataset_name}
path: {dataset_path.absolute()}
train: images/train
val: images/val
test: images/test

# Classes (only Human and Vehicle)
nc: 2
names: ['Human', 'Vehicle']

# Dataset Statistics
# Total frames: {stats['total_frames']}
# Train: {stats['train_frames']}, Val: {stats['val_frames']}, Test: {stats['test_frames']}
# Annotated: {stats['annotated_frames']}, Unannotated: {stats['unannotated_frames']}
# Human annotations: {stats['class_counts'].get('Human', 0)}
# Vehicle annotations: {stats['class_counts'].get('Vehicle', 0)}

# Source: Folder {dataset_name}
# Split: 70% train, 10% val, 20% test (random)
"""
        
        yaml_path = dataset_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"📄 Dataset config saved: {yaml_path}")
    
    def process_single_folder(self, folder_identifier):
        """Process a single folder to create one YOLO dataset
        
        Args:
            folder_identifier: Can be an integer (folder number) or string (folder name)
        """
        # Convert folder identifier to string for path construction
        folder_name = str(folder_identifier)
        folder_path = self.input_dir / folder_name
        
        if not folder_path.exists():
            print(f"⚠️  Folder '{folder_name}' not found, skipping...")
            return None
        
        print(f"\n🔄 Processing Folder: {folder_name}")
        
        # Find dataset directory
        dataset_dirs = list(folder_path.glob('dataset*'))
        if not dataset_dirs:
            print(f"❌ No dataset directory found in folder '{folder_name}'")
            return None
        
        dataset_dir = dataset_dirs[0]
        ann_dir = dataset_dir / 'ann'
        video_dir = dataset_dir / 'video'
        
        if not ann_dir.exists() or not video_dir.exists():
            print(f"❌ Missing ann or video directory in folder '{folder_name}'")
            return None
        
        # Setup output dataset - use a clean name for the dataset
        # Replace any special characters that might cause issues
        clean_name = str(folder_identifier).replace(' ', '_').replace('-', '_')
        dataset_name = f"dataset_{clean_name}"
        dataset_path = self.setup_dataset_dirs(dataset_name)
        
        # Statistics
        stats = {
            'total_frames': 0,
            'annotated_frames': 0,
            'unannotated_frames': 0,
            'train_frames': 0,
            'val_frames': 0,
            'test_frames': 0,
            'class_counts': {'Human': 0, 'Vehicle': 0},
            'processed_videos': 0,
            'skipped_frames': 0
        }
        
        # Get all annotation files
        ann_files = list(ann_dir.glob('*.json'))
        print(f"📁 Found {len(ann_files)} annotation files")
        
        # Process each video
        all_video_frames = []  # Collect all frames from all videos
        
        for ann_file in ann_files:
            video_name = ann_file.stem
            video_file = str(video_dir / f"{video_name}")
            
            if not Path(video_file).exists():
                print(f"⚠️  Video file not found: {video_name}")
                continue
            
            # Get all frames (annotated + unannotated) from this video
            frames = self.get_all_frames_from_annotation(str(ann_file))
            
            if not frames:
                print(f"⚠️  No frames found in {video_name}")
                continue
            
            # Add video info to each frame
            for frame in frames:
                frame['video_file'] = video_file
                frame['video_name'] = video_name
                frame['ann_file'] = str(ann_file)
            
            all_video_frames.extend(frames)
            stats['processed_videos'] += 1
            
            print(f"  📹 {video_name}: {len(frames)} frames ({sum(1 for f in frames if f['annotated'])} annotated)")
        
        if not all_video_frames:
            print(f"❌ No frames to process in folder {folder_num}")
            return None
        
        # RANDOM SPLIT: Shuffle all frames then split
        print(f"🔀 Randomly splitting {len(all_video_frames)} frames...")
        random.shuffle(all_video_frames)
        
        total_frames = len(all_video_frames)
        train_count = int(total_frames * self.train_ratio)
        val_count = int(total_frames * self.val_ratio)
        
        train_frames = all_video_frames[:train_count]
        val_frames = all_video_frames[train_count:train_count + val_count]
        test_frames = all_video_frames[train_count + val_count:]
        
        print(f"📊 Split: {len(train_frames)} train, {len(val_frames)} val, {len(test_frames)} test")
        
        # Process frames for each split
        for split_name, frames in [('train', train_frames), ('val', val_frames), ('test', test_frames)]:
            print(f"  🔄 Processing {split_name} split ({len(frames)} frames)...")
            
            for i, frame in enumerate(frames):
                if i % 100 == 0:
                    print(f"    Progress: {i}/{len(frames)}")
                
                if self.process_frame(frame, frame['video_file'], frame['video_name'], split_name, dataset_path):
                    stats['total_frames'] += 1
                    stats[f'{split_name}_frames'] += 1
                    
                    if frame['annotated']:
                        stats['annotated_frames'] += 1
                        # Count class annotations (simplified)
                        if frame['data'] and 'figures' in frame['data']:
                            stats['class_counts']['Human'] += len([f for f in frame['data']['figures'] if 'Human' in str(f)])
                            stats['class_counts']['Vehicle'] += len([f for f in frame['data']['figures'] if 'Vehicle' in str(f)])
                    else:
                        stats['unannotated_frames'] += 1
                else:
                    stats['skipped_frames'] += 1
        
        # Create dataset.yaml
        self.create_dataset_yaml(dataset_path, dataset_name, stats)
        
        # Store stats
        self.all_stats[folder_identifier] = stats
        
        print(f"✅ Dataset {dataset_name} created successfully!")
        print(f"   📊 {stats['total_frames']} total frames")
        print(f"   📊 {stats['annotated_frames']} annotated, {stats['unannotated_frames']} unannotated")
        print(f"   📁 Saved to: {dataset_path}")
        
        return dataset_path
    
    def create_selected_datasets(self, folder_identifiers):
        """Create individual datasets for specified folder identifiers (numbers or names)"""
        print(f"🚀 Creating YOLO Datasets for folders: {', '.join(map(str, folder_identifiers))}")
        print("=" * 50)
        
        successful_datasets = []
        
        for folder_identifier in folder_identifiers:
            try:
                dataset_path = self.process_single_folder(folder_identifier)
                if dataset_path:
                    successful_datasets.append((folder_identifier, dataset_path))
            except Exception as e:
                print(f"❌ Error processing folder {folder_identifier}: {e}")
                continue
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 SUMMARY")
        print("=" * 50)
        
        for folder_identifier, dataset_path in successful_datasets:
            stats = self.all_stats[folder_identifier]
            print(f"Dataset {folder_identifier}:")
            print(f"  📁 Path: {dataset_path}")
            print(f"  📊 Frames: {stats['total_frames']} ({stats['annotated_frames']} annotated)")
            print(f"  🎯 Split: {stats['train_frames']} train, {stats['val_frames']} val, {stats['test_frames']} test")
            print(f"  🏷️  Classes: {stats['class_counts']['Human']} Human, {stats['class_counts']['Vehicle']} Vehicle")
        
        print(f"\n✅ Successfully created {len(successful_datasets)} out of {len(folder_identifiers)} requested datasets!")
        print(f"📁 All datasets saved in: {self.output_base_dir}")
        
        return successful_datasets

def get_folder_identifiers():
    """Get folder identifiers (numbers or names) from user input"""
    while True:
        try:
            user_input = input("\nEnter folder numbers/names separated by commas (e.g., 10,11,bird_dataset,annotated_lizards): ").strip()
            if not user_input:
                print("❌ Please enter at least one folder identifier.")
                continue
            
            # Parse comma-separated identifiers (numbers or strings)
            folder_identifiers = []
            for identifier in user_input.split(','):
                identifier = identifier.strip()
                if identifier:
                    # Try to convert to int if it's a number, otherwise keep as string
                    try:
                        folder_identifiers.append(int(identifier))
                    except ValueError:
                        folder_identifiers.append(identifier)
            
            if not folder_identifiers:
                print("❌ Please enter valid folder identifiers.")
                continue
                
            # Remove duplicates while preserving order and mixed types
            seen = set()
            unique_identifiers = []
            for item in folder_identifiers:
                if item not in seen:
                    seen.add(item)
                    unique_identifiers.append(item)
            
            print(f"✅ Selected folders: {', '.join(map(str, unique_identifiers))}")
            return unique_identifiers
            
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user.")
            return None

# Keep the old function for backward compatibility
def get_folder_numbers():
    """Get folder numbers from user input (backward compatibility)"""
    identifiers = get_folder_identifiers()
    if identifiers is None:
        return None
    
    # Convert all identifiers to strings for backward compatibility
    # The process_single_folder method will be updated to handle both
    return identifiers

def main():
    parser = argparse.ArgumentParser(description='Create individual YOLO datasets from Supervisely format for selected folders')
    parser.add_argument('--input', '-i', default='dataset_supervisely_format', help='Input directory containing numbered folders (default: dataset_supervisely_format)')
    parser.add_argument('--output', '-o', default='dataset_yolo_format', help='Output directory for individual datasets (default: dataset_yolo_format)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible splits')
    parser.add_argument('--folders', '-f', type=str, help='Comma-separated folder numbers (e.g., "10,11,12"). If not provided, will prompt user.')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible splits
    random.seed(args.seed)
    
    print(f"🎲 Using random seed: {args.seed}")
    print(f"📂 Input directory: {args.input}")
    print(f"📁 Output directory: {args.output}")
    
    # Get folder numbers
    if args.folders:
        try:
            folder_numbers = [int(x.strip()) for x in args.folders.split(',') if x.strip()]
            folder_numbers = sorted(list(set(folder_numbers)))
            print(f"✅ Using folders from command line: {', '.join(map(str, folder_numbers))}")
        except ValueError:
            print("❌ Invalid folder numbers in command line argument. Please use format: --folders '10,11,12'")
            return
    else:
        folder_numbers = get_folder_numbers()
        if folder_numbers is None:
            return
    
    creator = IndividualDatasetCreator(args.input, args.output)
    creator.create_selected_datasets(folder_numbers)

if __name__ == "__main__":
    main()
