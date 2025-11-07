#!/usr/bin/env python3
"""
YOLO Annotation Viewer
Preview images with bounding boxes drawn from YOLO format labels.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import os


def get_class_colors(num_classes):
    """Generate distinct colors for each class"""
    # Generate colors that are visually distinct
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        colors.append([int(c) for c in color])
    return colors


def read_yolo_label(label_path, img_width, img_height):
    """Read YOLO format label file and convert to pixel coordinates"""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert normalized coordinates to pixel coordinates
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            
            boxes.append({
                'class_id': class_id,
                'bbox': [x1, y1, x2, y2]
            })
    
    return boxes


def draw_boxes(image, boxes, class_names, colors, line_thickness=2):
    """Draw bounding boxes on image"""
    for box in boxes:
        class_id = box['class_id']
        x1, y1, x2, y2 = box['bbox']
        
        # Get color for this class
        color = colors[class_id % len(colors)]
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label background and text
        if class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"Class {class_id}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return image


def load_dataset_yaml(dataset_path):
    """Load dataset.yaml to get class names"""
    yaml_path = dataset_path / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    class_names = data.get('names', [])
    return class_names


def get_image_files(image_dir, max_frames=None):
    """Get sorted list of image files"""
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    for ext in extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
        image_files.extend(list(image_dir.glob(f"*{ext.upper()}")))
    
    # Sort by filename
    image_files.sort(key=lambda x: x.name)
    
    # Limit number of frames if specified
    if max_frames and max_frames > 0:
        image_files = image_files[:max_frames]
    
    return image_files


def add_frame_info(image, frame_num, total_frames, filename):
    """Add frame information overlay to image"""
    # Create overlay for frame info
    overlay = image.copy()
    
    # Draw semi-transparent background for text
    cv2.rectangle(overlay, (10, 10), (500, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Add text information
    info_text = [
        f"Frame: {frame_num + 1}/{total_frames}",
        f"File: {filename[:50]}"
    ]
    
    y_offset = 30
    for i, text in enumerate(info_text):
        cv2.putText(
            image,
            text,
            (15, y_offset + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return image


def preview_annotations(dataset_path, split, max_frames, delay_ms=30):
    """Preview images with annotations in a window"""
    
    # Load class names
    print(f"Loading dataset configuration...")
    class_names = load_dataset_yaml(dataset_path)
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {', '.join(class_names)}")
    
    # Generate colors for classes
    colors = get_class_colors(num_classes)
    
    # Set up paths
    images_dir = dataset_path / "images" / split
    labels_dir = dataset_path / "labels" / split
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Get image files
    print(f"Scanning images in {split} split...")
    image_files = get_image_files(images_dir, max_frames)
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    total_frames = len(image_files)
    print(f"Found {total_frames} images")
    print(f"\nControls:")
    print(f"  SPACE or RIGHT ARROW: Next image")
    print(f"  LEFT ARROW: Previous image")
    print(f"  ESC or Q: Quit")
    print(f"  R: Reset to first image")
    print(f"\nPress any key to start preview...")
    
    # Create window
    window_name = f"YOLO Annotations - {split} ({total_frames} images)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    current_index = 0
    line_thickness = 2
    
    # Progress bar
    pbar = tqdm(total=total_frames, desc="Previewing", unit="frame")
    
    while True:
        if current_index < 0:
            current_index = 0
        if current_index >= total_frames:
            current_index = total_frames - 1
        
        # Read current image
        image_file = image_files[current_index]
        image = cv2.imread(str(image_file))
        
        if image is None:
            print(f"Warning: Could not read {image_file}, skipping...")
            current_index += 1
            continue
        
        height, width = image.shape[:2]
        
        # Read corresponding label file
        label_file = labels_dir / (image_file.stem + ".txt")
        boxes = read_yolo_label(label_file, width, height)
        
        # Draw boxes
        annotated_image = draw_boxes(image.copy(), boxes, class_names, colors, line_thickness)
        
        # Add frame info overlay
        annotated_image = add_frame_info(
            annotated_image,
            current_index,
            total_frames,
            image_file.name
        )
        
        # Update progress bar
        pbar.n = current_index + 1
        pbar.refresh()
        
        # Display image
        cv2.imshow(window_name, annotated_image)
        
        # Wait for key press
        key = cv2.waitKey(delay_ms) & 0xFF
        
        # Handle key presses
        if key == 27 or key == ord('q') or key == ord('Q'):  # ESC or Q
            break
        elif key == ord(' ') or key == 83:  # SPACE or RIGHT ARROW
            current_index += 1
        elif key == 81:  # LEFT ARROW
            current_index -= 1
        elif key == ord('r') or key == ord('R'):  # R - Reset
            current_index = 0
    
    # Cleanup
    pbar.close()
    cv2.destroyAllWindows()
    print(f"\nSUCCESS Preview completed. Viewed {current_index + 1}/{total_frames} images")


def main():
    """Main function"""
    print("=" * 60)
    print("YOLO ANNOTATION VIEWER")
    print("=" * 60)
    
    # Get dataset path
    while True:
        dataset_path_input = input("\nEnter path to YOLO dataset folder: ").strip()
        if not dataset_path_input:
            print("ERROR Path cannot be empty")
            continue
        
        dataset_path = Path(dataset_path_input)
        if not dataset_path.exists():
            print(f"ERROR Path does not exist: {dataset_path}")
            continue
        
        if not (dataset_path / "dataset.yaml").exists():
            print(f"ERROR dataset.yaml not found in {dataset_path}")
            continue
        
        break
    
    # Get split selection
    print("\nSelect split:")
    print("1. train")
    print("2. val")
    print("3. test")
    
    while True:
        split_choice = input("Enter choice (1-3): ").strip()
        if split_choice in ['1', '2', '3']:
            break
        print("ERROR Invalid choice. Please enter 1, 2, or 3")
    
    splits_map = {
        '1': 'train',
        '2': 'val',
        '3': 'test'
    }
    selected_split = splits_map[split_choice]
    
    # Get max frames
    while True:
        max_frames_input = input("\nEnter maximum number of frames (or press Enter for all): ").strip()
        if not max_frames_input:
            max_frames = None
            break
        
        try:
            max_frames = int(max_frames_input)
            if max_frames > 0:
                break
            print("ERROR Number must be positive")
        except ValueError:
            print("ERROR Please enter a valid number")
    
    # Check if split exists
    images_dir = dataset_path / "images" / selected_split
    if not images_dir.exists():
        print(f"ERROR {selected_split} split not found")
        return
    
    # Preview annotations
    try:
        preview_annotations(dataset_path, selected_split, max_frames)
    except Exception as e:
        print(f"ERROR Failed to preview annotations: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("PREVIEW COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nERROR Operation cancelled by user")
    except Exception as e:
        print(f"\nERROR Unexpected error: {e}")
        import traceback
        traceback.print_exc()

