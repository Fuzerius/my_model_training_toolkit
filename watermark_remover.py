#!/usr/bin/env python3
"""
Video Watermark Remover

This script removes watermarks from videos by mirroring adjacent areas to cover the watermarks.
Supports watermarks in top-left and top-right corners.
"""

import cv2
import numpy as np
import os
import sys
import glob
from pathlib import Path

# Output directory will be dynamically set based on video location
# OUTPUT_DIR will be set to 'output' folder in the same directory as the videos

def ensure_output_directory(base_path):
    """Create output directory if it doesn't exist.
    
    Args:
        base_path: Base directory where videos are located
    
    Returns:
        Path to the output directory if successful, None if failed
    """
    output_dir = os.path.join(base_path, "output")
    output_path = Path(output_dir)
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"FOLDER Created output directory: {output_dir}")
        except Exception as e:
            print(f"ERROR Error creating output directory: {e}")
            return None
    else:
        print(f"FOLDER Using output directory: {output_dir}")
    return output_dir

def get_processing_mode():
    """Ask user to choose between single video or folder processing."""
    print("=== Video Watermark Remover ===\n")
    print("Processing options:")
    print("1. Process a single video file")
    print("2. Process all videos in a folder")
    
    while True:
        try:
            choice = int(input("\nChoose processing mode (1-2): ").strip())
            if choice in [1, 2]:
                return choice
            else:
                print("Error: Please choose 1 or 2.")
        except ValueError:
            print("Error: Please enter a valid number.")

def get_single_video_path():
    """Get single video file path from user."""
    while True:
        video_path = input("\nEnter the path to your video file: ").strip().strip('"')
        if os.path.exists(video_path) and os.path.isfile(video_path):
            # Check if it's a video file
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
            if any(video_path.lower().endswith(ext) for ext in video_extensions):
                return video_path
            else:
                print("Error: File doesn't appear to be a video file. Please try again.")
        else:
            print(f"Error: File '{video_path}' not found. Please try again.")

def get_folder_path():
    """Get folder path from user."""
    while True:
        folder_path = input("\nEnter the path to the folder containing videos: ").strip().strip('"')
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Check if folder contains video files
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm', '*.m4v']
            video_files = []
            for ext in video_extensions:
                video_files.extend(glob.glob(os.path.join(folder_path, ext)))
                video_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            
            # Remove duplicates that may occur on case-insensitive file systems
            video_files = list(set(video_files))
            
            if video_files:
                print(f"Found {len(video_files)} video file(s) in the folder:")
                for i, video in enumerate(video_files[:5], 1):  # Show first 5 files
                    print(f"  {i}. {os.path.basename(video)}")
                if len(video_files) > 5:
                    print(f"  ... and {len(video_files) - 5} more files")
                return folder_path, video_files
            else:
                print("Error: No video files found in the specified folder. Please try again.")
        else:
            print(f"Error: Folder '{folder_path}' not found. Please try again.")

def get_user_input():
    """Get processing mode and watermark dimensions from user."""
    # Get processing mode
    mode = get_processing_mode()
    
    # Get path(s) based on mode
    if mode == 1:
        video_path = get_single_video_path()
        video_files = [video_path]
        base_path = os.path.dirname(video_path)
    else:
        base_path, video_files = get_folder_path()
    
    # Get watermark dimensions
    print(f"\n Watermark Configuration:")
    while True:
        try:
            width = int(input("Enter watermark width (pixels): ").strip())
            height = int(input("Enter watermark height (pixels): ").strip())
            if width > 0 and height > 0:
                break
            else:
                print("Error: Width and height must be positive integers.")
        except ValueError:
            print("Error: Please enter valid integers for width and height.")
    
    # Get watermark locations
    print("\nWatermark locations:")
    print("1. Top-left only")
    print("2. Top-right only")
    print("3. Both top-left and top-right")
    
    while True:
        try:
            choice = int(input("Choose watermark location (1-3): ").strip())
            if choice in [1, 2, 3]:
                break
            else:
                print("Error: Please choose 1, 2, or 3.")
        except ValueError:
            print("Error: Please enter a valid number.")
    
    locations = []
    if choice == 1:
        locations = ['top-left']
    elif choice == 2:
        locations = ['top-right']
    else:
        locations = ['top-left', 'top-right']
    
    return mode, video_files, base_path, width, height, locations


def create_mirrored_patch(frame, x, y, width, height, location):
    """
    Create a mirrored patch to cover the watermark.
    
    Args:
        frame: The video frame
        x, y: Top-left corner of the watermark area
        width, height: Dimensions of the watermark
        location: 'top-left' or 'top-right'
    
    Returns:
        Mirrored patch to cover the watermark
    """
    frame_height, frame_width = frame.shape[:2]
    
    if location == 'top-left':
        # For top-left watermark, mirror the area to the right of it
        source_x = x + width
        source_y = y
        
        # Ensure we don't go beyond frame boundaries
        available_width = min(width, frame_width - source_x)
        available_height = min(height, frame_height - source_y)
        
        if available_width <= 0 or available_height <= 0:
            print(f"Warning: Cannot create mirror patch for top-left watermark. Using adjacent area.")
            # Fallback: use area below the watermark
            source_x = x
            source_y = y + height
            available_width = min(width, frame_width - source_x)
            available_height = min(height, frame_height - source_y)
        
        # Extract the source patch
        source_patch = frame[source_y:source_y + available_height, 
                           source_x:source_x + available_width]
        
        # Flip horizontally to create mirror effect
        mirrored_patch = cv2.flip(source_patch, 1)
        
    elif location == 'top-right':
        # For top-right watermark, mirror the area to the left of it
        source_x = max(0, x - width)
        source_y = y
        
        # Ensure we don't go beyond frame boundaries
        available_width = min(width, x - source_x)
        available_height = min(height, frame_height - source_y)
        
        if available_width <= 0 or available_height <= 0:
            print(f"Warning: Cannot create mirror patch for top-right watermark. Using adjacent area.")
            # Fallback: use area below the watermark
            source_x = x
            source_y = y + height
            available_width = min(width, frame_width - source_x)
            available_height = min(height, frame_height - source_y)
        
        # Extract the source patch
        source_patch = frame[source_y:source_y + available_height, 
                           source_x:source_x + available_width]
        
        # Flip horizontally to create mirror effect
        mirrored_patch = cv2.flip(source_patch, 1)
    
    return mirrored_patch


def remove_watermarks(frame, width, height, locations):
    """
    Remove watermarks from a frame by applying mirrored patches.
    
    Args:
        frame: The video frame
        width, height: Watermark dimensions
        locations: List of watermark locations
    
    Returns:
        Frame with watermarks removed
    """
    frame_height, frame_width = frame.shape[:2]
    result_frame = frame.copy()
    
    for location in locations:
        if location == 'top-left':
            x, y = 0, 0
        elif location == 'top-right':
            x, y = frame_width - width, 0
        
        # Ensure watermark area is within frame bounds
        actual_width = min(width, frame_width - x)
        actual_height = min(height, frame_height - y)
        
        if actual_width <= 0 or actual_height <= 0:
            print(f"Warning: Watermark area for {location} is outside frame bounds.")
            continue
        
        # Create mirrored patch
        mirrored_patch = create_mirrored_patch(frame, x, y, actual_width, actual_height, location)
        
        # Apply the patch to cover the watermark
        patch_height, patch_width = mirrored_patch.shape[:2]
        result_frame[y:y + patch_height, x:x + patch_width] = mirrored_patch
    
    return result_frame


def process_video(video_path, width, height, locations, output_dir):
    """
    Process the entire video to remove watermarks.
    
    Args:
        video_path: Path to input video
        width, height: Watermark dimensions
        locations: List of watermark locations
        output_dir: Directory to save the processed video
    """
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file '{video_path}'")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo info:")
    print(f"  Resolution: {frame_width}x{frame_height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Watermark size: {width}x{height}")
    print(f"  Locations: {', '.join(locations)}")
    
    # Create output filename in the specified directory
    input_path = Path(video_path)
    output_filename = f"{input_path.stem}_watermark_removed{input_path.suffix}"
    output_path = Path(output_dir) / output_filename
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    
    if not out.isOpened():
        print("Error: Cannot create output video file")
        cap.release()
        return False
    
    print(f"\nProcessing video...")
    print(f"Output will be saved as: {output_path}")
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Remove watermarks from the frame
            processed_frame = remove_watermarks(frame, width, height, locations)
            
            # Write the processed frame
            out.write(processed_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Update progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\nError during processing: {e}")
    
    finally:
        # Clean up
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    if frame_count > 0:
        print(f"\nProcessing complete!")
        print(f"Processed {frame_count} frames")
        print(f"Output saved as: {output_path}")
        return True
    else:
        print("No frames were processed.")
        return False


def process_multiple_videos(video_files, base_path, width, height, locations, output_dir):
    """
    Process multiple videos to remove watermarks.
    
    Args:
        video_files: List of video file paths
        base_path: Base directory path
        width, height: Watermark dimensions
        locations: List of watermark locations
        output_dir: Directory to save the processed videos
    """
    total_videos = len(video_files)
    successful_videos = 0
    failed_videos = []
    
    print(f"\nVIDEO Processing {total_videos} video(s)...")
    print("=" * 60)
    
    for i, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"\nVIDEO Processing video {i}/{total_videos}: {video_name}")
        
        try:
            success = process_video(video_path, width, height, locations, output_dir)
            if success:
                successful_videos += 1
                print(f"SUCCESS Successfully processed: {video_name}")
            else:
                failed_videos.append(video_name)
                print(f"ERROR Failed to process: {video_name}")
        except Exception as e:
            failed_videos.append(video_name)
            print(f"ERROR Error processing {video_name}: {e}")
        
        # Add separator between videos
        if i < total_videos:
            print("-" * 40)
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"CHART PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {total_videos}")
    print(f"SUCCESS Successful: {successful_videos}")
    print(f"ERROR Failed: {len(failed_videos)}")
    
    if failed_videos:
        print(f"\nFailed videos:")
        for video in failed_videos:
            print(f"  - {video}")
    
    print(f"\nFOLDER All processed videos saved in: {output_dir}")
    
    return successful_videos, failed_videos


def main():
    """Main function to run the watermark remover."""
    try:
        # Get user input first to determine base path
        mode, video_files, base_path, width, height, locations = get_user_input()
        
        # Ensure output directory exists in the same location as videos
        output_dir = ensure_output_directory(base_path)
        if not output_dir:
            print("ERROR Cannot create output directory. Exiting.")
            sys.exit(1)
        
        if mode == 1:
            # Single video processing
            video_path = video_files[0]
            print(f"\nTARGET Processing single video: {os.path.basename(video_path)}")
            success = process_video(video_path, width, height, locations, output_dir)
            
            if success:
                print(f"\nSUCCESS Watermark removal completed successfully!")
                print(f"FOLDER Output saved in: {output_dir}")
            else:
                print("\nERROR Watermark removal failed.")
                sys.exit(1)
        else:
            # Multiple video processing
            successful_videos, failed_videos = process_multiple_videos(
                video_files, base_path, width, height, locations, output_dir
            )
            
            if successful_videos > 0:
                if len(failed_videos) == 0:
                    print(f"\nCOMPLETE All {successful_videos} video(s) processed successfully!")
                else:
                    print(f"\nWARNING  {successful_videos} video(s) processed successfully, {len(failed_videos)} failed.")
                print(f"FOLDER All outputs saved in: {output_dir}")
            else:
                print("\nERROR No videos were processed successfully.")
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nSTOP Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
