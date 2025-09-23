# Data Processing Toolkit

A comprehensive toolkit for machine learning data processing, model training, and video operations.

## Features

### 🔄 Format Conversion
- **Supervisely → YOLO**: Convert Supervisely format datasets to YOLO format with individual dataset creation
- **Support for mixed identifiers**: Use both folder numbers (1,2,3) and folder names (bird_dataset, annotated_lizards)
- **Configurable settings**: Set default input/output folders

### 🏋️ Training & Evaluation
- **YOLO Training**: Train models with progressive training support
- **Model Evaluation**: Comprehensive evaluation with visualization support
- **Quick Results Summary**: Compare and analyze training results across multiple models
- **Visualization Support**: View confusion matrices, precision/recall curves, and more

### 🎬 Video Operations
- **Watermark Removal**: Remove watermarks from videos using mirroring techniques
- **Video Splitting**: Split large videos into smaller chunks (requires FFmpeg)
- **Batch Processing**: Process single videos or entire folders

## Requirements

```
ultralytics
opencv-python
numpy
pandas
pathlib
```

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. For video operations, install FFmpeg:
   - Download from https://www.gyan.dev/ffmpeg/builds/
   - Extract and place `ffmpeg.exe` and `ffprobe.exe` in `./ffmpeg_bin/` folder
   - Or install system-wide and add to PATH

## Usage

Run the main program:
```bash
python main.py
```

### Menu Options

1. **Format Conversion**
   - Convert between different dataset formats
   - Configure default folders in settings

2. **Train/Evaluate Datasets**
   - Train YOLO models with custom configurations
   - Evaluate trained models with detailed metrics
   - Compare results across multiple training runs

3. **Video Operations**
   - Remove watermarks from videos
   - Split videos into manageable chunks

## Configuration

The program uses `settings.txt` to store configuration:
- `default_input_folder`: Default input directory for dataset conversion
- `default_output_folder`: Default output directory for converted datasets
- `default_results_folder`: Default folder for training results

## File Structure

- `main.py`: Main program with unified menu interface
- `create_individual_datasets.py`: Supervisely to YOLO conversion
- `train_datasets.py`: YOLO model training functionality
- `quick_results_summary.py`: Training results analysis
- `watermark_remover.py`: Video watermark removal
- `video_splitter.py`: Video splitting functionality
- `train_config.yaml`: Training configuration parameters
- `settings.txt`: Program settings and defaults

## Training Configuration

Customize training parameters in `train_config.yaml`:
- Model selection (yolo11m.pt, etc.)
- Training epochs, batch size, image size
- Learning rates and optimization settings
- Data augmentation parameters

## License

For personal project only
