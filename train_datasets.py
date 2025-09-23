#!/usr/bin/env python3
"""
Train Selected Datasets with Optional Progressive Training
Each dataset will be trained sequentially and results saved in respective folders
Supports both normal and progressive training modes
"""

from ultralytics import YOLO
import time
import datetime
import os
from pathlib import Path

def check_config_file(config_path='train_config.yaml'):
    """Check if training configuration file exists"""
    if Path(config_path).exists():
        print(f"SUCCESS Training configuration found: {config_path}")
        return config_path
    else:
        print(f"ERROR Configuration file {config_path} not found. YOLO will use default parameters.")
        return None

def get_available_models(models_dir='models'):
    """Get list of available model files"""
    models_path = Path(models_dir)
    if not models_path.exists():
        print(f"ERROR Models directory '{models_dir}' not found.")
        return []
    
    model_files = list(models_path.glob('*.pt'))
    if not model_files:
        print(f"ERROR No .pt model files found in '{models_dir}' directory.")
        return []
    
    return [str(model.relative_to('.')) for model in model_files]

def get_model_selection():
    """Ask user to select a model from available models"""
    available_models = get_available_models()
    
    if not available_models:
        print("WARNING  No models found. Using default yolo11m.pt")
        return 'yolo11m.pt'
    
    while True:
        try:
            print(f"\nPACKAGE Available models:")
            for i, model in enumerate(available_models, 1):
                model_name = Path(model).name
                print(f"  {i}. {model_name}")
            
            choice = input(f"\nSelect a model (1-{len(available_models)}): ").strip()
            
            if not choice:
                print("ERROR Please enter a valid number.")
                continue
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(available_models):
                selected_model = available_models[choice_idx]
                print(f"SUCCESS Selected model: {Path(selected_model).name}")
                return selected_model
            else:
                print(f"ERROR Please enter a number between 1 and {len(available_models)}")
                
        except ValueError:
            print("ERROR Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled by user.")
            return None

def get_dataset_numbers():
    """Get dataset numbers from user input"""
    while True:
        try:
            user_input = input("\nEnter dataset numbers separated by commas (e.g., 1,2,3): ").strip()
            if not user_input:
                print("ERROR Please enter at least one dataset number.")
                continue
            
            # Parse comma-separated numbers
            dataset_numbers = []
            for num_str in user_input.split(','):
                num_str = num_str.strip()
                if num_str:
                    dataset_numbers.append(int(num_str))
            
            if not dataset_numbers:
                print("ERROR Please enter valid dataset numbers.")
                continue
                
            # Remove duplicates and sort
            dataset_numbers = sorted(list(set(dataset_numbers)))
            
            print(f"SUCCESS Selected datasets: {', '.join(map(str, dataset_numbers))}")
            return dataset_numbers
            
        except ValueError:
            print("ERROR Please enter valid numbers separated by commas (e.g., 1,2,3)")
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled by user.")
            return None

def get_training_mode():
    """Ask user for training mode preference"""
    while True:
        try:
            user_input = input("\nDo you want progressive training? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                print("SUCCESS Progressive training selected - each model will use the best.pt from the previous model")
                return True
            elif user_input in ['n', 'no']:
                print("SUCCESS Normal training selected - each model will start from yolo11m.pt")
                return False
            else:
                print("ERROR Please enter 'y' for yes or 'n' for no")
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled by user.")
            return None

def get_report_name():
    """Get custom report name from user"""
    while True:
        try:
            user_input = input("\nEnter the name for the training run (will be saved in runs/detect/[name]): ").strip()
            if not user_input:
                print("ERROR Please enter a valid name.")
                continue
            
            # Remove invalid characters for folder names
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                user_input = user_input.replace(char, '_')
            
            print(f"SUCCESS Training results will be saved as: runs/detect/{user_input}_dataset_X")
            return user_input
            
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled by user.")
            return None

def find_best_model(run_name, dataset_num, fallback_model='yolo11m.pt'):
    """Find the best.pt file from a previous training run"""
    run_path = Path(f"runs/detect/{run_name}_dataset_{dataset_num}")
    best_model_path = run_path / "weights" / "best.pt"
    
    if best_model_path.exists():
        return str(best_model_path)
    else:
        print(f"WARNING  Best model not found for dataset {dataset_num}, using {fallback_model}")
        return fallback_model

def train_single_dataset(dataset_num, progressive_mode=False, run_name="test", previous_dataset_num=None, 
                        base_model='yolo11m.pt', config_path=None):
    """Train a single dataset and return results"""
    print(f"\n{'='*60}")
    print(f"LAUNCH Starting training on Dataset {dataset_num}")
    print(f" Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Determine which model to use
    if progressive_mode and previous_dataset_num is not None:
        model_path = find_best_model(run_name, previous_dataset_num, base_model)
        print(f"PROCESSING Progressive training: Using {model_path}")
    else:
        model_path = base_model
        print(f" Normal training: Using {model_path}")
    
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Prepare basic training parameters
        train_params = {
            'data': f'dataset_yolo_format/dataset_{dataset_num}/dataset.yaml',
            'name': f'{run_name}_dataset_{dataset_num}'
        }
        
        # Use config file if available, otherwise use basic defaults
        if config_path:
            print(f"LIST Using configuration from {config_path}")
            # Let YOLO load the config file directly
            train_params['cfg'] = config_path
        else:
            print("LIST Using YOLO default parameters")
            # Only set essential parameters if no config file
            train_params.update({
                'epochs': 50,
                'batch': 16,
                'imgsz': 640
            })
        
        # Train the model - YOLO will handle all the config parsing
        results = model.train(**train_params)
        
        # Calculate training time
        training_time = time.time() - start_time
        hours = training_time // 3600
        minutes = (training_time % 3600) // 60
        
        print(f"\nSUCCESS Dataset {dataset_num} training completed!")
        print(f"TIME  Training time: {int(hours)}h {int(minutes)}m")
        print(f"FOLDER Results saved in: runs/detect/{run_name}_dataset_{dataset_num}/")
        
        # Print final metrics
        try:
            final_results = results.results_dict
            map50 = final_results.get('metrics/mAP50(B)', 0)
            map50_95 = final_results.get('metrics/mAP50-95(B)', 0)
            print(f"CHART Final mAP50: {map50:.4f}")
            print(f"CHART Final mAP50-95: {map50_95:.4f}")
            
            return {
                'dataset': dataset_num,
                'success': True,
                'training_time': training_time,
                'map50': map50,
                'map50_95': map50_95,
                'model_used': model_path,
                'run_name': run_name
            }
        except:
            print("CHART Check results.csv for detailed metrics")
            return {
                'dataset': dataset_num,
                'success': True,
                'training_time': training_time,
                'map50': None,
                'map50_95': None,
                'model_used': model_path,
                'run_name': run_name
            }
            
    except Exception as e:
        print(f"ERROR Error training Dataset {dataset_num}: {e}")
        return {
            'dataset': dataset_num,
            'success': False,
            'error': str(e),
            'training_time': time.time() - start_time,
            'model_used': model_path if 'model_path' in locals() else 'unknown',
            'run_name': run_name
        }

def main():
    print("TARGET AUTOMATED YOLO TRAINING WITH OPTIONS")
    print("=" * 60)
    print("TIP You can stop anytime with Ctrl+C")
    print("=" * 60)
    
    # Check for training configuration file
    config_path = check_config_file()
    
    # Get user inputs
    dataset_numbers = get_dataset_numbers()
    if dataset_numbers is None:
        return
    
    progressive_mode = get_training_mode()
    if progressive_mode is None:
        return
    
    base_model = get_model_selection()
    if base_model is None:
        return
    
    run_name = get_report_name()
    if run_name is None:
        return
    
    # Display training plan
    print(f"\nLIST TRAINING PLAN:")
    print(f"CHART Datasets to train: {', '.join(map(str, dataset_numbers))}")
    print(f"PROCESSING Progressive training: {'Yes' if progressive_mode else 'No'}")
    print(f"MODEL Base model: {Path(base_model).name}")
    print(f"SETTINGS  Configuration: {config_path if config_path else 'YOLO defaults'}")
    print(f"FOLDER Results will be saved in: runs/detect/{run_name}_dataset_X/")
    print(f"TIME  Estimated time per dataset: 2-5 hours (depending on GPU and dataset size)")
    print("=" * 60)
    
    # Confirm before starting
    try:
        confirm = input("\nProceed with training? (y/n): ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("ERROR Training cancelled.")
            return
    except KeyboardInterrupt:
        print("\nERROR Training cancelled.")
        return
    
    results_summary = []
    overall_start_time = time.time()
    
    for i, dataset_num in enumerate(dataset_numbers):
        try:
            # For progressive training, use the previous dataset number (if available)
            previous_dataset = dataset_numbers[i-1] if progressive_mode and i > 0 else None
            
            result = train_single_dataset(
                dataset_num=dataset_num,
                progressive_mode=progressive_mode,
                run_name=run_name,
                previous_dataset_num=previous_dataset,
                base_model=base_model,
                config_path=config_path
            )
            results_summary.append(result)
            
            # Brief pause between trainings
            if i < len(dataset_numbers) - 1:  # Not the last dataset
                print(f"\nPAUSE  Brief pause before next dataset...")
                time.sleep(5)
            
        except KeyboardInterrupt:
            print(f"\n\nSTOP Training interrupted by user!")
            print(f"CHART Completed datasets so far: {[r['dataset'] for r in results_summary if r['success']]}")
            break
        except Exception as e:
            print(f"\nERROR Unexpected error: {e}")
            continue
    
    # Print final summary
    total_time = time.time() - overall_start_time
    total_hours = total_time // 3600
    total_minutes = (total_time % 3600) // 60
    
    print(f"\n{'='*80}")
    print(f" TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"CHART Run Name: {run_name}")
    print(f"PROCESSING Training Mode: {'Progressive' if progressive_mode else 'Normal'}")
    print(f"MODEL Base Model: {Path(base_model).name}")
    print(f"SETTINGS  Configuration: {config_path if config_path else 'YOLO defaults'}")
    print(f"TIME  Total time: {int(total_hours)}h {int(total_minutes)}m")
    print(f"CHART Datasets processed: {len(results_summary)}")
    
    successful_trainings = [r for r in results_summary if r['success']]
    failed_trainings = [r for r in results_summary if not r['success']]
    
    print(f"SUCCESS Successful: {len(successful_trainings)}")
    print(f"ERROR Failed: {len(failed_trainings)}")
    
    if successful_trainings:
        print(f"\nMETRICS PERFORMANCE SUMMARY:")
        print(f"{'Dataset':<10} {'mAP50':<10} {'mAP50-95':<12} {'Time (min)':<12} {'Model Used':<20}")
        print(f"{'-'*70}")
        
        for result in successful_trainings:
            dataset = result['dataset']
            map50 = f"{result['map50']:.3f}" if result['map50'] else "N/A"
            map50_95 = f"{result['map50_95']:.3f}" if result['map50_95'] else "N/A"
            time_min = f"{result['training_time']/60:.0f}"
            model_used = result.get('model_used', 'unknown').split('/')[-1]  # Get filename only
            print(f"{dataset:<10} {map50:<10} {map50_95:<12} {time_min:<12} {model_used:<20}")
        
        # Identify best performing dataset
        valid_results = [r for r in successful_trainings if r['map50'] is not None]
        if valid_results:
            best_dataset = max(valid_results, key=lambda x: x['map50'])
            print(f"\nBEST Best performing dataset: Dataset {best_dataset['dataset']} (mAP50: {best_dataset['map50']:.3f})")
    
    if failed_trainings:
        print(f"\nERROR FAILED TRAININGS:")
        for result in failed_trainings:
            print(f"   Dataset {result['dataset']}: {result.get('error', 'Unknown error')}")
    
    print(f"\nFOLDER All results saved in: runs/detect/{run_name}_dataset_X/")
    print(f"SEARCH Check individual result folders for detailed metrics and visualizations")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
