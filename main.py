#!/usr/bin/env python3
"""
Main Program - Data Processing Toolkit
A unified interface for various data processing operations including:
1. Format conversion (Supervisely to YOLO, etc.)
2. Train/evaluate datasets
3. Video operations (split video, remove watermark)
"""

import os
import sys
from pathlib import Path
import json

# Import the existing functionality
from create_individual_datasets import IndividualDatasetCreator, get_folder_identifiers

# Import training functionality
try:
    from train_datasets import (
        check_config_file, get_available_models, get_model_selection, 
        get_training_mode, get_report_name, train_single_dataset,
        SettingsManager as TrainingSettingsManager, manage_settings,
        get_dataset_folder, list_available_datasets, get_shuffle_option
    )
    TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Training functionality not available: {e}")
    TRAINING_AVAILABLE = False

# Import evaluation functionality
try:
    from quick_results_summary import get_final_metrics, classify_difficulty
    import pandas as pd
    EVALUATION_AVAILABLE = True
except ImportError as e:
    print(f"WARNING  Evaluation functionality not available: {e}")
    EVALUATION_AVAILABLE = False

# Import video processing functionality
try:
    from video_splitter import main as video_splitter_main
    VIDEO_SPLITTER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING  Video splitter functionality not available: {e}")
    VIDEO_SPLITTER_AVAILABLE = False

try:
    from watermark_remover import main as watermark_remover_main
    WATERMARK_REMOVER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING  Watermark remover functionality not available: {e}")
    WATERMARK_REMOVER_AVAILABLE = False

# Import dataset merger functionality
try:
    from dataset_merger import DatasetMerger
    DATASET_MERGER_AVAILABLE = True
except ImportError as e:
    print(f"WARNING  Dataset merger functionality not available: {e}")
    DATASET_MERGER_AVAILABLE = False

class SettingsManager:
    def __init__(self, settings_file="settings.txt"):
        self.settings_file = Path(settings_file)
        self.default_settings = {
            "default_input_folder": "dataset_supervisely_format",
            "default_output_folder": "dataset_yolo_format",
            "default_results_folder": "runs/detect"
        }
        self.settings = self.load_settings()
    
    def load_settings(self):
        """Load settings from file, create with defaults if doesn't exist"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                # Ensure all required keys exist
                for key, value in self.default_settings.items():
                    if key not in settings:
                        settings[key] = value
                return settings
            except (json.JSONDecodeError, IOError) as e:
                print(f"WARNING  Error loading settings: {e}")
                print("Using default settings.")
                return self.default_settings.copy()
        else:
            # Create settings file with defaults
            self.save_settings(self.default_settings)
            return self.default_settings.copy()
    
    def save_settings(self, settings=None):
        """Save settings to file"""
        if settings is None:
            settings = self.settings
        
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"ERROR Error saving settings: {e}")
            return False
    
    def get_setting(self, key):
        """Get a specific setting value"""
        return self.settings.get(key, self.default_settings.get(key))
    
    def set_setting(self, key, value):
        """Set a specific setting value and save"""
        self.settings[key] = value
        return self.save_settings()
    
    def get_default_input_folder(self):
        """Get default input folder"""
        return self.get_setting("default_input_folder")
    
    def get_default_output_folder(self):
        """Get default output folder"""
        return self.get_setting("default_output_folder")
    
    def set_default_input_folder(self, folder):
        """Set default input folder"""
        return self.set_setting("default_input_folder", folder)
    
    def set_default_output_folder(self, folder):
        """Set default output folder"""
        return self.set_setting("default_output_folder", folder)
    
    def get_default_results_folder(self):
        """Get default results folder"""
        return self.get_setting("default_results_folder")
    
    def set_default_results_folder(self, folder):
        """Set default results folder"""
        return self.set_setting("default_results_folder", folder)

class DataProcessingToolkit:
    def __init__(self):
        self.current_dir = Path.cwd()
        self.settings_manager = SettingsManager()
        
    def show_main_menu(self):
        """Display the main menu options"""
        print("\n" + "="*60)
        print("TOOLS - Data Processing Toolkit")
        print("="*60)
        print("Choose an operation:")
        print("1. Convert format (Supervisely → YOLO → Darknet)")
        print("2. Train/evaluate datasets")
        print("3. Video operations (split video/remove watermark)")
        print("4. Merge YOLO datasets")
        print("5. Exit")
        print("="*60)
        
    def get_user_choice(self):
        """Get and validate user menu choice"""
        while True:
            try:
                choice = int(input("Enter your choice (1-5): ").strip())
                if 1 <= choice <= 5:
                    return choice
                else:
                    print("ERROR Please choose a number between 1 and 5.")
            except ValueError:
                print("ERROR Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nEXIT Goodbye!")
                sys.exit(0)
    
    def format_conversion_menu(self):
        """Handle format conversion operations"""
        while True:
            print("\n" + "="*50)
            print("DOCUMENT FORMAT CONVERSION")
            print("="*50)
            print("Available conversions:")
            print("1. Supervisely → YOLO (Individual datasets)")
            print("2. YOLO → Darknet (Coming soon)")
            print("3. Darknet → YOLO (Coming soon)")
            print("4. SETTINGS  Settings (Default folders)")
            print("5. ← Back to main menu")
            
            try:
                choice = int(input("\nChoose conversion type (1-5): ").strip())
                if choice == 1:
                    self.supervisely_to_yolo()
                    break
                elif choice == 2:
                    print(" YOLO → Darknet conversion coming soon!")
                    break
                elif choice == 3:
                    print(" Darknet → YOLO conversion coming soon!")
                    break
                elif choice == 4:
                    self.format_conversion_settings()
                    # Continue to show the format conversion menu again
                elif choice == 5:
                    break
                else:
                    print("ERROR Please choose a number between 1 and 5.")
            except ValueError:
                print("ERROR Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nEXIT Returning to main menu...")
                break
    
    def format_conversion_settings(self):
        """Handle format conversion settings"""
        print("\n" + "="*50)
        print("SETTINGS  FORMAT CONVERSION SETTINGS")
        print("="*50)
        
        # Show current settings
        current_input = self.settings_manager.get_default_input_folder()
        current_output = self.settings_manager.get_default_output_folder()
        
        print(f"Current settings:")
        print(f"FOLDER Default input folder:  {current_input}")
        print(f"FOLDER Default output folder: {current_output}")
        
        print("\nOptions:")
        print("1. Change default input folder")
        print("2. Change default output folder")
        print("3. Reset to defaults")
        print("4. ← Back to format conversion menu")
        
        while True:
            try:
                choice = int(input("\nChoose option (1-4): ").strip())
                if choice == 1:
                    self.change_default_input_folder()
                    break
                elif choice == 2:
                    self.change_default_output_folder()
                    break
                elif choice == 3:
                    self.reset_settings_to_defaults()
                    break
                elif choice == 4:
                    break
                else:
                    print("ERROR Please choose a number between 1 and 4.")
            except ValueError:
                print("ERROR Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nEXIT Returning to format conversion menu...")
                break
    
    def change_default_input_folder(self):
        """Change the default input folder"""
        current = self.settings_manager.get_default_input_folder()
        print(f"\nFOLDER Current default input folder: {current}")
        
        new_folder = input("Enter new default input folder path: ").strip()
        if not new_folder:
            print("ERROR No folder specified. Keeping current setting.")
            return
        
        # Optional: Check if folder exists and warn if it doesn't
        if not Path(new_folder).exists():
            confirm = input(f"WARNING  Folder '{new_folder}' doesn't exist. Set anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                print("ERROR Operation cancelled.")
                return
        
        if self.settings_manager.set_default_input_folder(new_folder):
            print(f"SUCCESS Default input folder updated to: {new_folder}")
        else:
            print("ERROR Failed to save settings.")
    
    def change_default_output_folder(self):
        """Change the default output folder"""
        current = self.settings_manager.get_default_output_folder()
        print(f"\nFOLDER Current default output folder: {current}")
        
        new_folder = input("Enter new default output folder path: ").strip()
        if not new_folder:
            print("ERROR No folder specified. Keeping current setting.")
            return
        
        if self.settings_manager.set_default_output_folder(new_folder):
            print(f"SUCCESS Default output folder updated to: {new_folder}")
        else:
            print("ERROR Failed to save settings.")
    
    def reset_settings_to_defaults(self):
        """Reset settings to default values"""
        confirm = input("WARNING  Reset all settings to defaults? (y/N): ").strip().lower()
        if confirm == 'y':
            self.settings_manager.set_default_input_folder("dataset_supervisely_format")
            self.settings_manager.set_default_output_folder("dataset_yolo_format")
            print("SUCCESS Settings reset to defaults:")
            print("FOLDER Input folder:  dataset_supervisely_format")
            print("FOLDER Output folder: dataset_yolo_format")
        else:
            print("ERROR Reset cancelled.")
    
    def supervisely_to_yolo(self):
        """Convert Supervisely format to YOLO format using existing functionality"""
        print("\n" + "="*50)
        print("DOCUMENT SUPERVISELY → YOLO CONVERSION")
        print("="*50)
        
        # Get input directory using settings
        default_input = self.settings_manager.get_default_input_folder()
        input_dir = input(f"FOLDER Provide folder path where datasets exist (default: {default_input}): ").strip()
        if not input_dir:
            input_dir = default_input
            
        # Check if input directory exists
        if not Path(input_dir).exists():
            print(f"ERROR Input directory '{input_dir}' not found!")
            print("Please make sure the directory exists and try again.")
            print("TIP Tip: Use option 4 (Settings) to change default folders.")
            return
            
        # Get output directory using settings
        default_output = self.settings_manager.get_default_output_folder()
        output_dir = input(f"Enter output directory (default: {default_output}): ").strip()
        if not output_dir:
            output_dir = default_output
            
        print(f"\nFOLDER Input directory: {input_dir}")
        print(f"FOLDER Output directory: {output_dir}")
        
        # Get folder identifiers to process
        print("\nLIST Select folders to convert:")
        folder_identifiers = get_folder_identifiers()
        
        if folder_identifiers is None:
            print("ERROR Operation cancelled.")
            return
            
        # Create the dataset creator and process
        try:
            print(f"\nLAUNCH Starting conversion...")
            creator = IndividualDatasetCreator(input_dir, output_dir)
            successful_datasets = creator.create_selected_datasets(folder_identifiers)
            
            if successful_datasets:
                print(f"\nCOMPLETE Conversion completed successfully!")
                print(f"SUCCESS Created {len(successful_datasets)} YOLO dataset(s)")
                print(f"FOLDER Output location: {Path(output_dir).absolute()}")
            else:
                print(f"\nERROR No datasets were created successfully.")
                
        except Exception as e:
            print(f"\nERROR Error during conversion: {e}")
            print("Please check your input data and try again.")
    
    def train_evaluate_menu(self):
        """Handle training and evaluation operations"""
        if not TRAINING_AVAILABLE:
            print("\n" + "="*50)
            print("TRAINING  TRAIN/EVALUATE DATASETS")
            print("="*50)
            print("ERROR Training functionality is not available!")
            print("Please ensure you have:")
            print("  • ultralytics package installed")
            print("  • train_datasets.py in the same directory")
            print("  • PyTorch installed with CUDA support (for GPU training)")
            input("\nPress Enter to continue...")
            return
            
        while True:
            print("\n" + "="*50)
            print("TRAINING  TRAIN/EVALUATE DATASETS")
            print("="*50)
            print("Available operations:")
            print("1. Train YOLO datasets")
            print("2. Evaluate trained models")
            print("3. Quick results summary")
            print("4. Compare model performance (Coming soon)")
            print("5. SETTINGS  Training settings")
            print("6. ← Back to main menu")
            
            try:
                choice = int(input("\nChoose operation (1-6): ").strip())
                if choice == 1:
                    self.train_yolo_datasets()
                    break
                elif choice == 2:
                    self.evaluate_trained_models()
                elif choice == 3:
                    self.quick_results_summary()
                elif choice == 4:
                    print(" Model comparison coming soon!")
                    break
                elif choice == 5:
                    self.manage_training_settings()
                elif choice == 6:
                    break
                else:
                    print("ERROR Please choose a number between 1 and 6.")
            except ValueError:
                print("ERROR Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nEXIT Returning to main menu...")
                break
    
    def train_yolo_datasets(self):
        """Train YOLO datasets using existing functionality"""
        print("\n" + "="*50)
        print("TRAINING  YOLO DATASET TRAINING")
        print("="*50)
        
        try:
            import time
            import datetime
            
            # Check for training configuration file
            config_path = check_config_file()
            
            # Get dataset folder from user input
            dataset_folder = get_dataset_folder()
            if dataset_folder is None:
                print("ERROR Dataset folder selection cancelled.")
                return
            
            # Get user inputs using enhanced dataset selection
            print("CHART Select datasets to train:")
            dataset_identifiers = self.get_dataset_identifiers_for_training(dataset_folder)
            if dataset_identifiers is None:
                print("ERROR Operation cancelled.")
                return
                
            print("\nPROCESSING Choose training mode:")
            progressive_mode = get_training_mode()
            if progressive_mode is None:
                print("ERROR Operation cancelled.")
                return
                
            print("\nMODEL Select base model:")
            base_model = get_model_selection()
            if base_model is None:
                print("ERROR Operation cancelled.")
                return
                
            print("\nFOLDER Set training run name:")
            run_name = get_report_name()
            if run_name is None:
                print("ERROR Operation cancelled.")
                return
            
            # Get shuffle option
            shuffle = get_shuffle_option()
            if shuffle is None:
                print("ERROR Shuffle option selection cancelled.")
                return
            
            # Display training plan
            print(f"\nLIST TRAINING PLAN:")
            print(f"CHART Datasets to train: {', '.join(map(str, dataset_identifiers))}")
            print(f"FOLDER Dataset folder: {dataset_folder}")
            print(f"PROCESSING Progressive training: {'Yes' if progressive_mode else 'No'}")
            print(f"MODEL Base model: {Path(base_model).name}")
            print(f"SETTINGS  Configuration: {config_path if config_path else 'YOLO defaults'}")
            print(f"SETTINGS  Shuffle: {shuffle}")
            print(f"FOLDER Results will be saved in: runs/detect/{run_name}/")
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
            
            # Start training process
            print(f"\nLAUNCH Starting training process...")
            print(f"TIP You can stop anytime with Ctrl+C")
            
            results_summary = []
            overall_start_time = time.time()
            
            for i, dataset_identifier in enumerate(dataset_identifiers):
                try:
                    # Check if dataset exists in the default folder location
                    dataset_identifier_str = str(dataset_identifier)
                    dataset_path = None
                    
                    # Try multiple naming conventions in the default folder
                    # Try with dataset_ prefix
                    test_path = Path(f"{dataset_folder}/dataset_{dataset_identifier_str}/dataset.yaml")
                    if test_path.exists():
                        dataset_path = test_path
                    else:
                        # Try without dataset_ prefix
                        test_path = Path(f"{dataset_folder}/{dataset_identifier_str}/dataset.yaml")
                        if test_path.exists():
                            dataset_path = test_path
                        # If identifier already has dataset_ prefix, try as-is
                        elif dataset_identifier_str.startswith('dataset_'):
                            test_path = Path(f"{dataset_folder}/{dataset_identifier_str}/dataset.yaml")
                            if test_path.exists():
                                dataset_path = test_path
                    
                    if not dataset_path or not dataset_path.exists():
                        print(f"ERROR Dataset {dataset_identifier} not found in {dataset_folder}")
                        continue
                    
                    # For progressive training, use the previous dataset identifier (if available)
                    previous_dataset = dataset_identifiers[i-1] if progressive_mode and i > 0 else None
                    
                    result = train_single_dataset(
                        dataset_num=dataset_identifier,
                        progressive_mode=progressive_mode,
                        run_name=run_name,
                        previous_dataset_num=previous_dataset,
                        base_model=base_model,
                        config_path=config_path,
                        dataset_folder=dataset_folder,
                        shuffle=shuffle
                    )
                    results_summary.append(result)
                    
                    # Brief pause between trainings
                    if i < len(dataset_identifiers) - 1:  # Not the last dataset
                        print(f"\nPAUSE  Brief pause before next dataset...")
                        time.sleep(5)
                    
                except KeyboardInterrupt:
                    print(f"\n\nSTOP Training interrupted by user!")
                    print(f"CHART Completed datasets so far: {[r['dataset'] for r in results_summary if r['success']]}")
                    break
                except Exception as e:
                    print(f"\nERROR Unexpected error training dataset {dataset_identifier}: {e}")
                    continue
            
            # Print final summary
            self.print_training_summary(results_summary, run_name, progressive_mode, base_model, config_path, overall_start_time)
            
        except Exception as e:
            print(f"\nERROR Error during training setup: {e}")
            print("Please check your setup and try again.")
    
    def print_training_summary(self, results_summary, run_name, progressive_mode, base_model, config_path, overall_start_time):
        """Print detailed training summary"""
        import time
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
        
        print(f"\nFOLDER All results saved in: runs/detect/{run_name}/")
        print(f"SEARCH Check individual result folders for detailed metrics and visualizations")
        print(f"{'='*80}")
        
        input("\nPress Enter to continue...")
    
    def manage_training_settings(self):
        """Manage training settings"""
        if not TRAINING_AVAILABLE:
            print("\nERROR Training functionality is not available!")
            print("Settings require train_datasets.py to be available.")
            input("\nPress Enter to continue...")
            return
        
        # Create a training settings manager (use the one from train_datasets)
        training_settings = TrainingSettingsManager()
        
        # Use the manage_settings function from train_datasets
        manage_settings(training_settings)
    
    def get_dataset_identifiers_for_training(self, dataset_folder='dataset_yolo_format'):
        """Get dataset identifiers (numbers or names) for training from user input"""
        # First, list all available datasets
        available_datasets = list_available_datasets(dataset_folder)
        
        if not available_datasets:
            print(f"ERROR No datasets found in {dataset_folder}")
            return None
        
        print("\n" + "="*60)
        print("AVAILABLE DATASETS")
        print("="*60)
        for i, dataset in enumerate(available_datasets, 1):
            # Display both the identifier and the folder name
            if isinstance(dataset, int):
                folder_name = f"dataset_{dataset}"
            else:
                folder_name = str(dataset)
            
            print(f"  {i}. {dataset} (folder: {folder_name})")
        print("="*60)
        print(f"\nTIP You can type 'all' to train all {len(available_datasets)} datasets")
        print("TIP Or enter specific dataset names/numbers separated by commas")
        
        while True:
            try:
                user_input = input("\nEnter dataset names/numbers (or 'all' for all datasets): ").strip()
                if not user_input:
                    print("ERROR Please enter at least one dataset identifier or 'all'.")
                    continue
                
                # Check if user wants all datasets
                if user_input.lower() == 'all':
                    print(f"SUCCESS Selected all {len(available_datasets)} datasets")
                    return available_datasets
                
                # Parse comma-separated identifiers (numbers or strings)
                dataset_identifiers = []
                for identifier in user_input.split(','):
                    identifier = identifier.strip()
                    if identifier:
                        # Try to convert to int if it's a number, otherwise keep as string
                        try:
                            dataset_identifiers.append(int(identifier))
                        except ValueError:
                            dataset_identifiers.append(identifier)
                
                if not dataset_identifiers:
                    print("ERROR Please enter valid dataset identifiers.")
                    continue
                    
                # Remove duplicates while preserving order and mixed types
                seen = set()
                unique_identifiers = []
                for item in dataset_identifiers:
                    if item not in seen:
                        seen.add(item)
                        unique_identifiers.append(item)
                
                # Validate that datasets exist in the provided folder location
                valid_identifiers = []
                for identifier in unique_identifiers:
                    # Convert identifier to string for path operations
                    identifier_str = str(identifier)
                    found = False
                    
                    # Check in the default folder location
                    # If identifier already starts with 'dataset_', don't add it again
                    if identifier_str.startswith('dataset_'):
                        # Try as-is first
                        dataset_path = Path(f"{dataset_folder}/{identifier_str}")
                        if dataset_path.exists():
                            valid_identifiers.append(identifier)
                            found = True
                        else:
                            # Also try without the prefix (in case folder name doesn't have dataset_)
                            dataset_path_no_prefix = Path(f"{dataset_folder}/{identifier_str.replace('dataset_', '', 1)}")
                            if dataset_path_no_prefix.exists():
                                valid_identifiers.append(identifier)
                                found = True
                    else:
                        # Try with dataset_ prefix first (for numbered datasets like 1, 2, 3)
                        dataset_path = Path(f"{dataset_folder}/dataset_{identifier_str}")
                        if dataset_path.exists():
                            valid_identifiers.append(identifier)
                            found = True
                        else:
                            # Try without dataset_ prefix (for merged datasets, etc.)
                            dataset_path_no_prefix = Path(f"{dataset_folder}/{identifier_str}")
                            if dataset_path_no_prefix.exists():
                                valid_identifiers.append(identifier)
                                found = True
                    
                    if not found:
                        print(f"WARNING  Dataset '{identifier_str}' not found in {dataset_folder}/")
                
                if not valid_identifiers:
                    print("ERROR No valid datasets found. Please check your dataset names/numbers.")
                    continue
                
                print(f"SUCCESS Selected datasets: {', '.join(map(str, valid_identifiers))}")
                if len(valid_identifiers) != len(unique_identifiers):
                    print(f" Note: Only {len(valid_identifiers)} out of {len(unique_identifiers)} datasets were found and will be used.")
                
                return valid_identifiers
                
            except KeyboardInterrupt:
                print("\nERROR Operation cancelled by user.")
                return None
    
    def evaluate_trained_models(self):
        """Evaluate individual trained models with visualization"""
        if not EVALUATION_AVAILABLE:
            print("\n" + "="*50)
            print("CHART MODEL EVALUATION")
            print("="*50)
            print("ERROR Evaluation functionality is not available!")
            print("Please ensure you have:")
            print("  • pandas package installed (pip install pandas)")
            print("  • quick_results_summary.py in the same directory")
            input("\nPress Enter to continue...")
            return
            
        print("\n" + "="*50)
        print("CHART MODEL EVALUATION")
        print("="*50)
        
        # Get available training results
        runs_dir = Path("runs/detect")
        if not runs_dir.exists():
            print("ERROR No training results found!")
            print("Please train some models first using option 1.")
            input("\nPress Enter to continue...")
            return
            
        # Find all training result directories
        result_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "results.csv").exists()]
        
        if not result_dirs:
            print("ERROR No training results with results.csv found!")
            print("Please train some models first using option 1.")
            input("\nPress Enter to continue...")
            return
            
        # Let user choose a model
        print("FOLDER Available trained models:")
        for i, result_dir in enumerate(result_dirs, 1):
            print(f"  {i}. {result_dir.name}")
            
        try:
            choice = int(input(f"\nSelect model to evaluate (1-{len(result_dirs)}): ").strip())
            if not (1 <= choice <= len(result_dirs)):
                print("ERROR Invalid choice.")
                input("\nPress Enter to continue...")
                return
                
            selected_dir = result_dirs[choice - 1]
            self.show_model_evaluation(selected_dir)
            
        except ValueError:
            print("ERROR Please enter a valid number.")
            input("\nPress Enter to continue...")
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled.")
    
    def show_model_evaluation(self, model_dir):
        """Show detailed evaluation for a specific model"""
        print(f"\nCHART EVALUATING: {model_dir.name}")
        print("=" * 60)
        
        # Read and display basic metrics
        results_file = model_dir / "results.csv"
        try:
            df = pd.read_csv(results_file)
            final_row = df.iloc[-1]
            
            print("METRICS FINAL METRICS:")
            print(f"  mAP50: {final_row.get('metrics/mAP50(B)', 0):.4f}")
            print(f"  mAP50-95: {final_row.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"  Precision: {final_row.get('metrics/precision(B)', 0):.4f}")
            print(f"  Recall: {final_row.get('metrics/recall(B)', 0):.4f}")
            print(f"  Total Epochs: {len(df)}")
            print(f"  Best mAP50: {df['metrics/mAP50(B)'].max():.4f} (Epoch {df['metrics/mAP50(B)'].idxmax()})")
            
        except Exception as e:
            print(f"ERROR Error reading results: {e}")
            return
        
        # Show available visualizations
        while True:
            print(f"\nVISUALIZATIONS AVAILABLE VISUALIZATIONS:")
            visualizations = [
                ("confusion_matrix.png", "1. Confusion Matrix"),
                ("confusion_matrix_normalized.png", "2. Confusion Matrix (Normalized)"),
                ("F1_curve.png", "3. F1 Curve"),
                ("labels.jpg", "4. Labels Distribution"),
                ("P_curve.png", "5. Precision Curve"),
                ("PR_curve.png", "6. Precision-Recall Curve"),
                ("R_curve.png", "7. Recall Curve"),
                ("results.png", "8. Results Summary")
            ]
            
            available_viz = []
            for filename, description in visualizations:
                if (model_dir / filename).exists():
                    available_viz.append((filename, description))
                    print(f"  {description}")
                    
            if not available_viz:
                print("ERROR No visualization files found!")
                input("\nPress Enter to continue...")
                return
                
            print("  9. ← Back to model selection")
            
            try:
                viz_choice = int(input(f"\nSelect visualization (1-9): ").strip())
                
                if viz_choice == 9:
                    break
                elif 1 <= viz_choice <= 8:
                    # Find the corresponding visualization
                    viz_index = viz_choice - 1
                    if viz_index < len(available_viz):
                        filename, description = available_viz[viz_index]
                        self.display_visualization(model_dir / filename, description)
                    else:
                        print("ERROR Visualization not available.")
                else:
                    print("ERROR Please choose a number between 1 and 9.")
                    
            except ValueError:
                print("ERROR Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nEXIT Returning to model selection...")
                break
    
    def display_visualization(self, viz_path, description):
        """Display a visualization file"""
        import subprocess
        import platform
        
        print(f"\nIMAGE  Opening: {description}")
        print(f"FOLDER File: {viz_path}")
        
        try:
            # Open the image with the default system viewer
            system = platform.system()
            if system == "Windows":
                subprocess.run(["start", str(viz_path)], shell=True, check=True)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(viz_path)], check=True)
            else:  # Linux
                subprocess.run(["xdg-open", str(viz_path)], check=True)
                
            print("SUCCESS Visualization opened in default viewer.")
            
        except subprocess.CalledProcessError:
            print(f"ERROR Could not open visualization automatically.")
            print(f"Please manually open: {viz_path}")
        except Exception as e:
            print(f"ERROR Error opening visualization: {e}")
            
        input("\nPress Enter to continue...")
    
    def quick_results_summary(self):
        """Show quick summary of all training results"""
        if not EVALUATION_AVAILABLE:
            print("\nERROR Evaluation functionality not available!")
            input("\nPress Enter to continue...")
            return
            
        print("\n" + "="*50)
        print("CHART QUICK RESULTS SUMMARY")
        print("="*50)
        
        try:
            # Import and run the quick results summary
            from quick_results_summary import main as quick_summary_main
            quick_summary_main()
        except Exception as e:
            print(f"ERROR Error running quick results summary: {e}")
            
        input("\nPress Enter to continue...")
    
    def video_operations_menu(self):
        """Handle video-related operations"""
        while True:
            print("\n" + "="*50)
            print("VIDEO VIDEO OPERATIONS")
            print("="*50)
            print("Available operations:")
            print("1. Remove watermark from videos")
            print("2. Split videos into chunks")
            print("3. Video format conversion (Coming soon)")
            print("4. ← Back to main menu")
            
            try:
                choice = int(input("\nChoose operation (1-4): ").strip())
                if choice == 1:
                    self.remove_watermark()
                elif choice == 2:
                    self.split_videos()
                elif choice == 3:
                    print(" Video format conversion coming soon!")
                    input("\nPress Enter to continue...")
                elif choice == 4:
                    break
                else:
                    print("ERROR Please choose a number between 1 and 4.")
            except ValueError:
                print("ERROR Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nEXIT Returning to main menu...")
                break
    
    def remove_watermark(self):
        """Remove watermark from videos using existing functionality"""
        print("\n" + "="*50)
        print("VIDEO WATERMARK REMOVAL")
        print("="*50)
        
        if not WATERMARK_REMOVER_AVAILABLE:
            print("ERROR Watermark remover functionality is not available!")
            print("Please ensure you have:")
            print("  • opencv-python package installed (pip install opencv-python)")
            print("  • numpy package installed (pip install numpy)")
            print("  • watermark_remover.py in the same directory")
            input("\nPress Enter to continue...")
            return
        
        print("LAUNCH Launching watermark remover...")
        
        try:
            watermark_remover_main()
        except Exception as e:
            print(f"ERROR Error running watermark remover: {e}")
            print("Please check your setup and try again.")
        
        input("\nPress Enter to continue...")
    
    def split_videos(self):
        """Split videos into chunks using existing functionality"""
        print("\n" + "="*50)
        print("VIDEO VIDEO SPLITTER")
        print("="*50)
        
        if not VIDEO_SPLITTER_AVAILABLE:
            print("ERROR Video splitter functionality is not available!")
            print("Please ensure you have:")
            print("  • FFmpeg installed and in PATH")
            print("  • video_splitter.py in the same directory")
            print("\nTo install FFmpeg:")
            print("  • Download from: https://www.gyan.dev/ffmpeg/builds/")
            print("  • Extract to C:\\ffmpeg and add C:\\ffmpeg\\bin to PATH")
            print("  • Or use: winget install FFmpeg")
            input("\nPress Enter to continue...")
            return
        
        # Check if FFmpeg is available
        ffmpeg_available = self.check_ffmpeg_availability()
        if not ffmpeg_available:
            print("ERROR FFmpeg not found!")
            print("\nSETUP Quick Setup Options:")
            print("1. Download FFmpeg binaries to ./ffmpeg_bin/ folder")
            print("2. Add FFmpeg to system PATH")
            print("3. Install using winget")
            
            choice = input("\nWould you like setup instructions? (y/n): ").strip().lower()
            if choice == 'y':
                self.show_ffmpeg_setup_instructions()
            input("\nPress Enter to continue...")
            return
        
        print("LAUNCH Launching video splitter...")
        print("TIP This tool splits videos larger than 100MB into smaller chunks")
        
        try:
            video_splitter_main()
        except Exception as e:
            print(f"ERROR Error running video splitter: {e}")
            print("Please check your setup and try again.")
        
        input("\nPress Enter to continue...")
    
    def check_ffmpeg_availability(self):
        """Check if FFmpeg is available in PATH or local directory"""
        import subprocess
        
        # Check system PATH first
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("SUCCESS FFmpeg found in system PATH")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Check local ffmpeg_bin directory
        local_ffmpeg = Path("ffmpeg_bin") / "ffmpeg.exe"
        local_ffprobe = Path("ffmpeg_bin") / "ffprobe.exe"
        
        if local_ffmpeg.exists():
            if not local_ffprobe.exists():
                print(f"ERROR FFmpeg found but ffprobe.exe is missing!")
                print(f"DOWNLOAD Please download the complete FFmpeg package that includes both:")
                print(f"   • ffmpeg.exe SUCCESS (found)")
                print(f"   • ffprobe.exe ERROR (missing)")
                return False
                
            try:
                # Test both executables
                subprocess.run([str(local_ffmpeg), '-version'], capture_output=True, check=True)
                subprocess.run([str(local_ffprobe), '-version'], capture_output=True, check=True)
                print(f"SUCCESS FFmpeg and ffprobe found at: {Path('ffmpeg_bin').absolute()}")
                
                # Add local path to environment for this session
                import os
                current_path = os.environ.get('PATH', '')
                ffmpeg_dir = str(Path("ffmpeg_bin").absolute())
                if ffmpeg_dir not in current_path:
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
                    print(f"SUCCESS Added {ffmpeg_dir} to PATH for this session")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"ERROR Local FFmpeg/ffprobe found but not working: {e}")
                return False
            
        return False
    
    def show_ffmpeg_setup_instructions(self):
        """Show detailed FFmpeg setup instructions"""
        print("\n" + "="*60)
        print("SETUP FFMPEG SETUP INSTRUCTIONS")
        print("="*60)
        
        print("DOWNLOAD OPTION 1: Download Pre-built Binaries (Recommended)")
        print("1. Visit: https://www.gyan.dev/ffmpeg/builds/")
        print("2. Download: 'ffmpeg-release-essentials.zip'")
        print("3. Extract the zip file")
        print("4. Copy ffmpeg.exe to your ./ffmpeg_bin/ folder")
        print("   (or extract to C:\\ffmpeg and add C:\\ffmpeg\\bin to PATH)")
        
        print(f"\nFOLDER Local Setup (Current Directory):")
        print(f"   Create folder: {Path.cwd() / 'ffmpeg_bin'}")
        print(f"   Place ffmpeg.exe in: {Path.cwd() / 'ffmpeg_bin' / 'ffmpeg.exe'}")
        
        print(f"\nSYSTEM System-wide Setup:")
        print("1. Extract FFmpeg to C:\\ffmpeg")
        print("2. Add C:\\ffmpeg\\bin to your system PATH:")
        print("   - Press Win+R, type 'sysdm.cpl', press Enter")
        print("   - Click 'Environment Variables'")
        print("   - Edit 'Path' and add 'C:\\ffmpeg\\bin'")
        
        print(f"\nQUICK Quick Install (if you have winget):")
        print("   winget install FFmpeg")
        
        print("\nTIP After setup, restart this program to use video splitting!")
        print("="*60)
    
    def merge_yolo_datasets(self):
        """Merge multiple YOLO datasets using existing functionality"""
        print("\n" + "="*50)
        print("MERGE YOLO DATASET MERGER")
        print("="*50)
        
        if not DATASET_MERGER_AVAILABLE:
            print("ERROR Dataset merger functionality is not available!")
            print("Please ensure you have:")
            print("  • PyYAML package installed (pip install pyyaml)")
            print("  • Pillow package installed (pip install pillow)")
            print("  • dataset_merger.py in the same directory")
            input("\nPress Enter to continue...")
            return
        
        print("LAUNCH Launching dataset merger...")
        print("TIP This tool merges multiple YOLO datasets into a single dataset")
        
        try:
            merger = DatasetMerger()
            merger.merge_datasets()
        except Exception as e:
            print(f"ERROR Error running dataset merger: {e}")
            print("Please check your setup and try again.")
        
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main program loop"""
        print("LAUNCH Welcome to the Data Processing Toolkit!")
        
        while True:
            try:
                self.show_main_menu()
                choice = self.get_user_choice()
                
                if choice == 1:
                    self.format_conversion_menu()
                elif choice == 2:
                    self.train_evaluate_menu()
                elif choice == 3:
                    self.video_operations_menu()
                elif choice == 4:
                    self.merge_yolo_datasets()
                elif choice == 5:
                    print("\nEXIT Thank you for using the Data Processing Toolkit!")
                    print("Goodbye! COMPLETE")
                    break
                    
            except KeyboardInterrupt:
                print("\n\nEXIT Thank you for using the Data Processing Toolkit!")
                print("Goodbye! COMPLETE")
                break
            except Exception as e:
                print(f"\nERROR Unexpected error: {e}")
                print("Please try again or contact support.")

def main():
    """Entry point of the program"""
    toolkit = DataProcessingToolkit()
    toolkit.run()

if __name__ == "__main__":
    main()
