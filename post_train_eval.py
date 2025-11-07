#!/usr/bin/env python3
"""
Post-Training Evaluation - Analyze training results for datasets
"""

import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class SettingsManager:
    """Manage settings for post-training evaluation"""
    def __init__(self, settings_file="settings/post_train_eval.json"):
        self.settings_file = Path(settings_file)
        self.default_settings = {
            "default_run_folder": "runs/detect",
            "default_output_folder": "output/post_train_eval"
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
                print(f"WARNING: Error loading settings: {e}")
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
            # Ensure settings directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            print(f"ERROR: Error saving settings: {e}")
            return False
    
    def get_run_folder(self):
        """Get default run folder"""
        return self.settings.get("default_run_folder", self.default_settings["default_run_folder"])
    
    def set_run_folder(self, folder):
        """Set default run folder"""
        self.settings["default_run_folder"] = folder
        return self.save_settings()
    
    def get_output_folder(self):
        """Get default output folder"""
        return self.settings.get("default_output_folder", self.default_settings["default_output_folder"])
    
    def set_output_folder(self, folder):
        """Set default output folder"""
        self.settings["default_output_folder"] = folder
        return self.save_settings()

def get_available_run_folders(run_folder):
    """Get all available run folders that contain results.csv"""
    run_path = Path(run_folder)
    
    if not run_path.exists():
        return []
    
    run_folders = []
    # Look for any folder that contains results.csv
    for folder in run_path.iterdir():
        if folder.is_dir() and (folder / "results.csv").exists():
            run_folders.append(folder.name)
    
    return sorted(run_folders)

def select_run_folders(available_folders):
    """Let user select which run folders to evaluate"""
    print("\nAvailable run folders:")
    for i, folder_name in enumerate(available_folders, 1):
        print(f"  {i}. {folder_name}")
    
    print("\nSelection options:")
    print("  • Enter numbers separated by commas (e.g., 1,3,5)")
    print("  • Enter 'all' or press Enter to select all folders")
    
    while True:
        try:
            user_input = input("\nSelect folders to evaluate: ").strip()
            
            if not user_input or user_input.lower() == 'all':
                return available_folders
            
            # Parse comma-separated numbers
            selected_folders = []
            for num_str in user_input.split(','):
                num_str = num_str.strip()
                if num_str:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(available_folders):
                        if available_folders[idx] not in selected_folders:
                            selected_folders.append(available_folders[idx])
                    else:
                        print(f"ERROR: Invalid number: {num_str}. Please use numbers 1-{len(available_folders)}")
                        break
            else:
                if selected_folders:
                    return selected_folders
                print("ERROR: Please select at least one folder.")
        
        except ValueError:
            print("ERROR: Please enter valid numbers separated by commas.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None

def evaluate_single_run(folder_name, run_folder):
    """Evaluate a single run folder's results"""
    results_file = Path(run_folder) / folder_name / "results.csv"
    
    print(f"  Processing: {folder_name}...", end=" ")
    
    if not results_file.exists():
        print(f"SKIP (no results.csv)")
        return None
    
    try:
        df = pd.read_csv(results_file)
        
        # Key metrics
        final_row = df.iloc[-1]
        best_map50 = df['metrics/mAP50(B)'].max()
        best_epoch = df['metrics/mAP50(B)'].idxmax()
        
        print(f"OK (mAP50: {best_map50:.4f})")
        
        return {
            'folder': folder_name,
            'final_map50': final_row.get('metrics/mAP50(B)', 0),
            'final_map50_95': final_row.get('metrics/mAP50-95(B)', 0),
            'best_map50': best_map50,
            'best_epoch': best_epoch,
            'total_epochs': len(df),
            'final_precision': final_row.get('metrics/precision(B)', 0),
            'final_recall': final_row.get('metrics/recall(B)', 0)
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def classify_difficulty(map50):
    """Classify dataset difficulty based on mAP50"""
    if map50 >= 0.70:
        return "EASY", "Easy"
    elif map50 >= 0.50:
        return "MEDIUM", "Medium"
    else:
        return "HARD", "Hard"

def evaluate_runs(settings_manager):
    """Evaluate training runs for selected folders and save to CSV"""
    run_folder = settings_manager.get_run_folder()
    output_folder = settings_manager.get_output_folder()
    
    print("\n" + "="*80)
    print("EVALUATE TRAINING RUNS")
    print("="*80)
    print(f"Run folder: {run_folder}")
    print(f"Output folder: {output_folder}")
    
    # Check if run folder exists
    if not Path(run_folder).exists():
        print(f"\nERROR: Run folder not found: {run_folder}")
        print("Please change the default run folder path in settings.")
        return
    
    # Get available run folders
    available_folders = get_available_run_folders(run_folder)
    
    if not available_folders:
        print(f"\nERROR: No training results found in: {run_folder}")
        print("Make sure you have trained models with results.csv in their folders.")
        return
    
    print(f"\nFound {len(available_folders)} folder(s) with results")
    
    # Let user select folders
    selected_folders = select_run_folders(available_folders)
    
    if not selected_folders:
        return
    
    print(f"\nEvaluating {len(selected_folders)} folder(s)...")
    
    # Evaluate each folder
    all_results = []
    for folder_name in selected_folders:
        result = evaluate_single_run(folder_name, run_folder)
        if result:
            all_results.append(result)
    
    if not all_results:
        print("\nERROR: No valid results found")
        return
    
    # Add difficulty classification to results
    for result in all_results:
        difficulty_code, difficulty_name = classify_difficulty(result['best_map50'])
        result['difficulty_code'] = difficulty_code
        result['difficulty_name'] = difficulty_name
    
    # Create DataFrame
    df_results = pd.DataFrame(all_results)
    
    # Reorder columns for better readability
    column_order = [
        'folder',
        'best_map50',
        'final_map50',
        'final_map50_95',
        'final_precision',
        'final_recall',
        'best_epoch',
        'total_epochs',
        'difficulty_code',
        'difficulty_name'
    ]
    df_results = df_results[column_order]
    
    # Sort by best_map50 (descending)
    df_results = df_results.sort_values('best_map50', ascending=False)
    
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"evaluation_results_{timestamp}.csv"
    csv_filepath = output_path / csv_filename
    
    # Save to CSV
    df_results.to_csv(csv_filepath, index=False, float_format='%.4f')
    
    # Print summary to console
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Evaluated runs: {len(all_results)}")
    print(f"\nPerformance summary:")
    
    easy_count = len([r for r in all_results if r['difficulty_code'] == 'EASY'])
    medium_count = len([r for r in all_results if r['difficulty_code'] == 'MEDIUM'])
    hard_count = len([r for r in all_results if r['difficulty_code'] == 'HARD'])
    
    print(f"  EXCELLENT (mAP50 >= 0.70): {easy_count}")
    print(f"  GOOD (0.50-0.69):          {medium_count}")
    print(f"  POOR (< 0.50):             {hard_count}")
    
    # Show top 3 performers
    print(f"\nTop 3 performers:")
    for i, row in df_results.head(3).iterrows():
        print(f"  {i+1}. {row['folder']:<40} (mAP50: {row['best_map50']:.4f})")
    
    print(f"\nSUCCESS: Results saved to: {csv_filepath}")
    print("="*80)

def change_run_folder(settings_manager):
    """Change default run folder path"""
    print("\n" + "="*80)
    print("CHANGE DEFAULT RUN FOLDER")
    print("="*80)
    
    current_folder = settings_manager.get_run_folder()
    print(f"\nCurrent default run folder: {current_folder}")
    
    new_folder = input("\nEnter new run folder path (or press Enter to cancel): ").strip().strip('"').strip("'")
    
    if not new_folder:
        print("Cancelled.")
        return
    
    # Verify folder exists
    if not Path(new_folder).exists():
        print(f"\nWARNING: Folder does not exist: {new_folder}")
        confirm = input("Save anyway? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled.")
            return
    
    if settings_manager.set_run_folder(new_folder):
        print(f"\nSUCCESS: Default run folder updated to: {new_folder}")
    else:
        print("\nERROR: Failed to save settings")

def change_output_folder(settings_manager):
    """Change default output folder path"""
    print("\n" + "="*80)
    print("CHANGE DEFAULT OUTPUT FOLDER")
    print("="*80)
    
    current_folder = settings_manager.get_output_folder()
    print(f"\nCurrent default output folder: {current_folder}")
    
    new_folder = input("\nEnter new output folder path (or press Enter to cancel): ").strip().strip('"').strip("'")
    
    if not new_folder:
        print("Cancelled.")
        return
    
    # Create folder if it doesn't exist
    if not Path(new_folder).exists():
        print(f"\nFolder does not exist: {new_folder}")
        confirm = input("Create this folder? (y/n): ").strip().lower()
        if confirm == 'y':
            Path(new_folder).mkdir(parents=True, exist_ok=True)
            print(f"Created folder: {new_folder}")
        else:
            print("Cancelled.")
            return
    
    if settings_manager.set_output_folder(new_folder):
        print(f"\nSUCCESS: Default output folder updated to: {new_folder}")
    else:
        print("\nERROR: Failed to save settings")

def main():
    """Main menu"""
    # Initialize settings manager
    settings_manager = SettingsManager()
    
    while True:
        print("\n" + "="*80)
        print("POST-TRAINING EVALUATION")
        print("="*80)
        print(f"Run folder:    {settings_manager.get_run_folder()}")
        print(f"Output folder: {settings_manager.get_output_folder()}")
        print("\nMenu:")
        print("  1. Evaluate runs (save to CSV)")
        print("  2. Change default run folder path")
        print("  3. Change default output folder path")
        print("  4. Exit")
        
        try:
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                evaluate_runs(settings_manager)
            
            elif choice == '2':
                change_run_folder(settings_manager)
            
            elif choice == '3':
                change_output_folder(settings_manager)
            
            elif choice == '4':
                print("\nExiting...")
                break
            
            else:
                print("ERROR: Invalid choice. Please select 1-4")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nERROR: {e}")

if __name__ == "__main__":
    main()
