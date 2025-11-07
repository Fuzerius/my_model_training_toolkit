#!/usr/bin/env python3
"""
Quick Results Summary - Compare selected dataset performances
"""

import pandas as pd
from pathlib import Path
import json
import os

def load_settings():
    """Load settings from settings/quick_results_summary.json"""
    settings_file = Path("settings/quick_results_summary.json")
    default_settings = {
        "default_results_folder": "runs/detect"
    }
    
    if settings_file.exists():
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            return settings.get("default_results_folder", default_settings["default_results_folder"])
        except (json.JSONDecodeError, IOError):
            return default_settings["default_results_folder"]
    else:
        return default_settings["default_results_folder"]

def get_available_result_folders(base_path="runs/detect"):
    """Get all available result folders with results.csv"""
    base_dir = Path(base_path)
    if not base_dir.exists():
        return []
    
    result_folders = []
    for folder in base_dir.iterdir():
        if folder.is_dir() and (folder / "results.csv").exists():
            result_folders.append(folder.name)
    
    return sorted(result_folders)

def get_final_metrics_from_folder(folder_name, base_path="runs/detect"):
    """Get final metrics from a specific result folder"""
    results_file = Path(base_path) / folder_name / "results.csv"
    
    if not results_file.exists():
        return None
    
    try:
        df = pd.read_csv(results_file)
        final_row = df.iloc[-1]
        
        return {
            'folder': folder_name,
            'final_map50': final_row.get('metrics/mAP50(B)', 0),
            'final_map50_95': final_row.get('metrics/mAP50-95(B)', 0),
            'best_map50': df['metrics/mAP50(B)'].max(),
            'best_epoch': df['metrics/mAP50(B)'].idxmax(),
            'total_epochs': len(df),
            'final_precision': final_row.get('metrics/precision(B)', 0),
            'final_recall': final_row.get('metrics/recall(B)', 0)
        }
    except Exception as e:
        print(f"Error reading folder {folder_name}: {e}")
        return None

def get_final_metrics(dataset_num, base_path="runs/detect"):
    """Get final metrics from a dataset's results (backward compatibility)"""
    return get_final_metrics_from_folder(f"test_dataset_{dataset_num}", base_path)

def classify_difficulty(metrics):
    """Classify dataset difficulty based on performance"""
    map50 = metrics['best_map50']
    
    if map50 >= 0.70:
        return "EASY Easy", 3
    elif map50 >= 0.50:
        return "MEDIUM Medium", 2
    else:
        return "HARD Hard", 1

def get_folder_selection(available_folders):
    """Let user select which folders to compare"""
    print("FOLDER Available result folders:")
    for i, folder in enumerate(available_folders, 1):
        print(f"  {i}. {folder}")
    
    print("\nSelection options:")
    print("  • Enter numbers separated by commas (e.g., 1,3,5)")
    print("  • Enter 'all' to select all folders")
    print("  • Press Enter to select all folders")
    
    while True:
        try:
            user_input = input("\nSelect folders to compare: ").strip()
            
            if not user_input or user_input.lower() == 'all':
                return available_folders
            
            # Parse comma-separated numbers
            selected_indices = []
            valid_input = True
            for num_str in user_input.split(','):
                num_str = num_str.strip()
                if num_str:
                    idx = int(num_str) - 1
                    if 0 <= idx < len(available_folders):
                        selected_indices.append(idx)
                    else:
                        print(f"ERROR Invalid number: {num_str}. Please use numbers 1-{len(available_folders)}")
                        valid_input = False
                        break
            
            if valid_input and selected_indices:
                selected_folders = [available_folders[i] for i in selected_indices]
                return selected_folders
            elif valid_input:
                print("ERROR Please enter valid folder numbers.")
            
        except ValueError:
            print("ERROR Please enter valid numbers separated by commas.")
        except KeyboardInterrupt:
            print("\nERROR Operation cancelled.")
            return None

def main():
    print("CHART QUICK RESULTS SUMMARY")
    print("=" * 80)
    
    # Load settings to get default results folder
    default_results_folder = load_settings()
    
    # Check if default folder exists, if not ask user
    if not Path(default_results_folder).exists():
        print(f"ERROR Default results folder '{default_results_folder}' not found!")
        user_folder = input("Enter path to YOLO results folder (e.g., runs/detect): ").strip()
        if not user_folder:
            print("ERROR No folder provided. Exiting.")
            return
        results_folder = user_folder
    else:
        results_folder = default_results_folder
    
    print(f"FOLDER Using results folder: {results_folder}")
    
    # Get available result folders
    available_folders = get_available_result_folders(results_folder)
    
    if not available_folders:
        print("ERROR No training results found!")
        print(f"Make sure you have trained models with results.csv in: {results_folder}")
        return
    
    # Let user select folders to compare
    selected_folders = get_folder_selection(available_folders)
    if not selected_folders:
        return
    
    print(f"\nSUCCESS Selected {len(selected_folders)} folder(s) for comparison")
    print(f"LIST Comparing: {', '.join(selected_folders)}")
    print()
    
    # Get results for selected folders
    all_results = []
    for folder_name in selected_folders:
        metrics = get_final_metrics_from_folder(folder_name, results_folder)
        if metrics:
            all_results.append(metrics)
    
    if not all_results:
        print("ERROR No valid results found in selected folders!")
        return
    
    # Sort by performance (best mAP50 first)
    all_results.sort(key=lambda x: x['best_map50'], reverse=True)
    
    # Print detailed results
    print(f"{'Rank':<6} {'Folder':<25} {'Difficulty':<12} {'Best mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12} {'Best Epoch':<12}")
    print("-" * 120)
    
    easy_folders = []
    medium_folders = []
    hard_folders = []
    
    for rank, result in enumerate(all_results, 1):
        difficulty, score = classify_difficulty(result)
        
        # Categorize folders
        if score == 3:
            easy_folders.append(result['folder'])
        elif score == 2:
            medium_folders.append(result['folder'])
        else:
            hard_folders.append(result['folder'])
        
        print(f"{rank:<6} {result['folder']:<25} {difficulty:<12} {result['best_map50']:<12.3f} "
              f"{result['final_map50_95']:<12.3f} {result['final_precision']:<12.3f} "
              f"{result['final_recall']:<12.3f} {result['best_epoch']:<12}")
    
    # Print classification summary
    print("\n" + "=" * 80)
    print("TARGET MODEL DIFFICULTY CLASSIFICATION")
    print("=" * 80)
    
    print(f"EASY EASY MODELS (mAP50 ≥ 0.70): {len(easy_folders)} models")
    if easy_folders:
        print(f"   → Use for base training")
        print(f"   → Best starter: {easy_folders[0]}")
        for folder in easy_folders[:3]:  # Show top 3
            print(f"     • {folder}")
        if len(easy_folders) > 3:
            print(f"     ... and {len(easy_folders) - 3} more")
    
    print(f"\nMEDIUM MEDIUM MODELS (mAP50 = 0.50-0.69): {len(medium_folders)} models")
    if medium_folders:
        print(f"   → Use for progressive fine-tuning")
        for folder in medium_folders[:3]:  # Show top 3
            print(f"     • {folder}")
        if len(medium_folders) > 3:
            print(f"     ... and {len(medium_folders) - 3} more")
    
    print(f"\nHARD HARD MODELS (mAP50 < 0.50): {len(hard_folders)} models")
    if hard_folders:
        print(f"   → Use for final robustness training")
        print(f"   → Train with conservative settings")
        for folder in hard_folders[:3]:  # Show top 3
            print(f"     • {folder}")
        if len(hard_folders) > 3:
            print(f"     ... and {len(hard_folders) - 3} more")
    
    # Training recommendations
    print("\n" + "=" * 80)
    print("LAUNCH RECOMMENDED PROGRESSIVE TRAINING ORDER")
    print("=" * 80)
    
    if easy_folders:
        print(f"1.  Base training: {easy_folders[0]} (best easy model)")
        
    if medium_folders:
        print(f"2.  Progressive fine-tuning:")
        for i, folder in enumerate(medium_folders[:3], 1):  # Show top 3
            print(f"    {i}. {folder}")
        if len(medium_folders) > 3:
            print(f"    ... and {len(medium_folders) - 3} more medium models")
    
    if hard_folders:
        print(f"3.  Final robustness training:")
        for i, folder in enumerate(hard_folders[:3], 1):  # Show top 3
            print(f"    {i}. {folder}")
        if len(hard_folders) > 3:
            print(f"    ... and {len(hard_folders) - 3} more hard models")
    
    print(f"\nTIP Start with: {all_results[0]['folder']} (highest mAP50: {all_results[0]['best_map50']:.3f})")

if __name__ == "__main__":
    main()
