#!/usr/bin/env python3
"""
Debug Results - Show raw data from each dataset
"""

import pandas as pd
from pathlib import Path

def debug_single_dataset(dataset_num):
    """Debug a single dataset's results"""
    results_file = Path(f"runs/detect/test_dataset_{dataset_num}/results.csv")
    
    print(f"\n📊 Dataset {dataset_num}:")
    print("-" * 40)
    
    if not results_file.exists():
        print("❌ No results.csv found")
        return None
    
    try:
        df = pd.read_csv(results_file)
        print(f"✅ Results file found: {len(df)} epochs")
        
        # Show column names
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show last few rows
        print(f"\n📈 Last 3 rows:")
        print(df[['epoch', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']].tail(3))
        
        # Key metrics
        final_row = df.iloc[-1]
        best_map50 = df['metrics/mAP50(B)'].max()
        best_epoch = df['metrics/mAP50(B)'].idxmax()
        
        print(f"\n🎯 Key Metrics:")
        print(f"   Final mAP50: {final_row.get('metrics/mAP50(B)', 'N/A'):.4f}")
        print(f"   Best mAP50: {best_map50:.4f} (epoch {best_epoch})")
        print(f"   Final Precision: {final_row.get('metrics/precision(B)', 'N/A'):.4f}")
        print(f"   Final Recall: {final_row.get('metrics/recall(B)', 'N/A'):.4f}")
        
        return {
            'dataset': dataset_num,
            'final_map50': final_row.get('metrics/mAP50(B)', 0),
            'best_map50': best_map50,
            'best_epoch': best_epoch,
            'total_epochs': len(df)
        }
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def main():
    print("🔍 DEBUG: Raw Results Analysis")
    print("=" * 60)
    
    datasets_to_check = [1, 3, 4, 5, 6, 7, 8, 9]
    all_results = []
    
    for dataset_num in datasets_to_check:
        result = debug_single_dataset(dataset_num)
        if result:
            all_results.append(result)
    
    # Summary table
    print(f"\n📊 SUMMARY TABLE:")
    print(f"{'Dataset':<8} {'Final mAP50':<12} {'Best mAP50':<12} {'Best Epoch':<12} {'Total Epochs':<12}")
    print("-" * 60)
    
    for result in sorted(all_results, key=lambda x: x['best_map50'], reverse=True):
        print(f"{result['dataset']:<8} {result['final_map50']:<12.4f} {result['best_map50']:<12.4f} "
              f"{result['best_epoch']:<12} {result['total_epochs']:<12}")
    
    # Classification based on best mAP50
    print(f"\n🎯 CLASSIFICATION (based on best mAP50):")
    for result in sorted(all_results, key=lambda x: x['best_map50'], reverse=True):
        map50 = result['best_map50']
        if map50 >= 0.70:
            difficulty = "🟢 Easy"
        elif map50 >= 0.50:
            difficulty = "🟡 Medium"
        else:
            difficulty = "🔴 Hard"
        
        print(f"Dataset {result['dataset']}: {difficulty} (mAP50: {map50:.4f})")

if __name__ == "__main__":
    main()
