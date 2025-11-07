#!/usr/bin/env python3
"""
CSV ↔ Excel Converter
Utility to convert between CSV and Excel formats
"""

import pandas as pd
from pathlib import Path
import sys

def csv_to_excel(csv_file, excel_file=None):
    """Convert CSV file to Excel"""
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_file}")
        return False
    
    # Default Excel filename if not provided
    if excel_file is None:
        excel_file = csv_path.with_suffix('.xlsx')
    else:
        excel_file = Path(excel_file)
        if excel_file.suffix != '.xlsx':
            excel_file = excel_file.with_suffix('.xlsx')
    
    try:
        df = pd.read_csv(csv_path)
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"SUCCESS: Converted to Excel: {excel_file}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to convert CSV to Excel: {e}")
        return False

def excel_to_csv(excel_file, csv_file=None, sheet_name=0):
    """Convert Excel file to CSV"""
    excel_path = Path(excel_file)
    
    if not excel_path.exists():
        print(f"ERROR: Excel file not found: {excel_file}")
        return False
    
    # Default CSV filename if not provided
    if csv_file is None:
        csv_file = excel_path.with_suffix('.csv')
    else:
        csv_file = Path(csv_file)
        if csv_file.suffix != '.csv':
            csv_file = csv_file.with_suffix('.csv')
    
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
        df.to_csv(csv_file, index=False)
        print(f"SUCCESS: Converted to CSV: {csv_file}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to convert Excel to CSV: {e}")
        return False

def interactive_convert():
    """Interactive conversion mode"""
    print("\n" + "=" * 80)
    print("CSV ↔ EXCEL CONVERTER")
    print("=" * 80)
    
    print("\nConversion options:")
    print("  1. CSV to Excel")
    print("  2. Excel to CSV")
    print("  3. Exit")
    
    while True:
        choice = input("\nSelect conversion type (1-3): ").strip()
        
        if choice == '1':
            csv_file = input("Enter CSV file path: ").strip().strip('"').strip("'")
            if not csv_file:
                print("ERROR: No file path provided")
                continue
            
            excel_file = input("Enter Excel file path (press Enter for auto-name): ").strip().strip('"').strip("'")
            # If user provided a directory instead of a file, use auto-naming
            if excel_file and Path(excel_file).is_dir():
                excel_file = None
            elif not excel_file:
                excel_file = None
            
            csv_to_excel(csv_file, excel_file)
            
        elif choice == '2':
            excel_file = input("Enter Excel file path: ").strip().strip('"').strip("'")
            if not excel_file:
                print("ERROR: No file path provided")
                continue
            
            csv_file = input("Enter CSV file path (press Enter for auto-name): ").strip().strip('"').strip("'")
            # If user provided a directory instead of a file, use auto-naming
            if csv_file and Path(csv_file).is_dir():
                csv_file = None
            elif not csv_file:
                csv_file = None
            
            excel_to_csv(excel_file, csv_file)
            
        elif choice == '3':
            print("Exiting converter...")
            break
        else:
            print("ERROR: Invalid choice. Please select 1-3")
        
        # Ask if user wants to do another conversion
        another = input("\nConvert another file? (y/n): ").strip().lower()
        if another != 'y':
            break

def main():
    """Main function for command-line usage"""
    if len(sys.argv) < 2:
        # No arguments, run interactive mode
        interactive_convert()
    else:
        # Command-line mode
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        input_path = Path(input_file)
        
        if input_path.suffix.lower() == '.csv':
            csv_to_excel(input_file, output_file)
        elif input_path.suffix.lower() in ['.xlsx', '.xls']:
            excel_to_csv(input_file, output_file)
        else:
            print(f"ERROR: Unsupported file format: {input_path.suffix}")
            print("Supported formats: .csv, .xlsx, .xls")

if __name__ == "__main__":
    main()

