import os
import glob
import pandas as pd

def aggregate_reports(input_dir, output_file):
    # Find all CSV files in the input directory and its subdirectories
    csv_files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    dfs = []
    for file in csv_files:
        try:
            # Read CSV with utf-8-sig encoding
            df = pd.read_csv(file, encoding='utf-8-sig')
            dfs.append(df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        print("No valid data loaded.")
        return

    # Merge all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Total rows before deduplication: {len(merged_df)}")
    
    # Ensure required columns exist before sorting/deduplication
    if '发布时间' in merged_df.columns:
        # Sort by release time descending
        merged_df = merged_df.sort_values(by='发布时间', ascending=False)
    else:
        print("Warning: '发布时间' column not found for sorting.")
        
    dedup_subset = []
    if '研报名称' in merged_df.columns:
        dedup_subset.append('研报名称')
    else:
        print("Warning: '研报名称' column not found for deduplication.")
        
    if '研报地址' in merged_df.columns:
        dedup_subset.append('研报地址')
    else:
        print("Warning: '研报地址' column not found for deduplication.")
        
    if dedup_subset:
        # Deduplicate
        merged_df = merged_df.drop_duplicates(subset=dedup_subset, keep='first')
        
    print(f"Total rows after deduplication: {len(merged_df)}")

    # Output to the final summary file
    try:
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving to {output_file}: {e}")

if __name__ == "__main__":
    INPUT_DIR = 'eastmoney'
    OUTPUT_FILE = 'All_Reports_Summary.csv'
    
    # We assume this script runs in the project root directory
    aggregate_reports(INPUT_DIR, OUTPUT_FILE)
