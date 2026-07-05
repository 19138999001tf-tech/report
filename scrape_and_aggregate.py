import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import time
import csv

from scraper.report_client import EastMoneyReportClient, ReportType

def update_industry_reports(csv_path="scraper/industry.csv"):
    client = EastMoneyReportClient()
    begin_time = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
    end_time = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Scraping reports from {begin_time} to {end_time}...")
    
    # Read industry list
    industries = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['行业代码'] != '*':
                industries.append((row['行业名称'], row['行业代码']))
                
    total_new_reports = 0
    
    for name, code in industries:
        print(f"Fetching {name} ({code})...")
        
        # We only fetch page 1 with page_size=100. Usually that's enough for 2 days of a single industry.
        # If an industry has more than 100 reports in 2 days, we might need pagination, but 100 is highly safe for industry scope.
        raw_data = client.fetch_reports(
            report_type=ReportType.INDUSTRY,
            industry_code=code,
            page_no=1,
            page_size=100,
            begin_time=begin_time,
            end_time=end_time
        )
        
        if not raw_data:
            continue
            
        reports = client.parse_reports(raw_data, report_type=ReportType.INDUSTRY)
        if not reports:
            continue
            
        # Convert to our format
        new_data = []
        for r in reports:
            new_data.append({
                '研报名称': r.get('title', ''),
                '机构名称': r.get('org_name', ''),
                '发布时间': r.get('publish_date', ''),
                '行业': r.get('industry_name', name),
                '研报地址': r.get('url', '')
            })
            
        new_df = pd.DataFrame(new_data)
        
        file_path = f"eastmoney/{name}.csv"
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, encoding='utf-8-sig')
            combined_df = pd.concat([new_df, existing_df], ignore_index=True)
        else:
            combined_df = new_df
            os.makedirs('eastmoney', exist_ok=True)
            
        # Deduplicate
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['研报名称', '研报地址'], keep='first')
        combined_df = combined_df.sort_values(by='发布时间', ascending=False)
        
        new_added = len(combined_df) - (initial_len - len(new_df))
        if new_added > 0:
            print(f" -> Added {new_added} new reports for {name}")
            total_new_reports += new_added
            
        combined_df.to_csv(file_path, index=False, encoding='utf-8-sig')
        time.sleep(1) # Be nice to the API
        
    print(f"Total new reports added: {total_new_reports}")


def aggregate_all_reports(input_dir='eastmoney', output_file='All_Reports_Summary.csv'):
    csv_files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    
    if '发布时间' in merged_df.columns:
        merged_df = merged_df.sort_values(by='发布时间', ascending=False)
        
    dedup_subset = []
    if '研报名称' in merged_df.columns: dedup_subset.append('研报名称')
    if '研报地址' in merged_df.columns: dedup_subset.append('研报地址')
        
    if dedup_subset:
        merged_df = merged_df.drop_duplicates(subset=dedup_subset, keep='first')
        
    print(f"Aggregated total rows: {len(merged_df)}")
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    print("Starting daily EastMoney scrape...")
    update_industry_reports()
    print("Aggregating all reports...")
    aggregate_all_reports()
    print("Done!")
