import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import time
from collections import defaultdict

from scraper.report_client import EastMoneyReportClient, ReportType

def scrape_recent_reports(days=2):
    client = EastMoneyReportClient()
    
    end_date = datetime.now()
    begin_date = end_date - timedelta(days=days)
    
    end_time = end_date.strftime('%Y-%m-%d')
    begin_time = begin_date.strftime('%Y-%m-%d')
    
    print(f"Scraping global industry reports from {begin_time} to {end_time}...")
    
    page_size = 100
    page_no = 1
    total_pages = 1
    
    industry_groups = defaultdict(list)
    
    while page_no <= total_pages:
        try:
            # We use global wildcard '*' to fetch all market reports across all industries in a single stream
            raw_data = client.fetch_reports(
                report_type=ReportType.INDUSTRY,
                industry_code='*',
                page_no=page_no,
                page_size=page_size,
                begin_time=begin_time,
                end_time=end_time
            )
            
            if not raw_data:
                break
                
            if page_no == 1:
                total_pages = int(raw_data.get('TotalPage', 1))
                if total_pages == 0:
                    total_pages = 1
                print(f"Total global pages for last {days} days: {total_pages}")
                
            reports = client.parse_reports(raw_data, report_type=ReportType.INDUSTRY)
            if not reports:
                break
                
            for r in reports:
                industry_name = r.get('industry_name', '未分类')
                info_code = r.get('info_code', '')
                pdf_link = f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf" if info_code and info_code.startswith("AP") else ""
                
                industry_groups[industry_name].append({
                    '研报名称': r.get('title', ''),
                    '机构名称': r.get('org_name', ''),
                    '发布时间': r.get('publish_date', ''),
                    '行业': industry_name,
                    '页数': r.get('pages', ''),
                    '网页链接': r.get('url', ''),
                    'PDF直链': pdf_link
                })
                
            page_no += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"Error fetching global page {page_no}: {e}")
            break
            
    total_fetched = sum(len(v) for v in industry_groups.values())
    print(f"\nSuccessfully fetched {total_fetched} recent reports across {len(industry_groups)} active industries.")
    
    total_new_reports = 0
    os.makedirs('eastmoney', exist_ok=True)
    
    # Write directly to their respective industry csv files
    for industry_name, new_data in industry_groups.items():
        safe_name = "".join(c for c in industry_name if c not in r'\/:*?"<>|')
        if not safe_name:
            continue
            
        file_path = f"eastmoney/{safe_name}.csv"
        new_df = pd.DataFrame(new_data)
        
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path, encoding='utf-8-sig')
            combined_df = pd.concat([new_df, existing_df], ignore_index=True)
        else:
            combined_df = new_df
            
        initial_len = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['研报名称', '网页链接'], keep='first')
        combined_df = combined_df.sort_values(by='发布时间', ascending=False)
        
        new_added = len(combined_df) - (initial_len - len(new_df))
        if new_added > 0:
            print(f" -> Added {new_added} new reports for {industry_name}")
            total_new_reports += new_added
            
        combined_df.to_csv(file_path, index=False, encoding='utf-8-sig')

    print(f"Total new reports added to local base: {total_new_reports}")

def aggregate_all_reports(input_dir='eastmoney', output_file='All_Reports_Summary.csv'):
    csv_files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)
    dfs = []
    for file in csv_files:
        try:
            dfs.append(pd.read_csv(file, encoding='utf-8-sig'))
        except Exception as e:
            pass

    if dfs:
        merged_df = pd.concat(dfs, ignore_index=True)
        if '发布时间' in merged_df.columns:
            merged_df = merged_df.sort_values(by='发布时间', ascending=False)
            
        dedup_subset = []
        if '研报名称' in merged_df.columns: dedup_subset.append('研报名称')
        if '网页链接' in merged_df.columns: dedup_subset.append('网页链接')
        if dedup_subset:
            merged_df = merged_df.drop_duplicates(subset=dedup_subset, keep='first')
            
        print(f"Aggregated total rows: {len(merged_df)}")
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    scrape_recent_reports(days=2)
    print("Aggregating all reports...")
    aggregate_all_reports()
    print("Done!")
