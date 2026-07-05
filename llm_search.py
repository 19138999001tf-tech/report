import os
import sys
import json
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

def load_data(csv_path="All_Reports_Summary.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: 找不到数据文件 {csv_path}。请先运行爬虫抓取数据。")
        sys.exit(1)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    # 处理可能的 NaN
    df['研报名称'] = df['研报名称'].fillna('')
    df['行业'] = df['行业'].fillna('')
    # 将页数转换为数字，非法值设为0
    df['页数'] = pd.to_numeric(df['页数'], errors='coerce').fillna(0)
    return df

def rough_recall(df, queries, top_k=100):
    mask = pd.Series(False, index=df.index)
    
    for query in queries:
        keywords = [k.strip() for k in query.split() if k.strip()]
        if not keywords:
            continue
            
        regex_pattern = "".join([f"(?=.*{k})" for k in keywords])
        
        current_mask = df['研报名称'].str.contains(regex_pattern, case=False, regex=True) | \
                       df['行业'].str.contains(regex_pattern, case=False, regex=True)
        mask = mask | current_mask
           
    candidates = df[mask].copy()
    
    if len(candidates) == 0:
        return candidates
        
    # 如果候选太多，优先取页数多、时间新的
    if len(candidates) > top_k:
        candidates = candidates.sort_values(by=['页数', '发布时间'], ascending=[False, False])
        candidates = candidates.head(top_k)
        
    return candidates

def llm_query_expansion(query):
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        return [query]
        
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")
    model_name = os.getenv("MODEL_NAME") or os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    prompt = f"""用户正在金融研报库中搜索关键词：“{query}”。
为了防止字面匹配漏掉相关研报，请你发散并联想出 3 到 5 个强相关的同义词、上下游核心技术、或者是该领域的俗称。
请必须严格返回纯 JSON 格式（不要包含 markdown 代码块如 ```json ），格式如下：
{{"expanded": ["词1", "词2", "词3"]}}
"""
    print(f"正在请求大模型({model_name})进行搜索意图泛化 (Query Expansion)...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        expanded = result_json.get('expanded', [])
        
        # 将原词放在最前面
        final_queries = [query] + [q for q in expanded if q != query]
        print(f"💡 意图泛化完成！扩展后的检索矩阵为: {final_queries}")
        return final_queries
    except Exception as e:
        print(f"意图泛化失败: {e}")
        return [query]

def llm_rerank(candidates_df, query):
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("\n错误: 找不到 OPENAI_API_KEY 或 SILICONFLOW_API_KEY。")
        sys.exit(1)
        
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")
    model_name = os.getenv("MODEL_NAME") or os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    # 构造给 LLM 的轻量级候选列表
    llm_input = []
    for idx, row in candidates_df.iterrows():
        llm_input.append({
            "id": idx,
            "title": row['研报名称'],
            "industry": row['行业']
        })
        
    prompt = f"""你是一个高级金融投研助手。用户正在寻找关于“{query}”的高质量深度研报。
以下是系统初步找出的相关研报候选列表。请你根据每篇研报的“标题”和“行业”，判断它是否是一篇深入、高质量且紧密切中主题的报告。

打分维度参考：
1. 语义相关性：标题是否直接且强相关用户的关键词“{query}”？
2. 深度感知：标题是否暗示这是一篇深度研报？（如包含“深度研究”、“首次覆盖”、“专题”的高分；包含“短评”、“快评”、“周报”、“早知道”的低分）。

请为每一篇研报打分（0-100分）。
必须严格返回纯 JSON 格式（不要包含 markdown 代码块如 ```json ），格式如下：
{{"scores": [{{"id": 0, "score": 85}}, {{"id": 1, "score": 20}}]}}

研报列表：
{json.dumps(llm_input, ensure_ascii=False, indent=2)}
"""

    print(f"正在请求大模型({model_name})进行语义深度精排打分...")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        # 建立映射
        score_map = {item['id']: item['score'] for item in result_json.get('scores', [])}
        return score_map
    except Exception as e:
        print(f"大模型调用失败: {e}")
        return {}

import datetime
from dateutil.relativedelta import relativedelta

def search(query, top_n=5, min_pages=10, time_limit_months=18):
    load_dotenv()
    df = load_data()
    
    initial_len = len(df)
    
    # 1. 页数过滤
    df = df[df['页数'] >= min_pages]
    
    # 2. 时间过滤 (特权文件豁免)
    if time_limit_months > 0:
        cutoff_date = (datetime.datetime.now() - relativedelta(months=time_limit_months)).strftime('%Y-%m-%d')
        # 政策、规划、白皮书等权威文件不受时间限制
        privilege_keywords = '政策|规划|白皮书|蓝皮书|指南|年鉴|意见|办法|方案|纲要|权威|条例'
        mask_time = df['发布时间'] >= cutoff_date
        mask_privilege = df['研报名称'].str.contains(privilege_keywords, case=False, regex=True, na=False)
        df = df[mask_time | mask_privilege]
        
    df = df.copy()
    
    print(f"\n🔍 正在底座库({len(df)}/{initial_len}篇研报)中搜索: '{query}'")
    print(f"   ⚙️ 过滤条件: 页数>={min_pages}页 | 近{time_limit_months}个月 (政策/规划/白皮书等永久保留)")
    
    # 0. 意图泛化
    expanded_queries = llm_query_expansion(query)
    
    # 1. 宽口径粗排
    candidates = rough_recall(df, expanded_queries, top_k=100)
    print(f"✅ 宽口径粗排命中 {len(candidates)} 篇候选研报。")
    if len(candidates) == 0:
        print("没有找到相关研报，请尝试更换关键词。")
        return
        
    # 2. 精排
    llm_scores = llm_rerank(candidates, query)
    
    # 3. 综合排序
    # 最终得分 = LLM语义得分 * 0.7 + 页数标准化得分 * 0.3
    # 假设一篇优质深度研报正常是 50 页，超过 50 页的部分边际效用递减
    final_results = []
    for idx, row in candidates.iterrows():
        semantic_score = llm_scores.get(idx, 50) # 如果LLM漏了，默认50分
        pages = row['页数']
        
        # 页数得分：超过50页的算100分满分，少于的按比例
        page_score = min(pages / 50.0 * 100, 100)
        
        # 综合得分
        final_score = (semantic_score * 0.7) + (page_score * 0.3)
        
        final_results.append({
            '标题': row['研报名称'],
            '机构': row['机构名称'],
            '行业': row['行业'],
            '时间': row['发布时间'],
            '页数': pages,
            'LLM得分': semantic_score,
            '最终得分': round(final_score, 1),
            'PDF直链': row.get('PDF直链', row.get('网页链接', ''))
        })
        
    # 按最终得分降序
    final_results.sort(key=lambda x: x['最终得分'], reverse=True)
    
    # 输出 Top N
    print(f"\n🏆 综合语义深度与页数排序，为您精选 Top {top_n}：\n" + "="*80)
    for i, res in enumerate(final_results[:top_n]):
        print(f"[{i+1}] {res['标题']}")
        print(f"    🌟 综合评分: {res['最终得分']} (语义分:{res['LLM得分']}, 页数:{int(res['页数'])}页)")
        print(f"    🏢 机构: {res['机构']} | 行业: {res['行业']} | 时间: {res['时间']}")
        print(f"    🔗 下载直链: {res['PDF直链']}")
        print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 LLM 和页数深度的投研检索引擎")
    parser.add_argument("-q", "--query", type=str, required=True, help="搜索关键词")
    parser.add_argument("-n", "--top", type=int, default=5, help="返回的 Top N 数量")
    parser.add_argument("-m", "--min-pages", type=int, default=10, help="最低页数过滤（默认 10 页）")
    parser.add_argument("-t", "--time-limit", type=int, default=18, help="时间过滤(月)：普通研报只看近N个月，政策文件不限时（默认18）")
    args = parser.parse_args()
    
    search(args.query, args.top, args.min_pages, args.time_limit)
