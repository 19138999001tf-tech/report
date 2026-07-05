import os
import sys
import json
import time
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

def load_data(limit=30000):
    csv_path = "All_Reports_Summary.csv"
    if not os.path.exists(csv_path):
        print(f"Error: 找不到数据文件 {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df['研报名称'] = df['研报名称'].fillna('')
    df['行业'] = df['行业'].fillna('')
    
    # 截取前 N 条
    df = df.head(limit)
    return df

def test_long_context(query, limit):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.com/v1")
    model_name = os.getenv("MODEL_NAME") or os.getenv("SILICONFLOW_MODEL", "deepseek-ai/DeepSeek-V4-Flash")
    
    if not api_key:
        print("错误: 找不到 API KEY。")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    print(f"\n🚀 开始构建长上下文测试...")
    print(f"📦 目标数据量: {limit} 篇研报")
    
    df = load_data(limit)
    print(f"✅ 成功加载 {len(df)} 篇研报，正在序列化...")
    
    # 极简压缩序列化以节省 Token：
    # 格式：ID|研报标题|行业
    text_blocks = []
    for idx, row in df.iterrows():
        # 用 \t 替代字典可以极大节省 token
        text_blocks.append(f"{idx}|{row['研报名称']}|{row['行业']}")
        
    all_text = "\n".join(text_blocks)
    chars = len(all_text)
    print(f"📊 序列化完成！总字符数: {chars:,} (预估 Token 数约为 {chars/2.5:,.0f} ~ {chars/1.5:,.0f})")
    
    prompt = f"""你是一个高级金融投研助手，具备超长上下文处理能力。
用户正在寻找关于“{query}”的高质量深度研报。

以下是 {limit} 篇研报的列表，每行的格式为：[ID]|[研报标题]|[行业]。
请你从这 {limit} 篇研报中，犹如大海捞针般，找出**最相关、最具深度**的 Top 10 篇研报。

请直接返回纯 JSON 格式，不要包含 ```json 等任何额外内容。格式如下：
{{
  "top_results": [
    {{"id": 123, "reason": "标题高度匹配固态电池，且为行业深度报告"}}
  ]
}}

========== 研报数据开始 ==========
{all_text}
========== 研报数据结束 ==========
"""
    
    print(f"🤖 正在发送至大模型 {model_name}...")
    print(f"⏳ 这可能会花费较长时间，请耐心等待...")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        end_time = time.time()
        
        result_text = response.choices[0].message.content
        usage = response.usage
        
        print(f"\n🎉 测试成功完成！耗时: {end_time - start_time:.2f} 秒")
        print(f"📈 Token 消耗: Prompt {usage.prompt_tokens:,} | Completion {usage.completion_tokens:,} | Total {usage.total_tokens:,}")
        
        print("\n🏆 大模型返回的结果：")
        try:
            res_json = json.loads(result_text)
            for item in res_json.get('top_results', []):
                idx = int(item['id'])
                # 容错：可能大模型瞎编了超出的 ID
                if 0 <= idx < len(df):
                    row = df.iloc[idx]
                    print(f"- ID: {idx} | 标题: {row['研报名称']}")
                    print(f"  理由: {item.get('reason', '')}")
                else:
                    print(f"- 大模型幻觉生成了不存在的 ID: {idx}")
        except json.JSONDecodeError:
            print("返回的 JSON 解析失败，原始文本：")
            print(result_text)
            
    except Exception as e:
        end_time = time.time()
        print(f"\n❌ 测试失败！耗时: {end_time - start_time:.2f} 秒")
        print(f"错误信息: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="超长上下文模型测试")
    parser.add_argument("-q", "--query", type=str, required=True, help="搜索关键词")
    parser.add_argument("-l", "--limit", type=int, default=30000, help="送入模型的研报数量")
    args = parser.parse_args()
    
    test_long_context(args.query, args.limit)
