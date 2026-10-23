#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补全基座模型生成失败的样本 (增强版)
问题：Qwen3-8B 的思考模式导致输出包含 <thinking>...</thinking> 标签
解决：多策略去除标签后重新解析 JSON

增强功能：
1. 处理嵌套 thinking 标签
2. 处理 thinking 标签被截断的情况
3. 从长文本中提取 JSON 片段

Usage:
python step6/complete_base_failures.py \
  --results_file step6/test_results/results_base.json \
  --output_file step6/test_results/results_base_completed.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, List


def clean_thinking_tags(text: str) -> str:
    """去除 thinking 标签及其内容"""
    # 策略1: 去除完整的 <thinking>...</thinking>
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 策略2: 去除可能被截断的 <thinking> 标签
    text = re.sub(r'<thinking>.*', '', text, flags=re.DOTALL)
    
    # 策略3: 去除结尾的 </thinking>
    text = re.sub(r'</thinking>', '', text)
    
    return text.strip()


def extract_json_robust(text: str) -> Optional[Dict]:
    """
    增强版 JSON 提取函数，多策略处理
    """
    if not text:
        return None
    
    text = text.strip()
    original_text = text
    
    # 1. 预处理：去除 thinking 标签
    text = clean_thinking_tags(text)
    
    # 2. 尝试直接解析
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    
    # 3. 查找 JSON 块（第一个 { 到最后一个 }）
    start = text.find("{")
    end = text.rfind("}")
    
    if start >= 0 and end > start:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 4. 尝试查找 knowledge_mining 关键词附近的 JSON
    km_pattern = r'"knowledge_mining"\s*:\s*\['
    km_match = re.search(km_pattern, text)
    if km_match:
        km_start = km_match.start()
        brace_start = text.rfind("{", 0, km_start)
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i+1])
                        except:
                            pass
                        break
    
    # 5. 尝试查找任何看起来像知识链的结构
    chain_pattern = r'"knowledge_chain"\s*:\s*\{'
    chain_matches = list(re.finditer(chain_pattern, text))
    
    if chain_matches:
        last_match = chain_matches[-1]
        brace_start = text.rfind("{", 0, last_match.start())
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i+1])
                        except:
                            pass
                        break
    
    return None


def analyze_and_fix_results(results_file: Path, output_file: Path):
    """分析并修复生成失败的样本"""
    
    print(f"\n{'='*70}")
    print("📊 补全基座模型生成失败样本 (增强版)")
    print(f"{'='*70}")
    
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    total = len(results)
    print(f"\n总样本数: {total}")
    
    success = []
    failed = []
    no_text = []
    
    for i, item in enumerate(results):
        if item.get('generated_json') is not None:
            success.append(i)
        elif item.get('generated_text'):
            failed.append(i)
        else:
            no_text.append(i)
    
    print(f"\n当前状态:")
    print(f"  ✅ 成功解析: {len(success)} ({len(success)/total*100:.1f}%)")
    print(f"  ❌ 解析失败但有文本: {len(failed)} ({len(failed)/total*100:.1f}%)")
    print(f"  ⚠️  无生成文本: {len(no_text)} ({len(no_text)/total*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("🔧 开始补全...")
    print(f"{'='*70}")
    
    fixed_count = 0
    still_failed = []
    
    for idx in failed:
        item = results[idx]
        file_id = item.get('file_id', 'unknown')
        gen_text = item.get('generated_text', '')
        
        gen_json = extract_json_robust(gen_text)
        
        if gen_json is not None:
            results[idx]['generated_json'] = gen_json
            results[idx]['_fixed'] = True
            fixed_count += 1
            print(f"  ✅ file_id={file_id}: 修复成功")
        else:
            still_failed.append(idx)
            print(f"  ❌ file_id={file_id}: 仍然失败")
    
    print(f"\n{'='*70}")
    print("📋 补全结果")
    print(f"{'='*70}")
    print(f"  修复成功: {fixed_count}")
    print(f"  仍然失败: {len(still_failed)}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")
    
    new_success = sum(1 for item in results if item.get('generated_json') is not None)
    print(f"\n📊 更新后状态:")
    print(f"  成功解析: {new_success}/{total} ({new_success/total*100:.1f}%)")
    print(f"  失败: {total - new_success}/{total} ({(total-new_success)/total*100:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, 
                        default="step6/test_results/results_base.json",
                        help="原始结果文件")
    parser.add_argument("--output_file", type=str, 
                        default="step6/test_results/results_base_completed.json",
                        help="输出文件路径")
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    output_file = Path(args.output_file)
    
    if not results_file.exists():
        print(f"ERROR: 文件不存在: {results_file}")
        return
    
    analyze_and_fix_results(results_file, output_file)


if __name__ == "__main__":
    main()

