#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并正向和反向处理的结果

用法：
python merge_results.py --model deepseek-v3.2
"""

import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def merge_model_results(model_key: str):
    """合并单个模型的正向和反向结果"""
    
    forward_file = PROJECT_ROOT / "api_generation" / f"api_kc_{model_key}.json"
    reverse_file = PROJECT_ROOT / "api_generation" / f"api_kc_{model_key}_reverse.json"
    merged_file = PROJECT_ROOT / "api_generation" / f"api_kc_{model_key}_merged.json"
    
    print(f"\n{'='*60}")
    print(f"  合并 {model_key} 结果")
    print(f"{'='*60}")
    
    # 读取正向结果
    forward_samples = {}
    if forward_file.exists():
        with open(forward_file, 'r') as f:
            forward_data = json.load(f)
        for s in forward_data.get("samples", []):
            forward_samples[s.get("sample_id")] = s
        print(f"正向文件: {forward_file}")
        print(f"  - 样本数: {len(forward_samples)}")
        print(f"  - 成功数: {forward_data.get('success_count', 0)}")
    
    # 读取反向结果
    reverse_samples = {}
    if reverse_file.exists():
        with open(reverse_file, 'r') as f:
            reverse_data = json.load(f)
        for s in reverse_data.get("samples", []):
            reverse_samples[s.get("sample_id")] = s
        print(f"反向文件: {reverse_file}")
        print(f"  - 样本数: {len(reverse_samples)}")
        print(f"  - 成功数: {reverse_data.get('success_count', 0)}")
    
    # 合并（去重，优先使用成功的）
    merged = {}
    
    # 先添加正向的
    for sid, sample in forward_samples.items():
        merged[sid] = sample
    
    # 再添加反向的（如果正向没有，或者反向成功但正向失败）
    for sid, sample in reverse_samples.items():
        if sid not in merged:
            merged[sid] = sample
        elif not merged[sid].get("success") and sample.get("success"):
            # 用反向成功的替换正向失败的
            merged[sid] = sample
    
    # 计算成功数
    success_count = sum(1 for s in merged.values() if s.get("success"))
    
    print(f"\n合并结果:")
    print(f"  - 总样本数: {len(merged)}")
    print(f"  - 总成功数: {success_count}")
    
    # 写入合并文件
    merged_data = {
        "model": model_key,
        "merged_from": [str(forward_file), str(reverse_file)],
        "total_samples": len(merged),
        "success_count": success_count,
        "success_rate": success_count / len(merged) if merged else 0,
        "timestamp": "merged",
        "samples": list(merged.values())
    }
    
    with open(merged_file, 'w') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n输出: {merged_file}")
    
    return merged_data


def main():
    parser = argparse.ArgumentParser(description="合并正向反向结果")
    parser.add_argument("--model", type=str, required=True,
                       choices=["deepseek-v3.2", "gemini-3-flash-preview"],
                       help="模型")
    args = parser.parse_args()
    
    merge_model_results(args.model)


if __name__ == "__main__":
    main()
