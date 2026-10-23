#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API 生成结果评测脚本

使用 step7 的 r_rule_evaluator_final.py 计算四个模型的 R-rule 分数

用法：
python evaluate_api_results.py
python evaluate_api_results.py --results_dir api_generation
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "step7"))

# 导入评测器
from step7.r_rule_evaluator_final import compute_r_rule


def load_api_results(results_dir: Path) -> Dict[str, List[Dict]]:
    """加载四个模型的生成结果"""
    model_results = {}
    
    # 查找所有 api_kc_*.json 文件
    result_files = sorted(results_dir.glob("api_kc_*.json"))
    
    if not result_files:
        print(f"⚠️ 在 {results_dir} 中未找到 api_kc_*.json 文件")
        print("   请先运行: python api_generation/api_kc_generation.py")
        return {}
    
    for f in result_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            
            model_name = data.get("model", f.stem.replace("api_kc_", ""))
            samples = data.get("samples", [])
            
            model_results[model_name] = {
                "model_full_name": data.get("model_full_name", ""),
                "total_samples": data.get("total_samples", len(samples)),
                "success_count": data.get("success_count", 0),
                "error_count": data.get("error_count", 0),
                "samples": samples
            }
            
            print(f"✅ 加载 {f.name}: {len(samples)} 个样本")
            
        except Exception as e:
            print(f"⚠️ 加载 {f} 失败: {e}")
    
    return model_results


def evaluate_sample(sample: Dict) -> Dict[str, Any]:
    """评估单个样本"""
    # 准备评测数据
    # ground_truth 是列表，需要包装成 {"knowledge_mining": [...]} 格式
    ground_truth = sample.get("ground_truth", [])
    if isinstance(ground_truth, list):
        ground_truth = {"knowledge_mining": ground_truth}
    
    # generated_kc 已经是 {"knowledge_mining": [...]} 格式
    generated_kc = sample.get("generated_kc", {"knowledge_mining": []})
    
    eval_sample = {
        "sample_id": sample.get("sample_id", ""),
        "ground_truth": ground_truth,
        "generated_json": generated_kc
    }
    
    # 使用 step7 的评测器
    result = compute_r_rule(eval_sample)
    
    return result


def evaluate_model(model_name: str, model_data: Dict) -> Dict[str, Any]:
    """评测单个模型的所有样本"""
    samples = model_data["samples"]
    total = len(samples)
    
    results = {
        "model": model_name,
        "total_samples": total,
        "r_rule_scores": [],
        "condition_scores": [],
        "result_scores": [],
        "entity_scores": [],
    }
    
    # 用于统计
    parse_errors = 0
    empty_outputs = 0
    
    for i, sample in enumerate(samples):
        # 跳过失败的样本
        if not sample.get("success", False):
            parse_errors += 1
            results["r_rule_scores"].append(0.0)
            results["condition_scores"].append(0.0)
            results["result_scores"].append(0.0)
            results["entity_scores"].append(0.0)
            continue
        
        # 检查生成是否为空
        generated_kc = sample.get("generated_kc", {})
        if not generated_kc or not generated_kc.get("knowledge_mining"):
            empty_outputs += 1
            results["r_rule_scores"].append(0.0)
            results["condition_scores"].append(0.0)
            results["result_scores"].append(0.0)
            results["entity_scores"].append(0.0)
            continue
        
        # 评测
        eval_result = evaluate_sample(sample)
        
        results["r_rule_scores"].append(eval_result.get("R_rule", 0.0))
        results["condition_scores"].append(eval_result.get("S_content", 0.0))
        results["result_scores"].append(eval_result.get("S_structure", 0.0))
        results["entity_scores"].append(eval_result.get("entity_score", 0.0))
        
        # 进度
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  [{model_name}] {i+1}/{total}")
    
    # 计算平均分
    scores = results["r_rule_scores"]
    valid_scores = [s for s in scores if s > 0]
    
    results["avg_r_rule"] = sum(scores) / len(scores) if scores else 0.0
    results["avg_r_rule_valid"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    results["valid_count"] = len(valid_scores)
    results["parse_errors"] = parse_errors
    results["empty_outputs"] = empty_outputs
    
    results["avg_condition"] = sum(results["condition_scores"]) / len(results["condition_scores"]) if results["condition_scores"] else 0.0
    results["avg_result"] = sum(results["result_scores"]) / len(results["result_scores"]) if results["result_scores"] else 0.0
    results["avg_entity"] = sum(results["entity_scores"]) / len(results["entity_scores"]) if results["entity_scores"] else 0.0
    
    return results


def print_results(results: List[Dict]):
    """打印评测结果"""
    print("\n" + "=" * 80)
    print("  API 模型评测结果")
    print("=" * 80)
    
    # 按 R-rule 排序
    sorted_results = sorted(results, key=lambda x: x["avg_r_rule"], reverse=True)
    
    print(f"\n{'模型':<30} {'R-rule':>8} {'Condition':>10} {'Result':>10} {'Entity':>10} {'有效样本':>8}")
    print("-" * 80)
    
    for r in sorted_results:
        print(f"{r['model']:<30} {r['avg_r_rule']:>8.4f} {r['avg_condition']:>10.4f} "
              f"{r['avg_result']:>10.4f} {r['avg_entity']:>10.4f} {r['valid_count']:>8}")
    
    print("-" * 80)
    
    # 详细统计
    print("\n详细统计:")
    for r in sorted_results:
        total = r["total_samples"]
        valid = r["valid_count"]
        parse_err = r["parse_errors"]
        empty = r["empty_outputs"]
        print(f"\n  {r['model']}:")
        print(f"    总样本数: {total}")
        print(f"    有效样本: {valid} ({100*valid/total:.1f}%)")
        print(f"    解析错误: {parse_err} ({100*parse_err/total:.1f}%)")
        print(f"    空输出: {empty} ({100*empty/total:.1f}%)")
        print(f"    平均 R-rule (所有): {r['avg_r_rule']:.4f}")
        print(f"    平均 R-rule (有效): {r['avg_r_rule_valid']:.4f}")
        print(f"    S_structure (Result): {r['avg_result']:.4f}")
        print(f"    S_content (Condition): {r['avg_condition']:.4f}")


def save_results(results: List[Dict], output_dir: Path):
    """保存评测结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"api_evaluation_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 评测结果已保存到: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="API 生成结果评测")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="API 生成结果目录 (默认: api_generation)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="评测结果输出目录 (默认: api_generation)")
    args = parser.parse_args()
    
    # 确定目录
    results_dir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "api_generation"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "api_generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  API 模型评测")
    print("=" * 80)
    print(f"结果目录: {results_dir}")
    
    # 加载结果
    model_results = load_api_results(results_dir)
    
    if not model_results:
        print("\n❌ 未找到模型结果，请先运行生成脚本")
        return
    
    # 评测每个模型
    print("\n" + "=" * 60)
    print("  开始评测...")
    print("=" * 60)
    
    all_results = []
    for model_name, model_data in model_results.items():
        print(f"\n📊 评测模型: {model_name}")
        result = evaluate_model(model_name, model_data)
        all_results.append(result)
    
    # 打印结果
    print_results(all_results)
    
    # 保存结果
    output_file = save_results(all_results, output_dir)
    
    # 同时保存为 markdown
    md_file = output_dir / f"api_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# API 模型评测结果\n\n")
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        sorted_results = sorted(all_results, key=lambda x: x["avg_r_rule"], reverse=True)
        
        f.write("## 综合对比\n\n")
        f.write("| 模型 | R-rule | Condition | Result | Entity | 有效样本 |\n")
        f.write("|------|--------|-----------|--------|--------|----------|\n")
        for r in sorted_results:
            f.write(f"| {r['model']} | {r['avg_r_rule']:.4f} | {r['avg_condition']:.4f} | "
                   f"{r['avg_result']:.4f} | {r['avg_entity']:.4f} | {r['valid_count']} |\n")
        
        f.write("\n## 详细统计\n\n")
        for r in sorted_results:
            total = r["total_samples"]
            valid = r["valid_count"]
            f.write(f"### {r['model']}\n\n")
            f.write(f"- 总样本数: {total}\n")
            f.write(f"- 有效样本: {valid} ({100*valid/total:.1f}%)\n")
            f.write(f"- 解析错误: {r['parse_errors']} ({100*r['parse_errors']/total:.1f}%)\n")
            f.write(f"- 空输出: {r['empty_outputs']} ({100*r['empty_outputs']/total:.1f}%)\n")
            f.write(f"- 平均 R-rule (所有): {r['avg_r_rule']:.4f}\n")
            f.write(f"- 平均 R-rule (有效): {r['avg_r_rule_valid']:.4f}\n\n")
    
    print(f"\n📝 Markdown 结果已保存到: {md_file}")
    print("\n✅ 评测完成！")


if __name__ == "__main__":
    main()
