#!/usr/bin/env python3
"""
批量R-rule评测脚本
测试三种结果：
1. 基座模型 (Base Model)
2. SFT模型 (Trained Model)
3. 闭环修正结果 (Closed-Loop Correction)
"""
import json
import argparse
import sys
import os
from datetime import datetime
from typing import Dict, Any, List

# 添加step7目录到路径
sys.path.insert(0, '/home/weijie_yu/yjw/ECLO-RL/step7')
from r_rule_evaluator_final import evaluate_file_final, print_evaluation_report


def evaluate_all(
    base_results_path: str,
    trained_results_path: str,
    closed_loop_path: str,
    test_dir: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    评估所有结果并生成对比报告
    
    Args:
        base_results_path: 基座模型结果文件路径
        trained_results_path: SFT模型结果文件路径
        closed_loop_path: 闭环修正结果文件路径
        test_dir: 测试集目录（包含ground truth）
        output_dir: 输出目录
    """
    
    results = {}
    
    # 1. 加载测试集ground truth
    print("=" * 70)
    print("加载测试集ground truth...")
    print("=" * 70)
    
    test_samples = {}
    if os.path.isdir(test_dir):
        for filename in os.listdir(test_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(test_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    file_id = data.get('file_id', filename.replace('.json', ''))
                    # 提取ground truth
                    test_samples[file_id] = {
                        'file_id': file_id,
                        'ground_truth': {
                            'knowledge_mining': data.get('knowledge_mining', [])
                        }
                    }
    elif test_dir.endswith('.json'):
        with open(test_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for sample in data:
                    file_id = sample.get('file_id', 'unknown')
                    test_samples[file_id] = {
                        'file_id': file_id,
                        'ground_truth': {
                            'knowledge_mining': sample.get('knowledge_mining', [])
                        }
                    }
    
    print(f"加载了 {len(test_samples)} 个测试样本")
    
    # 2. 评估基座模型结果
    print("\n" + "=" * 70)
    print("评测 1: 基座模型 (Base Model)")
    print("=" * 70)
    
    if os.path.exists(base_results_path):
        with open(base_results_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 合并ground truth
        if isinstance(base_data, list):
            for sample in base_data:
                file_id = sample.get('file_id', 'unknown')
                if file_id in test_samples:
                    sample['ground_truth'] = test_samples[file_id]['ground_truth']
        
        # 临时写入合并后的数据
        temp_path = '/tmp/base_merged.json'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        base_stats = evaluate_file_final(temp_path)
        print_evaluation_report(base_stats, "基座模型 (Base Model) R-rule 评测结果")
        results['base'] = base_stats
    else:
        print(f"文件不存在: {base_results_path}")
        results['base'] = None
    
    # 3. 评估SFT模型结果
    print("\n" + "=" * 70)
    print("评测 2: SFT模型 (Trained Model)")
    print("=" * 70)
    
    if os.path.exists(trained_results_path):
        with open(trained_results_path, 'r', encoding='utf-8') as f:
            trained_data = json.load(f)
        
        # 合并ground truth
        if isinstance(trained_data, list):
            for sample in trained_data:
                file_id = sample.get('file_id', 'unknown')
                if file_id in test_samples:
                    sample['ground_truth'] = test_samples[file_id]['ground_truth']
        
        # 临时写入合并后的数据
        temp_path = '/tmp/trained_merged.json'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(trained_data, f, ensure_ascii=False, indent=2)
        
        trained_stats = evaluate_file_final(temp_path)
        print_evaluation_report(trained_stats, "SFT模型 (Trained Model) R-rule 评测结果")
        results['trained'] = trained_stats
    else:
        print(f"文件不存在: {trained_results_path}")
        results['trained'] = None
    
    # 4. 评估闭环修正结果
    print("\n" + "=" * 70)
    print("评测 3: 闭环修正结果 (Closed-Loop Correction)")
    print("=" * 70)
    
    if os.path.exists(closed_loop_path):
        with open(closed_loop_path, 'r', encoding='utf-8') as f:
            closed_data = json.load(f)
        
        # 转换闭环数据格式以适配评测函数
        if isinstance(closed_data, list):
            for sample in closed_data:
                file_id = sample.get('sample_id', 'unknown')
                if file_id in test_samples:
                    sample['ground_truth'] = test_samples[file_id]['ground_truth']
                
                # 使用final_generated作为generated_json
                if 'final_generated' in sample:
                    sample['generated_json'] = sample['final_generated']
        
        # 临时写入转换后的数据
        temp_path = '/tmp/closed_merged.json'
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(closed_data, f, ensure_ascii=False, indent=2)
        
        closed_stats = evaluate_file_final(temp_path)
        print_evaluation_report(closed_stats, "闭环修正结果 (Closed-Loop Correction) R-rule 评测结果")
        results['closed_loop'] = closed_stats
    else:
        print(f"文件不存在: {closed_loop_path}")
        results['closed_loop'] = None
    
    # 5. 生成对比报告
    print("\n" + "=" * 70)
    print("综合对比报告")
    print("=" * 70)
    
    report_lines = []
    report_lines.append("# R-rule 评测综合对比报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## 评测结果概览")
    report_lines.append("")
    report_lines.append("| 指标 | 基座模型 | SFT模型 | 闭环修正 |")
    report_lines.append("|------|---------|---------|----------|")
    
    if results['base']:
        report_lines.append(f"| 样本数 | {results['base']['total_samples']} | {results['trained']['total_samples'] if results['trained'] else '-'} | {results['closed_loop']['total_samples'] if results['closed_loop'] else '-'} |")
        report_lines.append(f"| Avg R_rule | {results['base']['avg_R_rule']:.4f} | {results['trained']['avg_R_rule']:.4f} | {results['closed_loop']['avg_R_rule']:.4f} |")
        report_lines.append(f"| Avg S_structure | {results['base']['avg_S_structure']:.4f} | {results['trained']['avg_S_structure']:.4f} | {results['closed_loop']['avg_S_structure']:.4f} |")
        report_lines.append(f"| Avg S_content | {results['base']['avg_S_content']:.4f} | {results['trained']['avg_S_content']:.4f} | {results['closed_loop']['avg_S_content']:.4f} |")
        report_lines.append(f"| Avg Pred Chains | {results['base']['avg_pred_chains']:.2f} | {results['trained']['avg_pred_chains']:.2f} | {results['closed_loop']['avg_pred_chains']:.2f} |")
        report_lines.append(f"| Avg Gold Chains | {results['base']['avg_gold_chains']:.2f} | {results['trained']['avg_gold_chains']:.2f} | {results['closed_loop']['avg_gold_chains']:.2f} |")
    
    report_lines.append("")
    report_lines.append("## 改进幅度")
    report_lines.append("")
    
    if results['base'] and results['trained']:
        r_rule_improvement = (results['trained']['avg_R_rule'] - results['base']['avg_R_rule']) / max(results['base']['avg_R_rule'], 0.001) * 100
        s_content_improvement = (results['trained']['avg_S_content'] - results['base']['avg_S_content']) / max(results['base']['avg_S_content'], 0.001) * 100
        report_lines.append(f"- SFT模型 vs 基座模型:")
        report_lines.append(f"  - R_rule 提升: {r_rule_improvement:+.2f}%")
        report_lines.append(f"  - S_content 提升: {s_content_improvement:+.2f}%")
    
    if results['trained'] and results['closed_loop']:
        r_rule_improvement = (results['closed_loop']['avg_R_rule'] - results['trained']['avg_R_rule']) / max(results['trained']['avg_R_rule'], 0.001) * 100
        s_content_improvement = (results['closed_loop']['avg_S_content'] - results['trained']['avg_S_content']) / max(results['trained']['avg_S_content'], 0.001) * 100
        report_lines.append(f"- 闭环修正 vs SFT模型:")
        report_lines.append(f"  - R_rule 提升: {r_rule_improvement:+.2f}%")
        report_lines.append(f"  - S_content 提升: {s_content_improvement:+.2f}%")
    
    report_lines.append("")
    report_lines.append("## 评测算法说明")
    report_lines.append("")
    report_lines.append("- **匹配算法**: 匈牙利算法进行双向最优匹配")
    report_lines.append("- **匹配策略**: 逐项精确匹配（匹配数/总数）")
    report_lines.append("- **归一化**: 大小写归一、去除空格、只保留字母和数字")
    report_lines.append("- **汇总策略**:")
    report_lines.append("  - 预测链多 → 按预测链数目平均")
    report_lines.append("  - 预测链少 → 按真实链数目平均")
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'r_rule_comparison_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n报告已保存到: {report_path}")
    
    # 保存详细结果（使用JSONEncoder处理numpy类型）
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    detailed_results = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_model': results['base'],
        'trained_model': results['trained'],
        'closed_loop': results['closed_loop'],
    }
    results_path = os.path.join(output_dir, 'r_rule_comparison_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    print(f"详细结果已保存到: {results_path}")


def _convert_to_native_types(obj: Any) -> Any:
    """将numpy类型转换为Python原生类型"""
    import numpy as np
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _convert_to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_native_types(item) for item in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description='批量R-rule评测脚本')
    parser.add_argument('--base', default='/home/weijie_yu/yjw/ECLO-RL/step6/test_results/results_base.json',
                        help='基座模型结果文件路径')
    parser.add_argument('--trained', default='/home/weijie_yu/yjw/ECLO-RL/step6/test_results/results_trained.json',
                        help='SFT模型结果文件路径')
    parser.add_argument('--closed-loop', 
                        default='/home/weijie_yu/yjw/ECLO-RL/step7/closed_loop_results/closed_loop_raw_deepseek_20260207_172058.json',
                        help='闭环修正结果文件路径')
    parser.add_argument('--test-dir', default='/home/weijie_yu/yjw/ECLO-RL/step6/Test',
                        help='测试集目录')
    parser.add_argument('--output-dir', default='/home/weijie_yu/yjw/ECLO-RL/step7/evaluation_results',
                        help='输出目录')
    
    args = parser.parse_args()
    
    evaluate_all(
        args.base,
        args.trained,
        args.closed_loop,
        args.test_dir,
        args.output_dir
    )


if __name__ == '__main__':
    main()

