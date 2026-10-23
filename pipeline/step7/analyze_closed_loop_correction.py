#!/usr/bin/env python3
"""
闭环修正前后R-rule分数变化分析脚本

分析closed_loop_raw_deepseek_*.json中每个样本的：
- 初始生成结果的R-rule分数
- 最终修正结果的R-rule分数
- 分数变化（提升/下降）
- 修正轮次与分数变化的关系
"""

import json
import os
import sys
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
sys.path.insert(0, '/home/weijie_yu/yjw/ECLO-RL/step7')

from r_rule_evaluator_final import compute_r_rule


def load_closed_loop_results(filepath):
    """加载闭环修正结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth(test_dir):
    """加载测试集ground truth"""
    ground_truth = {}
    for filename in os.listdir(test_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                sample_id = data.get('id') or data.get('sample_id') or filename.replace('.json', '')
                ground_truth[sample_id] = data
    return ground_truth


def extract_knowledge_chains(generated_data):
    """从生成数据中提取知识链"""
    if not generated_data:
        return []
    
    chains = []
    for item in generated_data.get('knowledge_mining', []):
        chain = item.get('knowledge_chain', {})
        entity = item.get('entity_recognition', {})
        chains.append({
            'condition': chain.get('condition', ''),
            'result': chain.get('result', ''),
            'entity_recognition': entity
        })
    return chains


def analyze_sample(sample, ground_truth_data=None):
    """分析单个样本的修正前后分数变化"""
    sample_id = sample.get('sample_id', 'unknown')
    
    # 直接使用闭环结果中的ground_truth
    gt_output = sample.get('ground_truth', {})
    
    if not gt_output:
        return None
    
    # 构建sample格式用于compute_r_rule
    gt_sample = {
        'ground_truth': gt_output
    }
    
    # 获取初始生成结果
    initial_generated = sample.get('initial_generated', {})
    if initial_generated:
        gt_sample['generated_json'] = initial_generated
        initial_result = compute_r_rule(gt_sample, 'generated_json')
    else:
        initial_result = None
    
    # 获取最终修正结果
    final_generated = sample.get('final_generated', {})
    if final_generated:
        gt_sample['generated_json'] = final_generated
        final_result = compute_r_rule(gt_sample, 'generated_json')
    else:
        final_result = None
    
    if not initial_result or not final_result:
        return None
    
    # 计算分数变化
    r_rule_change = final_result['R_rule'] - initial_result['R_rule']
    s_structure_change = final_result['S_structure'] - initial_result['S_structure']
    s_content_change = final_result['S_content'] - initial_result['S_content']
    
    # 获取修正历史信息
    correction_history = sample.get('correction_history', [])
    num_corrections = len([h for h in correction_history if h.get('action') == 'CORRECT'])
    
    # 获取每轮的分数
    stage_scores = []
    stages = sample.get('stages', {})
    
    # stages是字典，包含before和after
    if isinstance(stages, dict):
        for stage_name, stage_data in stages.items():
            if isinstance(stage_data, dict):
                r_rule_data = stage_data.get('methods', {}).get('r_rule', {})
                if r_rule_data:
                    stage_scores.append({
                        'stage': stage_data.get('stage', stage_name),
                        'r_rule': r_rule_data.get('R_rule', 0.0),
                        's_structure': r_rule_data.get('S_structure', 0.0),
                        's_content': r_rule_data.get('S_content', 0.0)
                    })
    elif isinstance(stages, list):
        # 兼容旧版本，可能是列表
        for stage in stages:
            stage_name = stage.get('stage', '')
            stage_corrected = stage.get('corrected_json', {})
            if stage_corrected:
                gt_sample['generated_json'] = stage_corrected
                stage_result = compute_r_rule(gt_sample, 'generated_json')
                stage_scores.append({
                    'stage': stage_name,
                    'r_rule': stage_result['R_rule'],
                    's_structure': stage_result['S_structure'],
                    's_content': stage_result['S_content']
                })
    
    return {
        'sample_id': sample_id,
        'initial_r_rule': initial_result['R_rule'],
        'final_r_rule': final_result['R_rule'],
        'r_rule_change': r_rule_change,
        'initial_s_structure': initial_result['S_structure'],
        'final_s_structure': final_result['S_structure'],
        's_structure_change': s_structure_change,
        'initial_s_content': initial_result['S_content'],
        'final_s_content': final_result['S_content'],
        's_content_change': s_content_change,
        'num_corrections': num_corrections,
        'num_initial_chains': initial_result.get('n_pred', 0),
        'num_final_chains': final_result.get('n_pred', 0),
        'num_gold_chains': initial_result.get('n_gold', 0),
        'stage_scores': stage_scores,
        'correction_history': [
            {
                'iteration': h.get('iteration'),
                'strategy': h.get('strategy'),
                'chains_before': h.get('chains_before'),
                'chains_after': h.get('chains_after'),
                'chains_change': h.get('chains_change'),
            }
            for h in correction_history if h.get('action') == 'CORRECT'
        ]
    }


def print_detailed_report(analysis_results):
    """打印详细分析报告"""
    print("\n" + "=" * 100)
    print("  闭环修正前后R-rule分数变化详细分析报告")
    print("=" * 100)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"有效样本数: {len(analysis_results)}")
    print()
    
    # 统计信息
    improvements = [r['r_rule_change'] for r in analysis_results]
    improved = sum(1 for x in improvements if x > 0.001)
    degraded = sum(1 for x in improvements if x < -0.001)
    unchanged = len(improvements) - improved - degraded
    
    print("-" * 100)
    print("  总体统计")
    print("-" * 100)
    print(f"  分数提升样本数:  {improved} ({improved/len(improvements)*100:.1f}%)")
    print(f"  分数下降样本数:  {degraded} ({degraded/len(improvements)*100:.1f}%)")
    print(f"  分数不变样本数:  {unchanged} ({unchanged/len(improvements)*100:.1f}%)")
    print()
    
    # 平均分数变化
    avg_initial = sum(r['initial_r_rule'] for r in analysis_results) / len(analysis_results)
    avg_final = sum(r['final_r_rule'] for r in analysis_results) / len(analysis_results)
    avg_change = sum(improvements) / len(improvements)
    
    print("-" * 100)
    print("  分数变化汇总")
    print("-" * 100)
    print(f"  平均初始R-rule:    {avg_initial:.4f}")
    print(f"  平均最终R-rule:    {avg_final:.4f}")
    print(f"  平均分数变化:      {avg_change:+.4f} ({avg_change/avg_initial*100:+.2f}%)")
    print()
    
    # 按分数变化分类
    print("-" * 100)
    print("  分数提升最多的样本 (Top 5)")
    print("-" * 100)
    top_improved = sorted(analysis_results, key=lambda x: x['r_rule_change'], reverse=True)[:5]
    for i, r in enumerate(top_improved, 1):
        print(f"\n  {i}. 样本 {r['sample_id']}:")
        print(f"     初始: {r['initial_r_rule']:.4f} → 最终: {r['final_r_rule']:.4f}")
        print(f"     变化: {r['r_rule_change']:+.4f} ({r['r_rule_change']/max(r['initial_r_rule'], 0.001)*100:+.2f}%)")
        print(f"     修正轮数: {r['num_corrections']}")
        print(f"     链条数: {r['num_initial_chains']} → {r['num_final_chains']} (真实: {r['num_gold_chains']})")
    
    print("\n" + "-" * 100)
    print("  分数下降最多的样本 (Top 5)")
    print("-" * 100)
    top_degraded = sorted(analysis_results, key=lambda x: x['r_rule_change'])[:5]
    for i, r in enumerate(top_degraded, 1):
        print(f"\n  {i}. 样本 {r['sample_id']}:")
        print(f"     初始: {r['initial_r_rule']:.4f} → 最终: {r['final_r_rule']:.4f}")
        print(f"     变化: {r['r_rule_change']:+.4f} ({r['r_rule_change']/max(r['initial_r_rule'], 0.001)*100:+.2f}%)")
        print(f"     修正轮数: {r['num_corrections']}")
        print(f"     链条数: {r['num_initial_chains']} → {r['num_final_chains']} (真实: {r['num_gold_chains']})")
    
    # 每个样本的详细表格
    print("\n" + "=" * 100)
    print("  所有样本详细数据")
    print("=" * 100)
    print(f"\n{'样本ID':<10} {'初始R-rule':<12} {'最终R-rule':<12} {'变化':<12} {'修正轮数':<10} {'链条数变化':<15}")
    print("-" * 100)
    
    for r in sorted(analysis_results, key=lambda x: x['r_rule_change'], reverse=True):
        chain_change = f"{r['num_initial_chains']}→{r['num_final_chains']}"
        change_str = f"{r['r_rule_change']:+.4f}"
        change_symbol = "↑" if r['r_rule_change'] > 0.001 else ("↓" if r['r_rule_change'] < -0.001 else "→")
        print(f"{r['sample_id']:<10} {r['initial_r_rule']:<12.4f} {r['final_r_rule']:<12.4f} {change_str:<12} {r['num_corrections']:<10} {chain_change:<15} {change_symbol}")
    
    return analysis_results


def generate_markdown_report(analysis_results, output_path):
    """生成Markdown格式的分析报告"""
    
    improvements = [r['r_rule_change'] for r in analysis_results]
    improved = sum(1 for x in improvements if x > 0.001)
    degraded = sum(1 for x in improvements if x < -0.001)
    unchanged = len(improvements) - improved - degraded
    
    avg_initial = sum(r['initial_r_rule'] for r in analysis_results) / len(analysis_results)
    avg_final = sum(r['final_r_rule'] for r in analysis_results) / len(analysis_results)
    avg_change = sum(improvements) / len(improvements)
    
    lines = []
    lines.append("# 闭环修正前后R-rule分数变化分析报告")
    lines.append(f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**有效样本数**: {len(analysis_results)}")
    lines.append("")
    lines.append("## 总体统计")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| 分数提升样本数 | {improved} ({improved/len(improvements)*100:.1f}%) |")
    lines.append(f"| 分数下降样本数 | {degraded} ({degraded/len(improvements)*100:.1f}%) |")
    lines.append(f"| 分数不变样本数 | {unchanged} ({unchanged/len(improvements)*100:.1f}%) |")
    lines.append(f"| 平均初始R-rule | {avg_initial:.4f} |")
    lines.append(f"| 平均最终R-rule | {avg_final:.4f} |")
    lines.append(f"| 平均分数变化 | {avg_change:+.4f} ({avg_change/avg_initial*100:+.2f}%) |")
    lines.append("")
    lines.append("## 所有样本详细数据")
    lines.append("")
    lines.append("| 样本ID | 初始R-rule | 最终R-rule | 变化 | 变化率 | 修正轮数 | 链条数(初→终) | 真实链条数 |")
    lines.append("|--------|-----------|-----------|------|-------|---------|-------------|-----------|")
    
    for r in sorted(analysis_results, key=lambda x: x['r_rule_change'], reverse=True):
        change_pct = r['r_rule_change'] / max(r['initial_r_rule'], 0.001) * 100
        change_symbol = "↑" if r['r_rule_change'] > 0.001 else ("↓" if r['r_rule_change'] < -0.001 else "→")
        lines.append(f"| {r['sample_id']} | {r['initial_r_rule']:.4f} | {r['final_r_rule']:.4f} | {r['r_rule_change']:+.4f} {change_symbol} | {change_pct:+.2f}% | {r['num_corrections']} | {r['num_initial_chains']}→{r['num_final_chains']} | {r['num_gold_chains']} |")
    
    lines.append("")
    lines.append("## 分数提升最多的样本 (Top 5)")
    lines.append("")
    top_improved = sorted(analysis_results, key=lambda x: x['r_rule_change'], reverse=True)[:5]
    for i, r in enumerate(top_improved, 1):
        change_pct = r['r_rule_change'] / max(r['initial_r_rule'], 0.001) * 100
        lines.append(f"\n### {i}. 样本 {r['sample_id']}")
        lines.append(f"- 初始R-rule: {r['initial_r_rule']:.4f}")
        lines.append(f"- 最终R-rule: {r['final_r_rule']:.4f}")
        lines.append(f"- 变化: {r['r_rule_change']:+.4f} ({change_pct:+.2f}%)")
        lines.append(f"- 修正轮数: {r['num_corrections']}")
        lines.append(f"- 链条数: {r['num_initial_chains']} → {r['num_final_chains']} (真实: {r['num_gold_chains']})")
    
    lines.append("")
    lines.append("## 分数下降最多的样本 (Top 5)")
    lines.append("")
    top_degraded = sorted(analysis_results, key=lambda x: x['r_rule_change'])[:5]
    for i, r in enumerate(top_degraded, 1):
        change_pct = r['r_rule_change'] / max(r['initial_r_rule'], 0.001) * 100
        lines.append(f"\n### {i}. 样本 {r['sample_id']}")
        lines.append(f"- 初始R-rule: {r['initial_r_rule']:.4f}")
        lines.append(f"- 最终R-rule: {r['final_r_rule']:.4f}")
        lines.append(f"- 变化: {r['r_rule_change']:+.4f} ({change_pct:+.2f}%)")
        lines.append(f"- 修正轮数: {r['num_corrections']}")
        lines.append(f"- 链条数: {r['num_initial_chains']} → {r['num_final_chains']} (真实: {r['num_gold_chains']})")
    
    lines.append("")
    lines.append("## 修正轮数与分数变化关系")
    lines.append("")
    correction_by_rounds = defaultdict(list)
    for r in analysis_results:
        correction_by_rounds[r['num_corrections']].append(r['r_rule_change'])
    
    lines.append("| 修正轮数 | 样本数 | 平均分数变化 |")
    lines.append("|---------|-------|-------------|")
    for num_rounds in sorted(correction_by_rounds.keys()):
        changes = correction_by_rounds[num_rounds]
        avg_change = sum(changes) / len(changes)
        lines.append(f"| {num_rounds} | {len(changes)} | {avg_change:+.4f} |")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\nMarkdown报告已保存到: {output_path}")
    return '\n'.join(lines)


def main():
    # 路径配置
    closed_loop_file = '/home/weijie_yu/yjw/ECLO-RL/step7/closed_loop_results/closed_loop_raw_deepseek_20260207_172058.json'
    output_dir = '/home/weijie_yu/yjw/ECLO-RL/step7/evaluation_results'
    
    print("=" * 100)
    print("  闭环修正前后R-rule分数变化分析")
    print("=" * 100)
    
    # 加载数据
    print("\n加载闭环修正结果...")
    closed_loop_data = load_closed_loop_results(closed_loop_file)
    print(f"加载了 {len(closed_loop_data)} 个闭环修正样本")
    
    # 分析每个样本
    print("\n分析样本中...")
    analysis_results = []
    for sample in closed_loop_data:
        sample_id = sample.get('sample_id', 'unknown')
        result = analyze_sample(sample)
        if result:
            analysis_results.append(result)
        else:
            print(f"  警告: 样本 {sample_id} 分析失败")
    
    print(f"成功分析了 {len(analysis_results)} 个样本")
    
    # 打印详细报告
    print_detailed_report(analysis_results)
    
    # 生成Markdown报告
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'closed_loop_correction_analysis.md')
    generate_markdown_report(analysis_results, report_path)
    
    # 保存详细数据为JSON
    json_path = os.path.join(output_dir, 'closed_loop_correction_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细数据已保存到: {json_path}")


if __name__ == '__main__':
    main()
