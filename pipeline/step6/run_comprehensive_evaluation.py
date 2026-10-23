"""
综合评估脚本 - 使用三种评测方法对比基座模型和SFT模型
"""
import json
import sys
sys.path.insert(0, 'step6')
sys.path.insert(0, 'step7')

from collections import Counter
from step6.evaluate_simplified import (
    evaluate_basic_validity,
    evaluate_chain_matching,
    evaluate_structure_integrity,
    evaluate_redundancy,
    calculate_overall_score,
    classify_quality_level
)
from step6.rule_based_evaluator import (
    split_condition_to_kv_pairs,
    split_result_to_kv,
    jaccard_pairs,
    best_chain_match
)
from step7.r_rule_evaluator import (
    compute_r_rule
)

print("=" * 80)
print("       综合评估脚本 - 基座模型 vs SFT模型 三种评测方法对比")
print("=" * 80)

# 加载共同成功样本
with open('step6/test_results/common_success_samples.json') as f:
    common_samples = json.load(f)

print(f"\n📊 加载了 {len(common_samples)} 个两个模型都成功生成的样本")

# ==================== 评测方法1: evaluate_simplified ====================
print("\n" + "=" * 80)
print("【评测方法1】evaluate_simplified - 简化版知识挖掘评估")
print("=" * 80)

def eval_simplified(sample):
    """使用evaluate_simplified方法评估"""
    generated_km = sample['generated_json'].get('knowledge_mining', [])
    ground_truth_km = sample.get('ground_truth', {}).get('knowledge_mining', [])
    
    basic_metrics = evaluate_basic_validity(generated_km, ground_truth_km)
    matching_metrics = evaluate_chain_matching(generated_km, ground_truth_km)
    structure_score = evaluate_structure_integrity(generated_km)
    redundancy_score = evaluate_redundancy(generated_km)
    
    overall = calculate_overall_score(
        basic_metrics['score'],
        matching_metrics['match_score'],
        structure_score,
        redundancy_score
    )
    
    return {
        'basic_score': basic_metrics['score'],
        'match_score': matching_metrics['match_score'],
        'structure_score': structure_score,
        'redundancy_score': redundancy_score,
        'overall': overall,
        'quality_level': classify_quality_level(overall),
        'match_ratio': matching_metrics['match_ratio'],
        'count_ratio': basic_metrics['count_ratio']
    }

# 评估所有样本
base_simplified = []
trained_simplified = []

for i, s in enumerate(common_samples):
    # 基座模型
    base_result = eval_simplified(s['base'])
    base_simplified.append(base_result)
    
    # SFT模型
    trained_result = eval_simplified(s['trained'])
    trained_simplified.append(trained_result)
    
    if (i + 1) % 500 == 0:
        print(f"  评估进度: {i+1}/{len(common_samples)}")

print(f"\n✅ 评估完成! 共 {len(common_samples)} 个样本")

# 统计结果
def calc_stats(results):
    scores = [r['overall'] for r in results]
    return {
        'avg': sum(scores) / len(scores),
        'min': min(scores),
        'max': max(scores),
        'levels': Counter(r['quality_level'] for r in results),
        'avg_match_ratio': sum(r['match_ratio'] for r in results) / len(results),
        'avg_count_ratio': sum(r['count_ratio'] for r in results) / len(results)
    }

base_stats = calc_stats(base_simplified)
trained_stats = calc_stats(trained_simplified)

print(f"\n基座模型 (Base):")
print(f"  平均综合得分: {base_stats['avg']:.4f}")
print(f"  得分范围: [{base_stats['min']:.4f}, {base_stats['max']:.4f}]")
print(f"  平均匹配比例: {base_stats['avg_match_ratio']:.2%}")
print(f"  平均数量比例: {base_stats['avg_count_ratio']:.2f}")

print(f"\nSFT模型 (Trained):")
print(f"  平均综合得分: {trained_stats['avg']:.4f}")
print(f"  得分范围: [{trained_stats['min']:.4f}, {trained_stats['max']:.4f}]")
print(f"  平均匹配比例: {trained_stats['avg_match_ratio']:.2%}")
print(f"  平均数量比例: {trained_stats['avg_count_ratio']:.2f}")

print(f"\n质量等级分布:")
print(f"  基座模型: {dict(base_stats['levels'])}")
print(f"  SFT模型:  {dict(trained_stats['levels'])}")

# ==================== 评测方法2: rule_based_evaluator ====================
print("\n" + "=" * 80)
print("【评测方法2】rule_based_evaluator - 基于规则的评估")
print("=" * 80)

def eval_rule_based(sample):
    """使用rule_based_evaluator评估"""
    generated_km = sample['generated_json'].get('knowledge_mining', [])
    ground_truth_km = sample.get('ground_truth', {}).get('knowledge_mining', [])
    
    if not generated_km:
        return {'r_rule': 0.0, 's_structure': 0.0, 's_content': 0.0}
    
    total_r = 0.0
    total_s_struct = 0.0
    total_s_content = 0.0
    count = 0
    
    for gen_chain in generated_km:
        match_result = best_chain_match(gen_chain, ground_truth_km, 'strict')
        total_r += match_result.get('r_rule', 0.0)
        total_s_struct += match_result.get('s_structure', 0.0)
        total_s_content += match_result.get('s_content', 0.0)
        count += 1
    
    return {
        'r_rule': total_r / count if count > 0 else 0.0,
        's_structure': total_s_struct / count if count > 0 else 0.0,
        's_content': total_s_content / count if count > 0 else 0.0
    }

base_rule = []
trained_rule = []

for i, s in enumerate(common_samples):
    base_result = eval_rule_based(s['base'])
    base_rule.append(base_result)
    
    trained_result = eval_rule_based(s['trained'])
    trained_rule.append(trained_result)
    
    if (i + 1) % 500 == 0:
        print(f"  评估进度: {i+1}/{len(common_samples)}")

print(f"\n✅ 评估完成!")

base_rule_stats = {
    'avg_r_rule': sum(r['r_rule'] for r in base_rule) / len(base_rule),
    'avg_s_structure': sum(r['s_structure'] for r in base_rule) / len(base_rule),
    'avg_s_content': sum(r['s_content'] for r in base_rule) / len(base_rule)
}

trained_rule_stats = {
    'avg_r_rule': sum(r['r_rule'] for r in trained_rule) / len(trained_rule),
    'avg_s_structure': sum(r['s_structure'] for r in trained_rule) / len(trained_rule),
    'avg_s_content': sum(r['s_content'] for r in trained_rule) / len(trained_rule)
}

print(f"\n基座模型 (Base):")
print(f"  平均 R_rule: {base_rule_stats['avg_r_rule']:.4f}")
print(f"  平均 S_structure: {base_rule_stats['avg_s_structure']:.4f}")
print(f"  平均 S_content: {base_rule_stats['avg_s_content']:.4f}")

print(f"\nSFT模型 (Trained):")
print(f"  平均 R_rule: {trained_rule_stats['avg_r_rule']:.4f}")
print(f"  平均 S_structure: {trained_rule_stats['avg_s_structure']:.4f}")
print(f"  平均 S_content: {trained_rule_stats['avg_s_content']:.4f}")

# ==================== 评测方法3: r_rule_evaluator ====================
print("\n" + "=" * 80)
print("【评测方法3】r_rule_evaluator - R_rule严格评估")
print("=" * 80)

def eval_r_rule(sample):
    """使用r_rule_evaluator评估"""
    result = compute_r_rule(sample)
    return {
        'r_rule': result.get('R_rule', 0.0),
        's_structure': result.get('S_structure', 0.0),
        's_content': result.get('S_content', 0.0)
    }

base_rrule = []
trained_rrule = []

for i, s in enumerate(common_samples):
    base_result = eval_r_rule(s['base'])
    base_rrule.append(base_result)
    
    trained_result = eval_r_rule(s['trained'])
    trained_rrule.append(trained_result)
    
    if (i + 1) % 500 == 0:
        print(f"  评估进度: {i+1}/{len(common_samples)}")

print(f"\n✅ 评估完成!")

base_rrule_stats = {
    'avg_r_rule': sum(r['r_rule'] for r in base_rrule) / len(base_rrule),
    'avg_s_structure': sum(r['s_structure'] for r in base_rrule) / len(base_rrule),
    'avg_s_content': sum(r['s_content'] for r in base_rrule) / len(base_rrule)
}

trained_rrule_stats = {
    'avg_r_rule': sum(r['r_rule'] for r in trained_rrule) / len(trained_rrule),
    'avg_s_structure': sum(r['s_structure'] for r in trained_rrule) / len(trained_rrule),
    'avg_s_content': sum(r['s_content'] for r in trained_rrule) / len(trained_rrule)
}

print(f"\n基座模型 (Base):")
print(f"  平均 R_rule: {base_rrule_stats['avg_r_rule']:.4f}")
print(f"  平均 S_structure: {base_rrule_stats['avg_s_structure']:.4f}")
print(f"  平均 S_content: {base_rrule_stats['avg_s_content']:.4f}")

print(f"\nSFT模型 (Trained):")
print(f"  平均 R_rule: {trained_rrule_stats['avg_r_rule']:.4f}")
print(f"  平均 S_structure: {trained_rrule_stats['avg_s_structure']:.4f}")
print(f"  平均 S_content: {trained_rrule_stats['avg_s_content']:.4f}")

# ==================== 最终汇总 ====================
print("\n" + "=" * 80)
print("                    最终评测结果汇总")
print("=" * 80)

print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
print("│                        基座模型 (Base)  vs  SFT模型 (Trained)                │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print(f"│ 【方法1】evaluate_simplified                                                │")
print(f"│   综合得分:     {base_stats['avg']:.4f}  vs  {trained_stats['avg']:.4f}  ({trained_stats['avg']-base_stats['avg']:+.4f})       │")
print(f"│   匹配比例:     {base_stats['avg_match_ratio']:.2%}  vs  {trained_stats['avg_match_ratio']:.2%}  ({trained_stats['avg_match_ratio']-base_stats['avg_match_ratio']:+.2%})     │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print(f"│ 【方法2】rule_based_evaluator (strict模式)                                  │")
print(f"│   R_rule:       {base_rule_stats['avg_r_rule']:.4f}  vs  {trained_rule_stats['avg_r_rule']:.4f}  ({trained_rule_stats['avg_r_rule']-base_rule_stats['avg_r_rule']:+.4f})       │")
print(f"│   S_structure:  {base_rule_stats['avg_s_structure']:.4f}  vs  {trained_rule_stats['avg_s_structure']:.4f}  ({trained_rule_stats['avg_s_structure']-base_rule_stats['avg_s_structure']:+.4f})       │")
print(f"│   S_content:    {base_rule_stats['avg_s_content']:.4f}  vs  {trained_rule_stats['avg_s_content']:.4f}  ({trained_rule_stats['avg_s_content']-base_rule_stats['avg_s_content']:+.4f})       │")
print("├─────────────────────────────────────────────────────────────────────────────┤")
print(f"│ 【方法3】r_rule_evaluator                                                    │")
print(f"│   R_rule:       {base_rrule_stats['avg_r_rule']:.4f}  vs  {trained_rrule_stats['avg_r_rule']:.4f}  ({trained_rrule_stats['avg_r_rule']-base_rrule_stats['avg_r_rule']:+.4f})       │")
print(f"│   S_structure:  {base_rrule_stats['avg_s_structure']:.4f}  vs  {trained_rrule_stats['avg_s_structure']:.4f}  ({trained_rrule_stats['avg_s_structure']-base_rrule_stats['avg_s_structure']:+.4f})       │")
print(f"│   S_content:    {base_rrule_stats['avg_s_content']:.4f}  vs  {trained_rrule_stats['avg_s_content']:.4f}  ({trained_rrule_stats['avg_s_content']-base_rrule_stats['avg_s_content']:+.4f})       │")
print("└─────────────────────────────────────────────────────────────────────────────┘")

print("\n✅ 评估完成!")

