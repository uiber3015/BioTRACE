#!/usr/bin/env python3
"""
分析 step5 生成的数据质量，识别问题分布，为后续数据清理提供依据
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
import random

def analyze_data_quality(output_dir: Path, sample_size: int = 500):
    """分析数据质量"""
    
    files = list(output_dir.glob("*.json"))
    print(f"总文件数: {len(files)}")
    
    # 随机采样
    if len(files) > sample_size:
        sample_files = random.sample(files, sample_size)
        print(f"采样文件数: {sample_size}")
    else:
        sample_files = files
        print(f"使用全部文件: {len(files)}")
    
    stats = {
        'total_files': len(files),
        'sampled_files': len(sample_files),
        'files_processed': 0,
        'files_skipped': 0,
        
        # 链级别统计
        'total_chains': 0,
        'chains_with_condition': 0,
        'chains_without_condition': 0,
        'chains_with_result': 0,
        'chains_without_result': 0,
        'chains_with_entities': 0,
        'chains_without_entities': 0,
        'chains_complete': 0,  # condition + result + entity
        'chains_partial': 0,  # 只有部分字段
        
        # result 质量
        'result_format_valid': 0,
        'result_format_invalid': 0,
        'result_with_indicator': 0,  # 包含指标名
        'result_without_indicator': 0,  # null:数值
        
        # condition 质量
        'condition_length_dist': Counter(),  # condition长度分布
        'condition_parts_dist': Counter(),  # condition片段数量分布
        
        # 文件级别统计
        'files_all_empty_result': 0,
        'files_some_result': 0,
        'files_all_result': 0,
        'files_no_chains': 0,
        
        # 实体类型统计
        'entity_types': Counter(),
        'chains_by_entity_count': Counter(),  # 按实体数量分组
        
        # 问题分类
        'problem_categories': {
            'no_condition': 0,
            'no_result': 0,
            'no_entity': 0,
            'no_condition_no_result': 0,
            'no_result_no_entity': 0,
            'all_empty': 0,
        }
    }
    
    # 详细问题记录
    problem_examples = {
        'no_result': [],
        'no_entity': [],
        'no_condition': [],
        'result_format_issue': [],
    }
    
    for f in sample_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            
            knowledge_mining = data.get('knowledge_mining', [])
            if not knowledge_mining:
                stats['files_no_chains'] += 1
                continue
            
            stats['files_processed'] += 1
            
            file_result_count = 0
            file_total_chains = len(knowledge_mining)
            
            for km in knowledge_mining:
                stats['total_chains'] += 1
                
                kc = km.get('knowledge_chain', {})
                er = km.get('entity_recognition', {})
                
                condition = kc.get('condition', '').strip()
                result = kc.get('result', '').strip()
                
                # Condition 统计
                if condition:
                    stats['chains_with_condition'] += 1
                    # condition 长度和片段数
                    parts = [p.strip() for p in condition.split('|') if p.strip()]
                    stats['condition_parts_dist'][len(parts)] += 1
                    stats['condition_length_dist'][len(condition)] += 1
                else:
                    stats['chains_without_condition'] += 1
                    stats['problem_categories']['no_condition'] += 1
                    if len(problem_examples['no_condition']) < 5:
                        problem_examples['no_condition'].append({
                            'file': f.name,
                            'condition': condition,
                            'result': result
                        })
                
                # Result 统计
                if result:
                    stats['chains_with_result'] += 1
                    file_result_count += 1
                    
                    # result 格式检查
                    if ':' in result:
                        stats['result_format_valid'] += 1
                        # 检查是否有指标名（不是 null:数值）
                        parts = result.split('|')
                        has_indicator = False
                        for part in parts:
                            part = part.strip()
                            if ':' in part:
                                indicator = part.split(':', 1)[0].strip()
                                if indicator and indicator.lower() != 'null':
                                    has_indicator = True
                                    break
                        
                        if has_indicator:
                            stats['result_with_indicator'] += 1
                        else:
                            stats['result_without_indicator'] += 1
                    else:
                        stats['result_format_invalid'] += 1
                        if len(problem_examples['result_format_issue']) < 5:
                            problem_examples['result_format_issue'].append({
                                'file': f.name,
                                'result': result
                            })
                else:
                    stats['chains_without_result'] += 1
                    stats['problem_categories']['no_result'] += 1
                    if len(problem_examples['no_result']) < 5:
                        problem_examples['no_result'].append({
                            'file': f.name,
                            'condition': condition[:100] if condition else '',
                        })
                
                # Entity 统计
                has_entities = False
                entity_count = 0
                if er and isinstance(er, dict):
                    for entity_type, entities in er.items():
                        if entities:
                            if isinstance(entities, list):
                                entity_count += len(entities)
                                stats['entity_types'][entity_type] += len(entities)
                            else:
                                entity_count += 1
                                stats['entity_types'][entity_type] += 1
                            has_entities = True
                
                if has_entities:
                    stats['chains_with_entities'] += 1
                    stats['chains_by_entity_count'][entity_count] += 1
                else:
                    stats['chains_without_entities'] += 1
                    stats['problem_categories']['no_entity'] += 1
                    if len(problem_examples['no_entity']) < 5:
                        problem_examples['no_entity'].append({
                            'file': f.name,
                            'condition': condition[:100] if condition else '',
                            'result': result[:100] if result else '',
                        })
                
                # 完整性统计
                if condition and result and has_entities:
                    stats['chains_complete'] += 1
                else:
                    stats['chains_partial'] += 1
                
                # 问题组合统计
                if not condition and not result:
                    stats['problem_categories']['no_condition_no_result'] += 1
                if not result and not has_entities:
                    stats['problem_categories']['no_result_no_entity'] += 1
                if not condition and not result and not has_entities:
                    stats['problem_categories']['all_empty'] += 1
            
            # 文件级别统计
            if file_result_count == 0:
                stats['files_all_empty_result'] += 1
            elif file_result_count == file_total_chains:
                stats['files_all_result'] += 1
            else:
                stats['files_some_result'] += 1
                
        except Exception as e:
            stats['files_skipped'] += 1
            print(f"Error processing {f.name}: {e}")
    
    return stats, problem_examples


def print_analysis_report(stats: Dict, problem_examples: Dict):
    """打印分析报告"""
    
    print("\n" + "="*80)
    print("数据质量分析报告")
    print("="*80)
    
    print(f"\n【总体统计】")
    print(f"总文件数: {stats['total_files']}")
    print(f"采样文件数: {stats['sampled_files']}")
    print(f"成功处理: {stats['files_processed']}")
    print(f"跳过文件: {stats['files_skipped']}")
    print(f"无知识链文件: {stats['files_no_chains']}")
    print(f"总知识链数: {stats['total_chains']}")
    
    print(f"\n【知识链完整性统计】")
    print(f"完整链 (condition+result+entity): {stats['chains_complete']} ({stats['chains_complete']/max(stats['total_chains'],1)*100:.1f}%)")
    print(f"部分链: {stats['chains_partial']} ({stats['chains_partial']/max(stats['total_chains'],1)*100:.1f}%)")
    
    print(f"\n【Condition 质量】")
    print(f"有 condition: {stats['chains_with_condition']} ({stats['chains_with_condition']/max(stats['total_chains'],1)*100:.1f}%)")
    print(f"无 condition: {stats['chains_without_condition']} ({stats['chains_without_condition']/max(stats['total_chains'],1)*100:.1f}%)")
    if stats['condition_parts_dist']:
        print(f"\nCondition 片段数分布 (Top 5):")
        for parts, count in stats['condition_parts_dist'].most_common(5):
            print(f"  {parts} 个片段: {count} 条链 ({count/max(stats['chains_with_condition'],1)*100:.1f}%)")
    
    print(f"\n【Result 质量】")
    print(f"有 result: {stats['chains_with_result']} ({stats['chains_with_result']/max(stats['total_chains'],1)*100:.1f}%)")
    print(f"无 result: {stats['chains_without_result']} ({stats['chains_without_result']/max(stats['total_chains'],1)*100:.1f}%)")
    print(f"result 格式有效 (含:): {stats['result_format_valid']} ({stats['result_format_valid']/max(stats['chains_with_result'],1)*100:.1f}%)")
    print(f"result 有指标名: {stats['result_with_indicator']} ({stats['result_with_indicator']/max(stats['result_format_valid'],1)*100:.1f}%)")
    print(f"result 无指标名 (null:数值): {stats['result_without_indicator']} ({stats['result_without_indicator']/max(stats['result_format_valid'],1)*100:.1f}%)")
    
    print(f"\n【Entity 质量】")
    print(f"有 entity: {stats['chains_with_entities']} ({stats['chains_with_entities']/max(stats['total_chains'],1)*100:.1f}%)")
    print(f"无 entity: {stats['chains_without_entities']} ({stats['chains_without_entities']/max(stats['total_chains'],1)*100:.1f}%)")
    if stats['chains_by_entity_count']:
        print(f"\n实体数量分布 (Top 5):")
        for count, chain_count in sorted(stats['chains_by_entity_count'].items())[:5]:
            print(f"  {count} 个实体: {chain_count} 条链")
    
    print(f"\n【文件级别统计】")
    print(f"所有链 result 都为空: {stats['files_all_empty_result']} ({stats['files_all_empty_result']/max(stats['files_processed'],1)*100:.1f}%)")
    print(f"部分链有 result: {stats['files_some_result']} ({stats['files_some_result']/max(stats['files_processed'],1)*100:.1f}%)")
    print(f"所有链都有 result: {stats['files_all_result']} ({stats['files_all_result']/max(stats['files_processed'],1)*100:.1f}%)")
    
    print(f"\n【问题分类统计】")
    for problem, count in stats['problem_categories'].items():
        if count > 0:
            print(f"{problem}: {count} ({count/max(stats['total_chains'],1)*100:.1f}%)")
    
    print(f"\n【实体类型分布 (Top 15)】")
    for entity_type, count in stats['entity_types'].most_common(15):
        print(f"  {entity_type}: {count}")
    
    print(f"\n【问题示例】")
    if problem_examples['no_result']:
        print(f"\n无 result 示例 (前3个):")
        for ex in problem_examples['no_result'][:3]:
            print(f"  文件: {ex['file']}")
            print(f"  condition: {ex.get('condition', '')[:80]}...")
    
    if problem_examples['no_entity']:
        print(f"\n无 entity 示例 (前3个):")
        for ex in problem_examples['no_entity'][:3]:
            print(f"  文件: {ex['file']}")
            print(f"  condition: {ex.get('condition', '')[:60]}...")
            print(f"  result: {ex.get('result', '')[:60]}...")


def suggest_filtering_criteria(stats: Dict) -> Dict[str, Any]:
    """根据统计数据建议筛选标准"""
    
    suggestions = {
        'strict': {
            'description': '严格标准：用于高质量训练数据',
            'criteria': {
                'require_condition': True,
                'require_result': True,
                'require_entity': True,
                'require_result_format': True,  # 必须包含 :
                'min_entity_count': 1,
            },
            'estimated_retention': stats['chains_complete'] / max(stats['total_chains'], 1) * 100
        },
        'moderate': {
            'description': '中等标准：允许无entity，但必须有condition和result',
            'criteria': {
                'require_condition': True,
                'require_result': True,
                'require_entity': False,
                'require_result_format': True,
                'min_entity_count': 0,
            },
            'estimated_retention': (stats['chains_with_condition'] * stats['chains_with_result']) / max(stats['total_chains']**2, 1) * 100
        },
        'loose': {
            'description': '宽松标准：只要有condition或result即可',
            'criteria': {
                'require_condition': False,
                'require_result': False,
                'require_entity': False,
                'require_result_format': False,
                'min_entity_count': 0,
            },
            'estimated_retention': 100.0
        },
        'result_focused': {
            'description': '结果导向：必须有result（用于需要数值的训练）',
            'criteria': {
                'require_condition': False,  # 可选
                'require_result': True,
                'require_entity': False,
                'require_result_format': True,
                'min_entity_count': 0,
            },
            'estimated_retention': stats['chains_with_result'] / max(stats['total_chains'], 1) * 100
        }
    }
    
    return suggestions


def main():
    output_dir = Path("gene_entity_output_v1")
    
    if not output_dir.exists():
        print(f"错误: 目录 {output_dir} 不存在")
        return
    
    print("开始分析数据质量...")
    stats, problem_examples = analyze_data_quality(output_dir, sample_size=500)
    
    print_analysis_report(stats, problem_examples)
    
    print("\n" + "="*80)
    print("筛选标准建议")
    print("="*80)
    
    suggestions = suggest_filtering_criteria(stats)
    for level, suggestion in suggestions.items():
        print(f"\n【{level.upper()}】")
        print(f"描述: {suggestion['description']}")
        print(f"标准:")
        for key, value in suggestion['criteria'].items():
            print(f"  {key}: {value}")
        print(f"预计保留率: {suggestion['estimated_retention']:.1f}%")


if __name__ == "__main__":
    main()

