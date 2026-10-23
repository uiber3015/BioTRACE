#!/usr/bin/env python3
"""
数据清理和筛选脚本
根据不同的筛选标准清理 step5 生成的数据，为后续训练准备数据集
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import argparse


class DataFilter:
    """数据筛选器"""
    
    def __init__(self, criteria: Dict[str, Any]):
        """
        Args:
            criteria: 筛选标准
                - require_condition: bool, 是否必须有condition
                - require_result: bool, 是否必须有result
                - require_entity: bool, 是否必须有entity
                - require_result_format: bool, result是否必须包含":"
                - min_entity_count: int, 最小实体数量
        """
        self.criteria = criteria
    
    def count_entities(self, entity_recognition: Dict) -> int:
        """统计实体数量"""
        if not entity_recognition or not isinstance(entity_recognition, dict):
            return 0
        count = 0
        for entities in entity_recognition.values():
            if isinstance(entities, list):
                count += len(entities)
            elif entities:
                count += 1
        return count
    
    def validate_chain(self, knowledge_chain: Dict, entity_recognition: Dict) -> tuple[bool, str]:
        """
        验证知识链是否符合筛选标准
        
        Returns:
            (is_valid, reason)
        """
        condition = knowledge_chain.get('condition', '').strip()
        result = knowledge_chain.get('result', '').strip()
        
        # 检查 condition
        if self.criteria.get('require_condition', False):
            if not condition:
                return False, 'no_condition'
        
        # 检查 result
        if self.criteria.get('require_result', False):
            if not result:
                return False, 'no_result'
            
            # 检查 result 格式
            if self.criteria.get('require_result_format', False):
                if ':' not in result:
                    return False, 'invalid_result_format'
        
        # 检查 entity
        if self.criteria.get('require_entity', False):
            entity_count = self.count_entities(entity_recognition)
            if entity_count < self.criteria.get('min_entity_count', 1):
                return False, 'insufficient_entities'
        
        return True, 'valid'
    
    def filter_file(self, file_data: Dict) -> tuple[List[Dict], Dict]:
        """
        筛选文件中的知识链
        
        Returns:
            (valid_chains, stats)
        """
        knowledge_mining = file_data.get('knowledge_mining', [])
        
        valid_chains = []
        stats = {
            'total': len(knowledge_mining),
            'valid': 0,
            'rejected': 0,
            'rejection_reasons': defaultdict(int)
        }
        
        for km in knowledge_mining:
            kc = km.get('knowledge_chain', {})
            er = km.get('entity_recognition', {})
            
            is_valid, reason = self.validate_chain(kc, er)
            
            if is_valid:
                valid_chains.append(km)
                stats['valid'] += 1
            else:
                stats['rejected'] += 1
                stats['rejection_reasons'][reason] += 1
        
        return valid_chains, stats


def load_entity_whitelist(whitelist_path: Optional[Path] = None) -> Optional[set]:
    """加载实体类型白名单（如果存在）"""
    if whitelist_path and whitelist_path.exists():
        try:
            with open(whitelist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    entity_types = data.get('entity_types', [])
                elif isinstance(data, list):
                    entity_types = data
                else:
                    return None
                return set(entity_types)
        except Exception as e:
            print(f"警告: 无法加载实体类型白名单: {e}")
    return None


def clean_and_filter_data(
    input_dir: Path,
    output_dir: Path,
    filter_criteria: Dict[str, Any],
    entity_whitelist_path: Optional[Path] = None,
    min_chains_per_file: int = 1,
    file_id_pattern: Optional[str] = None
):
    """
    清理和筛选数据
    
    Args:
        input_dir: 输入目录（gene_entity_output_v1）
        output_dir: 输出目录
        filter_criteria: 筛选标准
        entity_whitelist_path: 实体类型白名单路径（可选）
        min_chains_per_file: 每个文件最少保留的知识链数量
        file_id_pattern: 文件ID模式（如 "table_markdown_"）用于重命名
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filter_obj = DataFilter(filter_criteria)
    entity_whitelist = load_entity_whitelist(entity_whitelist_path)
    
    files = sorted(input_dir.glob("*.json"))
    print(f"总文件数: {len(files)}")
    
    global_stats = {
        'total_files': len(files),
        'processed_files': 0,
        'skipped_files': 0,
        'valid_files': 0,
        'rejected_files': 0,
        'total_chains': 0,
        'valid_chains': 0,
        'rejected_chains': 0,
        'rejection_reasons': defaultdict(int),
        'files_by_chain_count': defaultdict(int),
    }
    
    valid_files_data = []
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                file_data = json.load(fp)
            
            global_stats['processed_files'] += 1
            
            # 筛选知识链
            valid_chains, file_stats = filter_obj.filter_file(file_data)
            
            global_stats['total_chains'] += file_stats['total']
            global_stats['valid_chains'] += file_stats['valid']
            global_stats['rejected_chains'] += file_stats['rejected']
            
            for reason, count in file_stats['rejection_reasons'].items():
                global_stats['rejection_reasons'][reason] += count
            
            # 检查是否满足最小链数要求
            if len(valid_chains) < min_chains_per_file:
                global_stats['rejected_files'] += 1
                continue
            
            global_stats['valid_files'] += 1
            global_stats['files_by_chain_count'][len(valid_chains)] += 1
            
            # 更新文件数据
            file_data['knowledge_mining'] = valid_chains
            
            # 可选：应用实体类型白名单过滤
            if entity_whitelist:
                filtered_chains = []
                for km in valid_chains:
                    er = km.get('entity_recognition', {})
                    if isinstance(er, dict):
                        # 检查所有实体类型是否在白名单中
                        all_valid = True
                        for entity_type in er.keys():
                            if entity_type not in entity_whitelist:
                                all_valid = False
                                break
                        if all_valid:
                            filtered_chains.append(km)
                    else:
                        filtered_chains.append(km)
                
                if len(filtered_chains) >= min_chains_per_file:
                    file_data['knowledge_mining'] = filtered_chains
                    valid_files_data.append(file_data)
                else:
                    global_stats['rejected_files'] += 1
                    global_stats['valid_files'] -= 1
            else:
                valid_files_data.append(file_data)
                
        except Exception as e:
            global_stats['skipped_files'] += 1
            print(f"错误处理文件 {f.name}: {e}")
    
    # 保存清理后的文件
    print(f"\n保存清理后的数据...")
    for file_data in valid_files_data:
        file_id = file_data.get('file_id', 'unknown')
        
        # 可选：重命名文件ID
        if file_id_pattern:
            new_file_id = f"{file_id_pattern}{file_id}"
            file_data['file_id'] = new_file_id
        
        output_file = output_dir / f"{file_id}_cleaned.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats_file = output_dir / "cleaning_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(global_stats, f, ensure_ascii=False, indent=2)
    
    # 打印统计报告
    print("\n" + "="*80)
    print("数据清理统计报告")
    print("="*80)
    print(f"\n【文件统计】")
    print(f"总文件数: {global_stats['total_files']}")
    print(f"成功处理: {global_stats['processed_files']}")
    print(f"跳过文件: {global_stats['skipped_files']}")
    print(f"有效文件: {global_stats['valid_files']} ({global_stats['valid_files']/max(global_stats['processed_files'],1)*100:.1f}%)")
    print(f"剔除文件: {global_stats['rejected_files']} ({global_stats['rejected_files']/max(global_stats['processed_files'],1)*100:.1f}%)")
    
    print(f"\n【知识链统计】")
    print(f"总知识链数: {global_stats['total_chains']}")
    print(f"有效知识链: {global_stats['valid_chains']} ({global_stats['valid_chains']/max(global_stats['total_chains'],1)*100:.1f}%)")
    print(f"剔除知识链: {global_stats['rejected_chains']} ({global_stats['rejected_chains']/max(global_stats['total_chains'],1)*100:.1f}%)")
    
    if global_stats['rejection_reasons']:
        print(f"\n【剔除原因分布】")
        for reason, count in sorted(global_stats['rejection_reasons'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} ({count/max(global_stats['rejected_chains'],1)*100:.1f}%)")
    
    print(f"\n【文件知识链数量分布 (Top 10)】")
    for count, file_count in sorted(global_stats['files_by_chain_count'].items(), reverse=True)[:10]:
        print(f"  {count} 条链: {file_count} 个文件")
    
    print(f"\n【输出目录】")
    print(f"  {output_dir}")
    print(f"  清理后文件数: {len(valid_files_data)}")
    print(f"  统计文件: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="清理和筛选 step5 生成的数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
筛选标准示例:
  --strict: 严格标准 (condition+result+entity)
  --moderate: 中等标准 (condition+result, 允许无entity)
  --result-focused: 结果导向 (必须有result)
  --loose: 宽松标准 (只要有condition或result)

自定义标准:
  --require-condition: 必须包含condition
  --require-result: 必须包含result
  --require-entity: 必须包含entity
  --min-entity-count: 最小实体数量 (默认: 1)
        """
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='gene_entity_output_v1',
        help='输入目录 (默认: gene_entity_output_v1)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    
    # 预设标准
    parser.add_argument('--strict', action='store_true', help='严格标准: condition+result+entity')
    parser.add_argument('--moderate', action='store_true', help='中等标准: condition+result')
    parser.add_argument('--result-focused', action='store_true', help='结果导向: 必须有result')
    parser.add_argument('--loose', action='store_true', help='宽松标准: 只要有condition或result')
    
    # 自定义标准
    parser.add_argument('--require-condition', action='store_true', help='必须包含condition')
    parser.add_argument('--require-result', action='store_true', help='必须包含result')
    parser.add_argument('--require-entity', action='store_true', help='必须包含entity')
    parser.add_argument('--min-entity-count', type=int, default=1, help='最小实体数量')
    
    parser.add_argument(
        '--min-chains-per-file',
        type=int,
        default=1,
        help='每个文件最少保留的知识链数量 (默认: 1)'
    )
    
    parser.add_argument(
        '--entity-whitelist',
        type=str,
        help='实体类型白名单文件路径 (可选)'
    )
    
    parser.add_argument(
        '--file-id-pattern',
        type=str,
        help='文件ID模式，用于重命名 (如: "table_markdown_")'
    )
    
    args = parser.parse_args()
    
    # 确定筛选标准
    if args.strict:
        criteria = {
            'require_condition': True,
            'require_result': True,
            'require_entity': True,
            'require_result_format': True,
            'min_entity_count': 1,
        }
    elif args.moderate:
        criteria = {
            'require_condition': True,
            'require_result': True,
            'require_entity': False,
            'require_result_format': True,
            'min_entity_count': 0,
        }
    elif args.result_focused:
        criteria = {
            'require_condition': False,
            'require_result': True,
            'require_entity': False,
            'require_result_format': True,
            'min_entity_count': 0,
        }
    elif args.loose:
        criteria = {
            'require_condition': False,
            'require_result': False,
            'require_entity': False,
            'require_result_format': False,
            'min_entity_count': 0,
        }
    else:
        # 使用自定义标准
        criteria = {
            'require_condition': args.require_condition,
            'require_result': args.require_result,
            'require_entity': args.require_entity,
            'require_result_format': args.require_result if args.require_result else False,
            'min_entity_count': args.min_entity_count if args.require_entity else 0,
        }
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    entity_whitelist_path = Path(args.entity_whitelist) if args.entity_whitelist else None
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    print("="*80)
    print("数据清理和筛选")
    print("="*80)
    print(f"\n输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"\n筛选标准:")
    for key, value in criteria.items():
        print(f"  {key}: {value}")
    if entity_whitelist_path:
        print(f"实体类型白名单: {entity_whitelist_path}")
    print(f"每个文件最少链数: {args.min_chains_per_file}")
    print()
    
    clean_and_filter_data(
        input_dir=input_dir,
        output_dir=output_dir,
        filter_criteria=criteria,
        entity_whitelist_path=entity_whitelist_path,
        min_chains_per_file=args.min_chains_per_file,
        file_id_pattern=args.file_id_pattern
    )


if __name__ == "__main__":
    main()

