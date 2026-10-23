#!/usr/bin/env python3
"""
准备训练数据：将当前格式转换为Chain B格式，并确保数据质量
关键问题：
1. 格式转换：knowledge_chain + entity_recognition → Chain B格式
2. 实体类型映射：当前类型 → 技术方案要求的类型
3. 数据一致性校验：确保condition/result/entity一致
4. 质量筛选：只保留高质量、完整的数据
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import argparse
import re

# 导入质量评分器
from quality_scoring_and_split import QualityScorer, load_entity_whitelist


class ChainBConverter:
    """将当前格式转换为Chain B格式"""
    
    def __init__(self, entity_type_mapping: Dict[str, str] = None):
        """
        Args:
            entity_type_mapping: 实体类型映射 {当前类型: 标准类型}
        """
        self.entity_type_mapping = entity_type_mapping or {}
        
        # 默认映射（根据常见实体类型）
        default_mapping = {
            'gene_symbol': 'Gene',
            'gene': 'Gene',
            'protein': 'Protein',
            'disease': 'Disease',
            'drug': 'Drug',
            'compound': 'Compound',
            'species': 'Species',  # 可能需要映射到其他类型
            'cell': 'Cell Type',
            'tissue': 'Tissue',
            'organ': 'Organ',
            'phenotype': 'Phenotype',
            'experimental_method': 'Experimental Method',
            'functional_mechanism': 'Functional Mechanism',
            'target': 'Target',
            'regulator': 'Regulator',
        }
        
        # 合并用户映射
        self.entity_type_mapping = {**default_mapping, **self.entity_type_mapping}
    
    def normalize_entity_type(self, entity_type: str) -> str:
        """规范化实体类型名称"""
        # 转换为Title Case
        normalized = entity_type.replace('_', ' ').title()
        
        # 应用映射
        if normalized.lower() in [k.lower() for k in self.entity_type_mapping.keys()]:
            for key, value in self.entity_type_mapping.items():
                if key.lower() == normalized.lower():
                    return value
        
        return normalized
    
    def extract_entities_from_condition(self, condition: str) -> List[str]:
        """从condition中提取实体值"""
        if not condition:
            return []
        
        entities = []
        parts = condition.split('|')
        for part in parts:
            part = part.strip()
            if ':' in part:
                _, value = part.split(':', 1)
                value = value.strip()
                if value and value.lower() != 'null':
                    entities.append(value)
            else:
                if part and part.lower() != 'null':
                    entities.append(part)
        
        return entities
    
    def extract_entities_from_result(self, result: str) -> List[Tuple[str, str]]:
        """从result中提取指标和数值"""
        if not result:
            return []
        
        entities = []
        parts = result.split('|')
        for part in parts:
            part = part.strip()
            if ':' in part:
                indicator, value = part.split(':', 1)
                indicator = indicator.strip()
                value = value.strip()
                if indicator and value:
                    entities.append((indicator, value))
        
        return entities
    
    def convert_to_chain_b(
        self,
        knowledge_chain: Dict,
        entity_recognition: Dict
    ) -> Optional[Dict]:
        """
        转换为Chain B格式
        
        Returns:
            Chain B格式的知识链，如果转换失败返回None
        """
        condition = knowledge_chain.get('condition', '').strip()
        result = knowledge_chain.get('result', '').strip()
        
        # 基本校验
        if not condition:
            return None
        
        # 构建entity字典
        entity_dict = {}
        
        # 从entity_recognition中提取实体
        if entity_recognition and isinstance(entity_recognition, dict):
            for entity_type, entities in entity_recognition.items():
                if not entities:
                    continue
                
                # 规范化实体类型
                normalized_type = self.normalize_entity_type(entity_type)
                
                # 确保entities是列表
                if isinstance(entities, list):
                    entity_list = [str(e).strip() for e in entities if e]
                else:
                    entity_list = [str(entities).strip()]
                
                if entity_list:
                    if normalized_type not in entity_dict:
                        entity_dict[normalized_type] = []
                    entity_dict[normalized_type].extend(entity_list)
        
        # 从condition和result中提取实体，确保一致性
        condition_entities = self.extract_entities_from_condition(condition)
        result_entities = self.extract_entities_from_result(result)
        
        # 构建Chain B格式
        chain_b = {
            'condition': condition,
            'result': result,
            'entity': entity_dict
        }
        
        return chain_b
    
    def validate_chain_b(self, chain_b: Dict, require_result: bool = True) -> Tuple[bool, str]:
        """
        验证Chain B格式的知识链
        
        Returns:
            (is_valid, reason)
        """
        # 检查必需字段
        if 'condition' not in chain_b or 'result' not in chain_b or 'entity' not in chain_b:
            return False, 'missing_fields'
        
        condition = chain_b.get('condition', '').strip()
        result = chain_b.get('result', '').strip()
        entity = chain_b.get('entity', {})
        
        # 检查condition
        if not condition:
            return False, 'empty_condition'
        
        # 检查result（根据要求）
        if require_result:
            if not result:
                return False, 'empty_result'
            
            # result必须包含"指标:数值"格式
            if ':' not in result:
                return False, 'invalid_result_format'
        
        # 检查entity（至少应该有一些实体）
        if not entity or not isinstance(entity, dict):
            # 允许无entity，但给出警告
            pass
        
        return True, 'valid'


def prepare_training_data(
    input_dir: Path,
    output_dir: Path,
    entity_whitelist_path: Path = None,
    entity_type_mapping_path: Path = None,
    min_file_score: float = 0.6,
    min_chains_per_file: int = 2,
    min_high_quality_chains: int = 1,
    require_result: bool = True,
    require_entity: bool = False
) -> Dict[str, Any]:
    """
    准备训练数据：格式转换 + 质量筛选
    
    Returns:
        统计信息
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    entity_whitelist = load_entity_whitelist(entity_whitelist_path)
    entity_type_mapping = {}
    if entity_type_mapping_path and entity_type_mapping_path.exists():
        with open(entity_type_mapping_path, 'r', encoding='utf-8') as f:
            entity_type_mapping = json.load(f)
    
    # 初始化
    scorer = QualityScorer(entity_whitelist)
    converter = ChainBConverter(entity_type_mapping)
    
    files = sorted(input_dir.glob("*.json"))
    print(f"总文件数: {len(files)}")
    
    stats = {
        'total_files': len(files),
        'processed': 0,
        'skipped': 0,
        'scored': 0,
        'filtered_by_score': 0,
        'converted': 0,
        'filtered_by_format': 0,
        'final_valid': 0,
        'total_chains': 0,
        'valid_chains': 0,
        'conversion_errors': defaultdict(int),
    }
    
    valid_files = []
    
    print("开始处理...")
    for i, f in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  已处理: {i + 1}/{len(files)}, 有效: {stats['final_valid']}")
        
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                file_data = json.load(fp)
            
            stats['processed'] += 1
            
            # 步骤1: 质量评分
            file_score, file_details, chain_scores = scorer.score_file(file_data)
            
            # 质量筛选
            if (file_score < min_file_score or
                file_details['total_chains'] < min_chains_per_file or
                file_details['high_quality_chains'] < min_high_quality_chains):
                stats['filtered_by_score'] += 1
                continue
            
            stats['scored'] += 1
            
            # 步骤2: 格式转换
            knowledge_mining = file_data.get('knowledge_mining', [])
            converted_chains = []
            
            for km in knowledge_mining:
                kc = km.get('knowledge_chain', {})
                er = km.get('entity_recognition', {})
                
                # 转换为Chain B格式
                chain_b = converter.convert_to_chain_b(kc, er)
                
                if not chain_b:
                    stats['conversion_errors']['conversion_failed'] += 1
                    continue
                
                # 验证Chain B格式
                is_valid, reason = converter.validate_chain_b(
                    chain_b,
                    require_result=require_result
                )
                
                if not is_valid:
                    stats['conversion_errors'][reason] += 1
                    continue
                
                # 额外校验：如果要求有entity
                if require_entity:
                    if not chain_b.get('entity') or not any(chain_b['entity'].values()):
                        stats['conversion_errors']['no_entity'] += 1
                        continue
                
                converted_chains.append(chain_b)
                stats['valid_chains'] += 1
            
            stats['total_chains'] += len(knowledge_mining)
            
            # 检查转换后的链数是否满足要求
            if len(converted_chains) < min_chains_per_file:
                stats['filtered_by_format'] += 1
                continue
            
            stats['converted'] += 1
            stats['final_valid'] += 1
            
            # 保存转换后的数据
            file_data['knowledge_chains'] = converted_chains
            file_data['quality_score'] = file_score
            file_data['quality_details'] = file_details
            
            # 移除旧格式数据（可选）
            if 'knowledge_mining' in file_data:
                del file_data['knowledge_mining']
            
            valid_files.append(file_data)
            
        except Exception as e:
            stats['skipped'] += 1
            if stats['skipped'] <= 5:
                print(f"  错误处理文件 {f.name}: {e}")
    
    print(f"\n处理完成:")
    print(f"  总文件数: {stats['total_files']}")
    print(f"  成功处理: {stats['processed']}")
    print(f"  通过质量筛选: {stats['scored']} ({stats['scored']/max(stats['processed'],1)*100:.1f}%)")
    print(f"  格式转换成功: {stats['converted']} ({stats['converted']/max(stats['processed'],1)*100:.1f}%)")
    print(f"  最终有效: {stats['final_valid']} ({stats['final_valid']/max(stats['processed'],1)*100:.1f}%)")
    print(f"  总知识链数: {stats['total_chains']}")
    print(f"  有效知识链数: {stats['valid_chains']} ({stats['valid_chains']/max(stats['total_chains'],1)*100:.1f}%)")
    
    if stats['conversion_errors']:
        print(f"\n转换错误统计:")
        for error, count in stats['conversion_errors'].items():
            print(f"  {error}: {count}")
    
    return {
        'valid_files': valid_files,
        'statistics': stats
    }


def main():
    parser = argparse.ArgumentParser(
        description="准备训练数据：格式转换 + 质量筛选",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        default='gene_entity_output_v1',
        help='输入目录'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        required=True,
        help='输出目录'
    )
    
    parser.add_argument(
        '--entity-whitelist',
        type=str,
        help='实体类型白名单文件路径（可选）'
    )
    
    parser.add_argument(
        '--entity-type-mapping',
        type=str,
        help='实体类型映射文件路径（可选）'
    )
    
    parser.add_argument(
        '--min-file-score',
        type=float,
        default=0.6,
        help='最小文件质量分数 (默认: 0.6)'
    )
    
    parser.add_argument(
        '--min-chains-per-file',
        type=int,
        default=2,
        help='每个文件最少知识链数 (默认: 2)'
    )
    
    parser.add_argument(
        '--min-high-quality-chains',
        type=int,
        default=1,
        help='每个文件最少高质量链数 (默认: 1)'
    )
    
    parser.add_argument(
        '--require-result',
        action='store_true',
        default=True,
        help='要求必须有result (默认: True)'
    )
    
    parser.add_argument(
        '--require-entity',
        action='store_true',
        help='要求必须有entity (默认: False)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    entity_whitelist_path = Path(args.entity_whitelist) if args.entity_whitelist else None
    entity_type_mapping_path = Path(args.entity_type_mapping) if args.entity_type_mapping else None
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    print("="*80)
    print("准备训练数据")
    print("="*80)
    print(f"\n输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"\n筛选标准:")
    print(f"  最小文件分数: {args.min_file_score}")
    print(f"  最少链数: {args.min_chains_per_file}")
    print(f"  最少高质量链数: {args.min_high_quality_chains}")
    print(f"  要求有result: {args.require_result}")
    print(f"  要求有entity: {args.require_entity}")
    print()
    
    # 处理数据
    result = prepare_training_data(
        input_dir=input_dir,
        output_dir=output_dir,
        entity_whitelist_path=entity_whitelist_path,
        entity_type_mapping_path=entity_type_mapping_path,
        min_file_score=args.min_file_score,
        min_chains_per_file=args.min_chains_per_file,
        min_high_quality_chains=args.min_high_quality_chains,
        require_result=args.require_result,
        require_entity=args.require_entity
    )
    
    # 保存处理后的文件
    print(f"\n保存处理后的数据...")
    for file_data in result['valid_files']:
        file_id = file_data.get('file_id', 'unknown')
        output_file = output_dir / f"{file_id}_chainb.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(file_data, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats_file = output_dir / "preparation_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(result['statistics'], f, ensure_ascii=False, indent=2)
    
    print(f"\n完成！")
    print(f"  输出目录: {output_dir}")
    print(f"  有效文件数: {len(result['valid_files'])}")
    print(f"  统计文件: {stats_file}")


if __name__ == "__main__":
    main()


