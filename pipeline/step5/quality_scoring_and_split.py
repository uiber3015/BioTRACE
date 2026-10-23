#!/usr/bin/env python3
"""
数据质量评分与数据集划分系统
1. 对每个文件进行质量评分
2. 筛选高质量数据
3. 随机划分为4个集合（2:2:2:1）
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import argparse


class QualityScorer:
    """数据质量评分器"""
    
    def __init__(self, entity_whitelist: Dict[str, Any] = None):
        """
        Args:
            entity_whitelist: 实体类型白名单（可选）
        """
        self.entity_whitelist = set()
        if entity_whitelist:
            if isinstance(entity_whitelist, dict):
                self.entity_whitelist = set(entity_whitelist.get('entity_types', []))
            elif isinstance(entity_whitelist, list):
                self.entity_whitelist = set(entity_whitelist)
    
    def count_entities(self, entity_recognition: Dict) -> Tuple[int, int]:
        """
        统计实体数量和有效实体类型数量
        
        Returns:
            (total_entities, valid_entity_types)
        """
        if not entity_recognition or not isinstance(entity_recognition, dict):
            return 0, 0
        
        total = 0
        valid_types = 0
        
        for entity_type, entities in entity_recognition.items():
            if entities:
                if isinstance(entities, list):
                    count = len(entities)
                else:
                    count = 1
                
                total += count
                
                # 检查实体类型是否在白名单内
                if not self.entity_whitelist or entity_type in self.entity_whitelist:
                    valid_types += 1
        
        return total, valid_types
    
    def score_chain(self, knowledge_chain: Dict, entity_recognition: Dict) -> Tuple[float, Dict]:
        """
        对单条知识链进行评分
        
        Returns:
            (score, details)
            score: 0-1之间的分数
            details: 评分详情
        """
        condition = knowledge_chain.get('condition', '').strip()
        result = knowledge_chain.get('result', '').strip()
        
        details = {
            'has_condition': bool(condition),
            'has_result': bool(result),
            'result_format_valid': ':' in result if result else False,
            'result_has_indicator': False,
            'condition_length': len(condition),
            'condition_parts': len([p for p in condition.split('|') if p.strip()]) if condition else 0,
            'entity_count': 0,
            'valid_entity_types': 0,
        }
        
        score = 0.0
        
        # 1. Condition 评分 (30%)
        if condition:
            score += 0.3
            # condition 片段数奖励（最多2个片段）
            parts_bonus = min(details['condition_parts'] / 2.0, 1.0) * 0.05
            score += parts_bonus
        else:
            details['has_condition'] = False
        
        # 2. Result 评分 (40%)
        if result:
            score += 0.3
            details['has_result'] = True
            
            # result 格式检查
            if ':' in result:
                score += 0.1
                details['result_format_valid'] = True
                
                # 检查是否有指标名（不是 null:数值）
                parts = result.split('|')
                for part in parts:
                    part = part.strip()
                    if ':' in part:
                        indicator = part.split(':', 1)[0].strip()
                        if indicator and indicator.lower() != 'null':
                            details['result_has_indicator'] = True
                            score += 0.05  # 有指标名奖励
                            break
            else:
                details['result_format_valid'] = False
        else:
            details['has_result'] = False
        
        # 3. Entity 评分 (30%)
        entity_count, valid_types = self.count_entities(entity_recognition)
        details['entity_count'] = entity_count
        details['valid_entity_types'] = valid_types
        
        if entity_count > 0:
            # 有实体基础分
            score += 0.2
            
            # 实体数量奖励（最多3个实体）
            entity_bonus = min(entity_count / 3.0, 1.0) * 0.05
            score += entity_bonus
            
            # 有效实体类型奖励
            if valid_types > 0:
                score += 0.05
        
        # 确保分数在 0-1 之间
        score = min(score, 1.0)
        
        return score, details
    
    def score_file(self, file_data: Dict) -> Tuple[float, Dict, List[Dict]]:
        """
        对整个文件进行评分
        
        Returns:
            (file_score, file_details, chain_scores)
            file_score: 文件总分（0-1）
            file_details: 文件级别统计
            chain_scores: 每条链的评分详情
        """
        knowledge_mining = file_data.get('knowledge_mining', [])
        
        if not knowledge_mining:
            return 0.0, {'total_chains': 0, 'valid_chains': 0}, []
        
        chain_scores = []
        total_score = 0.0
        
        for km in knowledge_mining:
            kc = km.get('knowledge_chain', {})
            er = km.get('entity_recognition', {})
            
            chain_score, chain_details = self.score_chain(kc, er)
            chain_scores.append({
                'score': chain_score,
                'details': chain_details
            })
            total_score += chain_score
        
        # 文件分数 = 平均链分数
        file_score = total_score / len(knowledge_mining) if knowledge_mining else 0.0
        
        # 文件级别统计
        valid_chains = sum(1 for cs in chain_scores if cs['score'] > 0.5)
        high_quality_chains = sum(1 for cs in chain_scores if cs['score'] >= 0.7)
        
        file_details = {
            'total_chains': len(knowledge_mining),
            'valid_chains': valid_chains,
            'high_quality_chains': high_quality_chains,
            'avg_chain_score': file_score,
            'min_chain_score': min([cs['score'] for cs in chain_scores]) if chain_scores else 0.0,
            'max_chain_score': max([cs['score'] for cs in chain_scores]) if chain_scores else 0.0,
        }
        
        return file_score, file_details, chain_scores


def load_entity_whitelist(whitelist_path: Path = None) -> Dict[str, Any]:
    """加载实体类型白名单"""
    if whitelist_path and whitelist_path.exists():
        try:
            with open(whitelist_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 无法加载实体类型白名单: {e}")
    return None


def score_all_files(
    input_dir: Path,
    output_dir: Path,
    entity_whitelist_path: Path = None,
    min_file_score: float = 0.0,
    min_chains_per_file: int = 1,
    min_high_quality_chains: int = 0
) -> Dict[str, Any]:
    """
    对所有文件进行质量评分和筛选
    
    Returns:
        统计信息
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    entity_whitelist = load_entity_whitelist(entity_whitelist_path)
    scorer = QualityScorer(entity_whitelist)
    
    files = sorted(input_dir.glob("*.json"))
    print(f"总文件数: {len(files)}")
    
    scored_files = []
    global_stats = {
        'total_files': len(files),
        'processed': 0,
        'skipped': 0,
        'scored': 0,
        'filtered': 0,
        'score_distribution': Counter(),
        'chain_score_distribution': Counter(),
    }
    
    print("开始评分...")
    for i, f in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  已处理: {i + 1}/{len(files)}")
        
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                file_data = json.load(fp)
            
            global_stats['processed'] += 1
            
            file_score, file_details, chain_scores = scorer.score_file(file_data)
            
            # 筛选条件
            if (file_score < min_file_score or
                file_details['total_chains'] < min_chains_per_file or
                file_details['high_quality_chains'] < min_high_quality_chains):
                global_stats['filtered'] += 1
                continue
            
            global_stats['scored'] += 1
            global_stats['score_distribution'][round(file_score, 1)] += 1
            
            for cs in chain_scores:
                global_stats['chain_score_distribution'][round(cs['score'], 1)] += 1
            
            # 保存评分结果
            file_data['quality_score'] = file_score
            file_data['quality_details'] = file_details
            file_data['chain_quality_scores'] = chain_scores
            
            scored_files.append({
                'file_id': file_data.get('file_id', f.stem),
                'file_path': str(f),
                'score': file_score,
                'details': file_details,
                'data': file_data
            })
            
        except Exception as e:
            global_stats['skipped'] += 1
            if global_stats['skipped'] <= 5:
                print(f"  错误处理文件 {f.name}: {e}")
    
    # 按分数排序
    scored_files.sort(key=lambda x: x['score'], reverse=True)
    
    # 保存评分结果
    scores_file = output_dir / "quality_scores.json"
    with open(scores_file, 'w', encoding='utf-8') as f:
        json.dump({
            'statistics': global_stats,
            'files': [
                {
                    'file_id': sf['file_id'],
                    'score': sf['score'],
                    'details': sf['details']
                }
                for sf in scored_files
            ]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n评分完成:")
    print(f"  总文件数: {global_stats['total_files']}")
    print(f"  成功处理: {global_stats['processed']}")
    print(f"  跳过文件: {global_stats['skipped']}")
    print(f"  通过筛选: {global_stats['scored']} ({global_stats['scored']/max(global_stats['processed'],1)*100:.1f}%)")
    print(f"  被过滤: {global_stats['filtered']} ({global_stats['filtered']/max(global_stats['processed'],1)*100:.1f}%)")
    
    return {
        'scored_files': scored_files,
        'statistics': global_stats
    }


def split_dataset(
    scored_files: List[Dict],
    output_dir: Path,
    ratios: List[int] = [2, 2, 2, 1],
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    将高质量数据划分为4个集合
    
    Args:
        scored_files: 已评分的文件列表
        output_dir: 输出目录
        ratios: 划分比例 [集合1, 集合2, 集合3, 集合4]
        seed: 随机种子
    
    Returns:
        划分结果 {集合名: [file_ids]}
    """
    random.seed(seed)
    
    # 只使用文件ID列表
    file_ids = [sf['file_id'] for sf in scored_files]
    random.shuffle(file_ids)
    
    total = sum(ratios)
    n = len(file_ids)
    
    split_points = []
    cumsum = 0
    for ratio in ratios[:-1]:
        cumsum += ratio
        split_points.append(int(n * cumsum / total))
    
    splits = {
        "train_sft1": file_ids[:split_points[0]],
        "train_cot": file_ids[split_points[0]:split_points[1]],
        "train_rl": file_ids[split_points[1]:split_points[2]],
        "test_eval": file_ids[split_points[2]:]
    }
    
    # 创建输出目录结构
    for split_name in splits.keys():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文件ID列表
        ids_file = split_dir / "table_ids.json"
        with open(ids_file, 'w', encoding='utf-8') as f:
            json.dump(splits[split_name], f, ensure_ascii=False, indent=2)
        
        # 保存完整数据
        markdown_dir = split_dir / "markdown_tables"
        markdown_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建文件ID到数据的映射
        file_data_map = {sf['file_id']: sf['data'] for sf in scored_files}
        
        for file_id in splits[split_name]:
            if file_id in file_data_map:
                data = file_data_map[file_id]
                
                # 保存完整数据
                data_file = markdown_dir / f"{file_id}.json"
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"  {split_name}: {len(splits[split_name])} 个文件")
    
    # 保存划分信息
    split_info = {
        'total_files': n,
        'ratios': ratios,
        'seed': seed,
        'splits': {
            name: {
                'count': len(ids),
                'percentage': len(ids) / n * 100 if n > 0 else 0
            }
            for name, ids in splits.items()
        }
    }
    
    split_info_file = output_dir / "dataset_split.json"
    with open(split_info_file, 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据集划分完成:")
    print(f"  总文件数: {n}")
    for name, info in split_info['splits'].items():
        print(f"  {name}: {info['count']} ({info['percentage']:.1f}%)")
    
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="数据质量评分与数据集划分",
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
        '--min-file-score',
        type=float,
        default=0.5,
        help='最小文件质量分数 (0-1, 默认: 0.5)'
    )
    
    parser.add_argument(
        '--min-chains-per-file',
        type=int,
        default=1,
        help='每个文件最少知识链数 (默认: 1)'
    )
    
    parser.add_argument(
        '--min-high-quality-chains',
        type=int,
        default=0,
        help='每个文件最少高质量链数（分数>=0.7, 默认: 0）'
    )
    
    parser.add_argument(
        '--split-ratios',
        type=int,
        nargs=4,
        default=[2, 2, 2, 1],
        help='数据集划分比例 [集合1 集合2 集合3 集合4] (默认: 2 2 2 1)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    parser.add_argument(
        '--score-only',
        action='store_true',
        help='只进行评分，不进行划分'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    entity_whitelist_path = Path(args.entity_whitelist) if args.entity_whitelist else None
    
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    print("="*80)
    print("数据质量评分与数据集划分")
    print("="*80)
    print(f"\n输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"\n筛选标准:")
    print(f"  最小文件分数: {args.min_file_score}")
    print(f"  最少知识链数: {args.min_chains_per_file}")
    print(f"  最少高质量链数: {args.min_high_quality_chains}")
    if entity_whitelist_path:
        print(f"  实体类型白名单: {entity_whitelist_path}")
    print()
    
    # 步骤1: 评分和筛选
    result = score_all_files(
        input_dir=input_dir,
        output_dir=output_dir,
        entity_whitelist_path=entity_whitelist_path,
        min_file_score=args.min_file_score,
        min_chains_per_file=args.min_chains_per_file,
        min_high_quality_chains=args.min_high_quality_chains
    )
    
    if args.score_only:
        print("\n只进行评分，跳过数据集划分")
        return
    
    # 步骤2: 数据集划分
    print("\n" + "="*80)
    print("数据集划分")
    print("="*80)
    
    split_dataset(
        scored_files=result['scored_files'],
        output_dir=output_dir,
        ratios=args.split_ratios,
        seed=args.seed
    )
    
    print("\n完成！")


if __name__ == "__main__":
    main()


