#!/usr/bin/env python3
"""
后处理脚本：将知识链和实体识别结果合并为新的格式

输入格式：
{
  "knowledge_chains": [
    {"condition": "...", "result": "..."},
    ...
  ],
  "entity_recognitions": [
    {"entity": {...}, "knowledge_chain_index": 0},
    ...
  ]
}

输出格式：
{
  "knowledge_chains": {
    "knowledge_chain_0": {
      "condition": "...",
      "result": "...",
      "entity": {...}
    },
    ...
  }
}
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def postprocess_single_file(input_file: Path, output_file: Path) -> bool:
    """
    处理单个JSON文件，将知识链和实体识别结果合并
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        
    Returns:
        bool: 是否成功处理
    """
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取知识链和实体识别结果
        knowledge_chains = data.get('knowledge_chains', [])
        entity_recognitions = data.get('entity_recognitions', [])
        
        # 创建实体识别索引（按knowledge_chain_index）
        entity_index: Dict[int, Dict[str, Any]] = {}
        for entity_rec in entity_recognitions:
            chain_idx = entity_rec.get('knowledge_chain_index')
            if chain_idx is None:
                continue
            # 优先使用 entity 字段；如果不存在则保留原对象去掉 index
            entity_payload = entity_rec.get('entity', {})
            if not isinstance(entity_payload, dict):
                entity_payload = {}
            entity_index[chain_idx] = entity_payload
        
        # 构建新的 knowledge_mining 列表格式：
        # [
        #   { "knowledge_chain": {...}, "entity_recognition": {...} },
        #   ...
        # ]
        knowledge_mining: List[Dict[str, Any]] = []
        for idx, chain in enumerate(knowledge_chains):
            kc_obj = {
                "condition": chain.get('condition', ''),
                "result": chain.get('result', ''),
            }
            entity_obj = entity_index.get(idx, {})
            knowledge_mining.append(
                {
                    "knowledge_chain": kc_obj,
                    "entity_recognition": entity_obj,
            }
            )
        
        # 构建输出数据（保留其他字段）
        output_data: Dict[str, Any] = {}
        
        # 保留原有字段（除了knowledge_chains和entity_recognitions）
        for key, value in data.items():
            if key not in ['knowledge_chains', 'entity_recognitions']:
                output_data[key] = value
        
        # 添加新的 knowledge_mining 列表
        output_data['knowledge_mining'] = knowledge_mining
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 处理完成: {input_file.name} → {output_file.name} "
                   f"(共 {len(knowledge_mining)} 条知识链)")
        return True
        
    except Exception as e:
        logger.error(f"❌ 处理失败 {input_file.name}: {str(e)}")
        return False


def postprocess_directory(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.json"
) -> None:
    """
    批量处理目录中的所有JSON文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        pattern: 文件匹配模式
    """
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有输入文件
    input_files = sorted(input_dir.glob(pattern))
    
    if not input_files:
        logger.warning(f"未找到匹配的文件: {input_dir}/{pattern}")
        return
    
    logger.info(f"📂 输入目录: {input_dir}")
    logger.info(f"📂 输出目录: {output_dir}")
    logger.info(f"📄 找到 {len(input_files)} 个文件")
    logger.info("=" * 80)
    
    success_count = 0
    fail_count = 0
    
    for input_file in input_files:
        # 构建输出文件路径（保持相同文件名）
        output_file = output_dir / input_file.name
        
        if postprocess_single_file(input_file, output_file):
            success_count += 1
        else:
            fail_count += 1
    
    logger.info("=" * 80)
    logger.info(f"🎉 全部处理完成！")
    logger.info(f"✅ 成功: {success_count} 个文件")
    logger.info(f"❌ 失败: {fail_count} 个文件")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="后处理知识链和实体识别结果，合并为新的格式"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='gene_entity_output',
        help='输入目录（默认: gene_entity_output）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='gene_entity_output_postprocessed',
        help='输出目录（默认: gene_entity_output_postprocessed）'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help='处理单个文件（如果指定，则忽略input_dir）'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='单个文件的输出路径（需要与--input_file一起使用）'
    )
    
    args = parser.parse_args()
    
    if args.input_file:
        # 处理单个文件
        input_file = Path(args.input_file)
        output_file = Path(args.output_file) if args.output_file else input_file.parent / f"{input_file.stem}_postprocessed{input_file.suffix}"
        
        if not input_file.exists():
            logger.error(f"输入文件不存在: {input_file}")
            return
        
        postprocess_single_file(input_file, output_file)
    else:
        # 批量处理目录
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        
        if not input_dir.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            return
        
        postprocess_directory(input_dir, output_dir)


if __name__ == '__main__':
    main()

