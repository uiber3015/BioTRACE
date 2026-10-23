#!/usr/bin/env python
import os
# -*- coding: utf-8 -*-
"""
判断JSON表格是否适合生成知识链
使用qwq-32b模型进行语义判断
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional
import requests

# ============================================================================
# 配置
# ============================================================================

# API配置
API_URL = os.environ.get("QWEN_API_URL", "")
API_MODEL = "qwq-32b"  # 使用推理模型

# 路径配置
STEP4_DIR = Path(__file__).parent.resolve()
BASE_DIR = STEP4_DIR.parent

# 输入路径（成功的JSON + 任务2重生成的JSON）
SUCCESS_JSON_DIR = STEP4_DIR / "output" / "structured_json_tables"
TASK2_OUTPUT = STEP4_DIR / "post_processing_output" / "task2_regenerated_jsons"

# 输出路径
OUTPUT_DIR = STEP4_DIR / "post_processing_output"
JUDGMENT_OUTPUT = OUTPUT_DIR / "kc_potential_judgments.json"
LOG_FILE = STEP4_DIR / "logs" / "kc_potential_judgment.log"

# 创建目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STEP4_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)

# ============================================================================
# 日志配置
# ============================================================================

def setup_logger():
    """设置日志"""
    logger = logging.getLogger('kc_judgment')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    ch = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

# ============================================================================
# 提示词
# ============================================================================

JUDGMENT_PROMPT = """你是知识链生成专家。请判断以下表格JSON是否适合提取知识链。

**知识链格式要求**：
```json
{
  "condition": "条件描述（如：基因名称、疾病类型、处理方式等）",
  "result": "结果指标：数值（如：表达量：1.23、P值：0.001）",
  "entity": {
    "实体类型": "实体值"
  }
}
```

**判断标准**（必须同时满足）：

1. **有明确的条件变量**（condition）
   - 表格中有标识性字段（如基因名、蛋白质、疾病、药物、细胞系、样本等）
   - 可以作为知识链的condition

2. **有可量化的结果**（result）
   - 表格中有数值型数据列（如表达量、浓度、比率、p-value等）
   - 可以提取为"指标:数值"格式
   - 至少有1个非null的数值

3. **JSON结构完整**
   - 有table_title或caption（可为null）
   - 有rows数组
   - rows至少包含2行数据
   - 每行数据字段一致

**不适合的情况**：
- ❌ 纯描述性表格（无实验数据）
- ❌ 只有文本无数值
- ❌ 数据行少于2行
- ❌ 无法识别条件和结果的对应关系

---

## 待判断的表格JSON

```json
{json_content}
```

---

## 请输出判断结果

**严格按以下JSON格式输出，不要任何其他文字**：

```json
{
  "suitable": true/false,
  "confidence": 0.0-1.0,
  "reason": "判断理由（1句话）",
  "potential_conditions": ["可能的condition字段1", "字段2"],
  "potential_results": ["可能的result字段1", "字段2"],
  "estimated_chains": 整数
}
```

**字段说明**：
- suitable: 是否适合生成知识链
- confidence: 置信度（0-1）
- reason: 简短理由
- potential_conditions: 可作为condition的字段名
- potential_results: 可作为result的字段名
- estimated_chains: 预估可生成的知识链数量

只输出JSON，不要markdown标记或其他文字。
"""

# ============================================================================
# API调用
# ============================================================================

def call_qwq32b(prompt: str, timeout: int = 60) -> Optional[str]:
    """调用qwq-32b API"""
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,  # 低温度，更确定性
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content
        
    except requests.exceptions.Timeout:
        logger.error(f"API调用超时（{timeout}秒）")
        return None
    except Exception as e:
        logger.error(f"API调用失败: {e}")
        return None

def clean_json_output(raw_response: str) -> str:
    """清理API响应"""
    import re
    
    # 移除markdown标记
    cleaned = re.sub(r'```json\s*', '', raw_response)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # 提取JSON
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, cleaned)
    if matches:
        return matches[0]
    
    return cleaned.strip()

# ============================================================================
# 核心判断逻辑
# ============================================================================

def judge_single_table(json_data: dict, table_id: int) -> Optional[dict]:
    """判断单个表格"""
    
    # 构建提示词
    json_content = json.dumps(json_data, ensure_ascii=False, indent=2)
    prompt = JUDGMENT_PROMPT.replace('{json_content}', json_content)
    
    # 调用API
    response = call_qwq32b(prompt, timeout=60)
    
    if not response:
        return None
    
    # 解析结果
    try:
        cleaned = clean_json_output(response)
        result = json.loads(cleaned)
        
        # 添加table_id
        result['table_id'] = table_id
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"判断结果解析失败 ID={table_id}: {e}")
        logger.debug(f"原始响应: {response[:200]}")
        return None

# ============================================================================
# 主流程
# ============================================================================

def collect_all_jsons():
    """收集所有JSON文件"""
    all_jsons = {}
    
    # 1. 原始成功的JSON
    logger.info("收集原始成功的JSON...")
    for f in SUCCESS_JSON_DIR.glob("structured_*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                table_id = data.get('table_id')
                if table_id:
                    all_jsons[table_id] = {
                        'source': 'original',
                        'data': data.get('result', {}).get('json_data', {})
                    }
        except Exception as e:
            logger.error(f"读取失败: {f.name} - {e}")
    
    logger.info(f"原始成功JSON: {len(all_jsons)}")
    
    # 2. 任务2重生成的JSON
    if TASK2_OUTPUT.exists():
        logger.info("收集任务2重生成的JSON...")
        task2_count = 0
        for f in TASK2_OUTPUT.glob("regenerated_*.json"):
            try:
                table_id = int(f.stem.replace('regenerated_', ''))
                
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                
                all_jsons[table_id] = {
                    'source': 'task2_regenerated',
                    'data': data
                }
                task2_count += 1
                
            except Exception as e:
                logger.error(f"读取失败: {f.name} - {e}")
        
        logger.info(f"任务2重生成JSON: {task2_count}")
    
    logger.info(f"总计JSON数: {len(all_jsons)}")
    return all_jsons

def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始判断知识链生成潜力")
    logger.info("=" * 80)
    
    # 收集所有JSON
    all_jsons = collect_all_jsons()
    
    if not all_jsons:
        logger.warning("没有找到JSON文件")
        return
    
    # 加载已有判断结果（支持断点续传）
    existing_judgments = set()
    if JUDGMENT_OUTPUT.exists():
        try:
            with open(JUDGMENT_OUTPUT, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                existing_judgments = {j['table_id'] for j in existing}
                logger.info(f"已有判断结果: {len(existing_judgments)}")
        except:
            pass
    
    # 统计
    stats = {
        'total': len(all_jsons),
        'suitable': 0,
        'unsuitable': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # 判断结果列表
    all_judgments = []
    if JUDGMENT_OUTPUT.exists():
        with open(JUDGMENT_OUTPUT, 'r', encoding='utf-8') as f:
            all_judgments = json.load(f)
    
    # 逐个判断
    for idx, (table_id, info) in enumerate(sorted(all_jsons.items()), 1):
        
        # 断点续传
        if table_id in existing_judgments:
            stats['skipped'] += 1
            continue
        
        logger.info(f"[{idx}/{stats['total']}] 判断 ID={table_id} (来源: {info['source']})")
        
        # 判断
        judgment = judge_single_table(info['data'], table_id)
        
        if judgment:
            all_judgments.append(judgment)
            
            if judgment.get('suitable', False):
                stats['suitable'] += 1
                logger.info(f"  ✓ 适合 (置信度: {judgment.get('confidence', 0):.2f}, "
                           f"预估: {judgment.get('estimated_chains', 0)}条)")
            else:
                stats['unsuitable'] += 1
                logger.info(f"  ✗ 不适合 - {judgment.get('reason', 'N/A')}")
            
            # 实时保存（每10个保存一次）
            if idx % 10 == 0:
                with open(JUDGMENT_OUTPUT, 'w', encoding='utf-8') as f:
                    json.dump(all_judgments, f, ensure_ascii=False, indent=2)
        else:
            stats['failed'] += 1
        
        # 进度报告
        if idx % 100 == 0:
            logger.info(f"进度: {idx}/{stats['total']} | "
                       f"适合: {stats['suitable']} | "
                       f"不适合: {stats['unsuitable']} | "
                       f"失败: {stats['failed']}")
        
        # 避免API过载
        time.sleep(0.5)
    
    # 最终保存
    with open(JUDGMENT_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_judgments, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    logger.info("\n" + "=" * 80)
    logger.info("判断汇总")
    logger.info("=" * 80)
    logger.info(f"总表格数: {stats['total']}")
    logger.info(f"适合生成知识链: {stats['suitable']}")
    logger.info(f"不适合: {stats['unsuitable']}")
    logger.info(f"判断失败: {stats['failed']}")
    logger.info(f"已跳过: {stats['skipped']}")
    
    suitable_rate = stats['suitable'] / (stats['suitable'] + stats['unsuitable']) * 100 \
                    if (stats['suitable'] + stats['unsuitable']) > 0 else 0
    logger.info(f"适合率: {suitable_rate:.1f}%")
    
    # 统计预估知识链总数
    total_chains = sum(j.get('estimated_chains', 0) for j in all_judgments if j.get('suitable'))
    logger.info(f"预估知识链总数: {total_chains}")
    
    logger.info(f"\n结果已保存到: {JUDGMENT_OUTPUT}")
    logger.info("=" * 80 + "\n")

if __name__ == '__main__':
    main()
