#!/usr/bin/env python
import os
# -*- coding: utf-8 -*-
"""
任务2并发版本：使用5个进程并发重生成JSON
支持断点续传
"""

import json
import time
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional
import requests
from functools import partial

# ============================================================================
# 配置
# ============================================================================

# API配置
API_URL = os.environ.get("QWEN_API_URL", "")
API_MODEL = "qwen3-32b"

# 路径配置
STEP4_DIR = Path(__file__).parent.resolve()
BASE_DIR = STEP4_DIR.parent
INPUT_TABLES_DIR = BASE_DIR / "step3" / "step3" / "output" / "all_in"
SUCCESS_JSON_DIR = STEP4_DIR / "output" / "structured_json_tables"

# 输出路径
OUTPUT_DIR = STEP4_DIR / "post_processing_output"
TASK2_OUTPUT = OUTPUT_DIR / "task2_regenerated_jsons"
LOG_DIR = STEP4_DIR / "logs"
TASK2_LOG = LOG_DIR / "task2_regeneration_parallel.log"

# 提示词路径
PROMPTS_DIR = STEP4_DIR / "prompts"
PROMPT_FILE = PROMPTS_DIR / "task2_regeneration.md"

# 约束
MAX_TABLE_LENGTH = 2500
NUM_WORKERS = 5  # 并发数

# 创建目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TASK2_OUTPUT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 日志配置
# ============================================================================

def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # 文件处理器
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# 工具函数
# ============================================================================

def load_prompt() -> str:
    """加载提示词"""
    if PROMPT_FILE.exists():
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        # 默认提示词
        return """请将以下表格转换为JSON格式。

要求：
1. 输出标准JSON格式
2. 包含table_title和rows字段
3. 保持数据完整性

直接输出JSON，不要任何其他文字。"""

def get_table_content_from_json(table_id: int) -> Optional[str]:
    """从all_in的JSON文件中读取markdown_content"""
    json_path = INPUT_TABLES_DIR / f"{table_id}.json"
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('markdown_content')
    except Exception:
        return None

def extract_caption(table_content: str) -> Optional[str]:
    """提取表格标题"""
    lines = table_content.split('\n')
    for line in lines:
        if line.startswith('caption:'):
            caption = line.replace('caption:', '').strip()
            return caption if caption else None
    return None

def clean_json_output(raw_response: str) -> str:
    """清理API响应，提取纯JSON"""
    import re
    
    # 移除<think>标签
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
    
    # 移除markdown代码块标记
    cleaned = re.sub(r'^```json\s*', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^```\s*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'```$', '', cleaned)
    
    # 尝试提取JSON部分
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, cleaned)
    if matches:
        json_str = matches[0]
        
        # 修复常见的转义问题
        # 1. 先处理已经正确转义的 \\ 保护起来
        json_str = json_str.replace('\\\\', '\x00ESCAPED_BACKSLASH\x00')
        # 2. 处理未转义的 \n \t \r 等
        json_str = json_str.replace('\\n', '\x00NEWLINE\x00')
        json_str = json_str.replace('\\t', '\x00TAB\x00')
        json_str = json_str.replace('\\r', '\x00RETURN\x00')
        # 3. 替换剩余的单个 \ 为 \\
        json_str = json_str.replace('\\', '\\\\')
        # 4. 恢复之前保护的字符
        json_str = json_str.replace('\x00ESCAPED_BACKSLASH\x00', '\\\\')
        json_str = json_str.replace('\x00NEWLINE\x00', '\\n')
        json_str = json_str.replace('\x00TAB\x00', '\\t')
        json_str = json_str.replace('\x00RETURN\x00', '\\r')
        
        return json_str
    
    return cleaned.strip()

def call_qwen32b(prompt: str, temperature: float = 0.05, max_tokens: int = 4096, 
                 timeout: int = 120) -> Optional[str]:
    """调用Qwen32B API（无日志版本，避免多进程日志冲突）"""
    payload = {
        "model": API_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content
        
    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None

# ============================================================================
# 核心处理函数（单个worker）
# ============================================================================

def process_single_table(table_id: int, prompt_template: str) -> dict:
    """处理单个表格（worker函数）"""
    result = {
        'table_id': table_id,
        'status': 'failed',
        'error': None
    }
    
    # 检查是否已处理
    output_file = TASK2_OUTPUT / f"regenerated_{table_id}.json"
    if output_file.exists():
        result['status'] = 'skipped'
        return result
    
    # 读取表格内容
    table_content = get_table_content_from_json(table_id)
    if not table_content:
        result['error'] = '找不到表格'
        return result
    
    # 长度检查
    if len(table_content) > MAX_TABLE_LENGTH:
        result['status'] = 'too_long'
        result['error'] = f'表格过长: {len(table_content)} chars'
        return result
    
    # 提取caption
    caption = extract_caption(table_content)
    
    # 构建提示词
    prompt = prompt_template
    prompt += f"\n\n## 表格标题\n{caption or 'null'}\n\n## 表格内容\n{table_content}\n\n请转换为JSON:"
    
    # 调用API
    response = call_qwen32b(prompt, temperature=0.05, max_tokens=4096, timeout=120)
    
    if not response:
        result['error'] = 'API调用失败或超时'
        return result
    
    # 后处理
    cleaned_json = clean_json_output(response)
    
    # 解析验证
    try:
        json_data = json.loads(cleaned_json)
        
        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        result['status'] = 'success'
        
    except json.JSONDecodeError as e:
        result['error'] = f'JSON解析失败: {str(e)}'
        
        # 保存原始响应
        debug_file = TASK2_OUTPUT / f"regenerated_{table_id}_raw.txt"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(response)
    
    return result

# ============================================================================
# 主流程
# ============================================================================

def get_failed_samples() -> List[int]:
    """获取失败样本列表（支持断点续传）"""
    # 获取所有输入的ID
    all_input_ids = set()
    for f in INPUT_TABLES_DIR.glob("*.json"):
        if f.stem.isdigit():
            all_input_ids.add(int(f.stem))
    
    # 获取所有已成功处理的ID（从原始输出目录）
    all_output_ids = set()
    for f in SUCCESS_JSON_DIR.glob("structured_*.json"):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                table_id = data.get('table_id')
                if table_id:
                    all_output_ids.add(table_id)
        except Exception:
            continue
    
    # 获取任务2已重生成的ID（断点续传）
    for f in TASK2_OUTPUT.glob("regenerated_*.json"):
        try:
            # 从文件名提取ID
            table_id = int(f.stem.replace('regenerated_', ''))
            all_output_ids.add(table_id)
        except Exception:
            continue
    
    # 计算未处理的ID
    failed_ids = all_input_ids - all_output_ids
    
    # 过滤超长表格
    filtered_failed_ids = []
    for table_id in failed_ids:
        table_content = get_table_content_from_json(table_id)
        if not table_content:
            continue
        if len(table_content) > MAX_TABLE_LENGTH:
            continue
        filtered_failed_ids.append(table_id)
    
    return sorted(filtered_failed_ids)

def main():
    """主函数"""
    logger = setup_logger('task2_parallel', TASK2_LOG)
    
    logger.info("=" * 80)
    logger.info(f"开始任务2：JSON重生成（{NUM_WORKERS}进程并发）")
    logger.info("=" * 80)
    
    # 加载提示词
    prompt_template = load_prompt()
    
    # 获取失败样本
    failed_samples = get_failed_samples()
    
    logger.info(f"找到 {len(failed_samples)} 个需要重生成的样本")
    
    if not failed_samples:
        logger.info("没有需要处理的样本，退出")
        return
    
    # 统计
    stats = {
        'total': len(failed_samples),
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'too_long': 0
    }
    
    start_time = time.time()
    
    # 创建进程池
    logger.info(f"启动 {NUM_WORKERS} 个并发进程...")
    
    with mp.Pool(processes=NUM_WORKERS) as pool:
        # 使用partial固定prompt_template参数
        process_func = partial(process_single_table, prompt_template=prompt_template)
        
        # 并发处理
        results = []
        for i, result in enumerate(pool.imap_unordered(process_func, failed_samples), 1):
            results.append(result)
            
            # 更新统计
            if result['status'] == 'success':
                stats['success'] += 1
                logger.info(f"✓ 重生成成功: ID={result['table_id']}")
            elif result['status'] == 'skipped':
                stats['skipped'] += 1
            elif result['status'] == 'too_long':
                stats['too_long'] += 1
            else:
                stats['failed'] += 1
                logger.error(f"✗ 重生成失败: ID={result['table_id']} - {result['error']}")
            
            # 进度报告
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (stats['total'] - i) / rate if rate > 0 else 0
                logger.info(f"进度: {i}/{stats['total']} | "
                           f"成功: {stats['success']} | "
                           f"失败: {stats['failed']} | "
                           f"速度: {rate:.1f}个/秒 | "
                           f"预计剩余: {eta/3600:.1f}小时")
    
    # 打印汇总
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("任务2 重生成汇总（并发版）")
    logger.info("=" * 80)
    logger.info(f"总样本数: {stats['total']}")
    logger.info(f"成功: {stats['success']}")
    logger.info(f"失败: {stats['failed']}")
    logger.info(f"跳过(已处理): {stats['skipped']}")
    logger.info(f"跳过(太长): {stats['too_long']}")
    logger.info(f"总耗时: {elapsed/3600:.2f} 小时")
    logger.info(f"平均速度: {stats['total']/elapsed:.2f} 个/秒")
    if stats['success'] + stats['failed'] > 0:
        success_rate = stats['success'] / (stats['success'] + stats['failed']) * 100
        logger.info(f"成功率: {success_rate:.1f}%")
    logger.info("=" * 80 + "\n")

if __name__ == '__main__':
    # 设置multiprocessing启动方法（Windows需要）
    mp.set_start_method('spawn', force=True)
    main()
