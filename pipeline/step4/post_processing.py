#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4 后处理脚本：三任务顺序执行
- 任务1：JSON符合性验证（标记问题，不修正）
- 任务2：JSON重生成（失败样本重新生成+后处理）
- 任务3：领域相关性筛选

作者：自动生成
日期：2025-11-27
"""

import json
import os
import re
import time
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

# ============================================================================
# 配置区域
# ============================================================================

# API配置（根据qwen3-32b.md）
API_URL = os.environ.get("QWEN_API_URL", "")
API_MODEL = "qwen3-32b"  # 正确的模型名称

# 路径配置（自动适配Linux/Windows）
STEP4_DIR = Path(__file__).parent.resolve()
BASE_DIR = STEP4_DIR.parent
PROMPTS_DIR = STEP4_DIR / "prompts"

# 数据路径（修正：实际路径是step3/step3/output/all_in）
INPUT_TABLES_DIR = BASE_DIR / "step3" / "step3" / "output" / "all_in"
SUCCESS_JSON_DIR = STEP4_DIR / "output" / "structured_json_tables"
ORIGINAL_LOG = STEP4_DIR / "conver_json.log"

# 输出路径
OUTPUT_DIR = STEP4_DIR / "post_processing_output"
TASK1_OUTPUT = OUTPUT_DIR / "task1_validation_results.json"
TASK2_OUTPUT = OUTPUT_DIR / "task2_regenerated_jsons"
TASK3_OUTPUT = OUTPUT_DIR / "task3_domain_filter_results.json"

# 日志路径
LOG_DIR = STEP4_DIR / "logs"
TASK1_LOG = LOG_DIR / "task1_validation.log"
TASK2_LOG = LOG_DIR / "task2_regeneration.log"
TASK3_LOG = LOG_DIR / "task3_domain_filter.log"

# 约束
MAX_TABLE_LENGTH = 2500

# 创建目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TASK2_OUTPUT.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 日志配置
# ============================================================================

def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """设置独立的日志记录器"""
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
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# API调用
# ============================================================================

def call_qwen32b(prompt: str, temperature: float = 0.1, max_tokens: int = 4096, logger=None) -> Optional[str]:
    """调用Qwen-32B API"""
    try:
        payload = {
            "model": API_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
            # 不设置stop参数，让模型自然输出，在后处理中清理<think>标签
        }
        
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content'].strip()
        else:
            if logger:
                logger.error(f"API返回错误状态码: {response.status_code}, 响应: {response.text[:200]}")
            return None
            
    except requests.exceptions.Timeout:
        if logger:
            logger.error(f"API调用超时（>120s）")
        return None
    except requests.exceptions.RequestException as e:
        if logger:
            logger.error(f"API请求异常: {str(e)[:200]}")
        return None
    except Exception as e:
        if logger:
            logger.error(f"API调用未知错误: {type(e).__name__}: {str(e)[:200]}")
        return None

# ============================================================================
# 提示词加载
# ============================================================================

def load_prompt(task_name: str) -> str:
    """加载提示词模板"""
    prompt_file = PROMPTS_DIR / f"{task_name}_prompt.md"
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取```之间的内容
    match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
    if match:
        return match.group(1)
    else:
        # 如果没有```标记，尝试直接查找提示词部分
        lines = content.split('\n')
        in_prompt = False
        prompt_lines = []
        for line in lines:
            if line.strip() == '```' and not in_prompt:
                in_prompt = True
                continue
            elif line.strip() == '```' and in_prompt:
                break
            elif in_prompt:
                prompt_lines.append(line)
        
        if prompt_lines:
            return '\n'.join(prompt_lines)
        else:
            raise ValueError(f"无法从 {prompt_file} 提取提示词")

# ============================================================================
# 辅助函数
# ============================================================================

def clean_json_output(raw_response: str) -> str:
    """清理模型输出，提取纯JSON"""
    # 移除<think>标签
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
    
    # 移除markdown标记
    cleaned = re.sub(r'```json\s*', '', cleaned)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # 提取JSON
    json_match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        
        # 修复常见问题
        # 1. 移除尾随逗号
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # 2. 修复重复键（保留最后一个）
        try:
            # 尝试解析，如果成功就返回
            json.loads(json_str)
            return json_str
        except json.JSONDecodeError:
            # 如果失败，尝试修复
            pass
        
        return json_str
    
    return raw_response

def extract_failed_samples() -> List[Dict]:
    """从日志中提取失败样本"""
    failed_samples = []
    
    if not ORIGINAL_LOG.exists():
        return failed_samples
    
    with open(ORIGINAL_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            if 'ERROR' in line or 'Failed' in line:
                # 尝试提取文件名
                match = re.search(r'(\d+_\d+_table_\d+\.md)', line)
                if match:
                    filename = match.group(1)
                    failed_samples.append({'filename': filename})
    
    return failed_samples

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

def get_table_metadata(table_id: int) -> Optional[Dict]:
    """获取表格的完整元数据"""
    json_path = INPUT_TABLES_DIR / f"{table_id}.json"
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def extract_caption(table_content: str) -> Optional[str]:
    """提取表格标题"""
    # 查找caption行
    lines = table_content.split('\n')
    for line in lines:
        if line.startswith('caption:'):
            caption = line.replace('caption:', '').strip()
            return caption if caption else None
    return None

def count_columns(table_content: str) -> int:
    """统计表格列数"""
    lines = table_content.split('\n')
    for line in lines:
        if '|' in line and not line.strip().startswith('caption:'):
            cols = [c.strip() for c in line.split('|') if c.strip()]
            return len(cols)
    return 0

# ============================================================================
# 任务1：JSON符合性验证
# ============================================================================

class Task1Validator:
    def __init__(self):
        self.logger = setup_logger('task1', TASK1_LOG)
        self.prompt_template = load_prompt('task1_validation')
        self.results = {
            'total': 0,
            'validated': 0,
            'issues_found': 0,
            'perfect': 0,
            'failed_validation': 0,
            'skipped_too_long': 0,
            'details': []
        }
    
    def validate_all(self):
        """验证所有成功的JSON"""
        self.logger.info("=" * 80)
        self.logger.info("开始任务1：JSON符合性验证（调用32B API）")
        self.logger.info("=" * 80)
        
        # 获取所有成功的JSON文件（structured_N.json格式）
        all_json_files = list(SUCCESS_JSON_DIR.glob("structured_*.json"))
        
        self.results['total'] = len(all_json_files)
        
        self.logger.info(f"找到 {len(all_json_files)} 个JSON文件")
        
        for idx, json_file in enumerate(all_json_files, 1):
            if idx % 50 == 0:
                self.logger.info(f"进度: {idx}/{len(all_json_files)}")
            
            try:
                self._validate_single(json_file)
            except Exception as e:
                self.logger.error(f"验证失败 {json_file.name}: {e}")
                self.results['failed_validation'] += 1
            
            # 避免API过载
            time.sleep(0.3)
        
        # 保存结果
        self._save_results()
        self._print_summary()
    
    def _validate_single(self, json_file: Path):
        """验证单个JSON（调用32B API）"""
        # 读取structured JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            structured_data = json.load(f)
        
        # 获取table_id
        table_id = structured_data.get('table_id')
        if not table_id:
            self.logger.warning(f"JSON文件缺少table_id: {json_file.name}")
            return
        
        # 从all_in获取原始表格
        table_content = get_table_content_from_json(table_id)
        
        if not table_content:
            self.logger.warning(f"找不到原始表格: ID={table_id}")
            return
        
        # 提取实际的JSON数据
        json_data = structured_data.get('result', {}).get('json_data', {})
        if not json_data:
            self.logger.warning(f"JSON文件缺少json_data: {json_file.name}")
            return
        
        # 长度检查
        if len(table_content) > MAX_TABLE_LENGTH:
            self.results['skipped_too_long'] += 1
            return
        
        # 基本统计
        expected_columns = count_columns(table_content)
        expected_rows = len([l for l in table_content.split('\n') if '|' in l]) - 2
        
        # 获取第一列信息
        first_column_name = ""
        first_column_fill_rate = 0.0
        if json_data.get('rows') and len(json_data['rows']) > 0:
            first_row = json_data['rows'][0]
            if isinstance(first_row, dict) and len(first_row) > 0:
                first_column_name = list(first_row.keys())[0]
                # 计算非null率
                non_null_count = sum(1 for row in json_data['rows'] 
                                    if isinstance(row, dict) and 
                                    first_column_name in row and 
                                    row[first_column_name] is not None)
                first_column_fill_rate = non_null_count / len(json_data['rows'])
        
        actual_columns = self._count_fields(json_data['rows'][0]) if json_data.get('rows') else 0
        actual_rows = len(json_data.get('rows', []))
        
        # 检查字段一致性
        inconsistent_count = 0
        if json_data.get('rows'):
            field_sets = [set(self._get_all_keys(row)) for row in json_data['rows']]
            unique_sets = len(set(map(frozenset, field_sets)))
            if unique_sets > 1:
                inconsistent_count = sum(1 for fs in field_sets if fs != field_sets[0])
        
        total_rows = len(json_data.get('rows', []))
        inconsistency_rate = (inconsistent_count / total_rows * 100) if total_rows > 0 else 0
        
        # 构建提示词（拼接方式，避免format()的大括号冲突）
        prompt = self.prompt_template
        prompt += f"\n\n## 原始表格\n{table_content}\n\n"
        prompt += f"## 生成的JSON\n{json.dumps(json_data, ensure_ascii=False, indent=2)}\n\n"
        prompt += f"## 统计信息\n"
        prompt += f"- 预期列数: {expected_columns}, 实际: {actual_columns}\n"
        prompt += f"- 预期行数: {expected_rows}, 实际: {actual_rows}\n"
        prompt += f"- 第一列: {first_column_name}, 填充率: {first_column_fill_rate:.2f}\n"
        prompt += f"- 不一致数: {inconsistent_count}/{total_rows} ({inconsistency_rate:.1f}%)\n\n"
        prompt += "请验证并输出结果:"
        
        # 调用32B API
        response = call_qwen32b(prompt, temperature=0.1, max_tokens=2048, logger=self.logger)
        
        if not response:
            self.logger.error(f"API调用失败（返回空）: {json_file.name}")
            self.results['failed_validation'] += 1
            return
        
        # 解析验证结果
        try:
            cleaned = clean_json_output(response)
            validation_result = json.loads(cleaned)
            
            # 统计
            if validation_result.get('overall_valid', True):
                self.results['perfect'] += 1
                self.logger.debug(f"✓ 验证通过: {json_file.name}")
            else:
                self.results['issues_found'] += 1
                self.logger.info(f"✗ 发现问题: {json_file.name} - {validation_result.get('severity', 'unknown')}")
                
                # 记录详情
                self.results['details'].append({
                    'filename': json_file.name,
                    'validation_result': validation_result,
                    'severity': validation_result.get('severity', 'unknown'),
                    'action': validation_result.get('action', 'manual_review')
                })
            
            self.results['validated'] += 1
            
        except json.JSONDecodeError as e:
            self.logger.error(f"验证结果解析失败: {json_file.name} - {e}")
            self.results['failed_validation'] += 1
    
    def _count_fields(self, obj: Dict) -> int:
        """统计对象的字段数（递归展开嵌套）"""
        count = 0
        for key, value in obj.items():
            if isinstance(value, dict):
                count += self._count_fields(value)
            else:
                count += 1
        return count
    
    def _get_all_keys(self, obj: Dict) -> List[str]:
        """获取所有键名"""
        keys = []
        for key, value in obj.items():
            keys.append(key)
            if isinstance(value, dict):
                keys.extend(self._get_all_keys(value))
        return keys
    
    def _save_results(self):
        """保存验证结果"""
        with open(TASK1_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"验证结果已保存到: {TASK1_OUTPUT}")
    
    def _print_summary(self):
        """打印汇总信息"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("任务1 验证汇总")
        self.logger.info("=" * 80)
        self.logger.info(f"总文件数: {self.results['total']}")
        self.logger.info(f"完美JSON: {self.results['perfect']}")
        self.logger.info(f"发现问题: {self.results['issues_found']}")
        self.logger.info(f"验证失败: {self.results['failed_validation']}")
        self.logger.info(f"跳过(太长): {self.results['skipped_too_long']}")
        self.logger.info("=" * 80 + "\n")

# ============================================================================
# 任务2：JSON重生成
# ============================================================================

class Task2Regenerator:
    def __init__(self):
        self.logger = setup_logger('task2', TASK2_LOG)
        self.prompt_template = load_prompt('task2_regeneration')
        self.stats = {
            'total_failed': 0,
            'skipped_too_long': 0,
            'regenerated': 0,
            'success': 0,
            'failed': 0
        }
    
    def regenerate_all(self):
        """重新生成所有失败样本"""
        self.logger.info("=" * 80)
        self.logger.info("开始任务2：JSON重生成")
        self.logger.info("=" * 80)
        
        # 获取失败样本
        failed_samples = self._get_failed_samples()
        self.stats['total_failed'] = len(failed_samples)
        
        self.logger.info(f"找到 {len(failed_samples)} 个失败样本")
        
        for idx, sample in enumerate(failed_samples, 1):
            if idx % 50 == 0:
                self.logger.info(f"进度: {idx}/{len(failed_samples)}")
            
            try:
                self._regenerate_single(sample)
            except Exception as e:
                self.logger.error(f"重生成失败 {sample}: {e}")
                self.stats['failed'] += 1
            
            # 避免API过载
            time.sleep(0.5)
        
        self._print_summary()
    
    def _get_failed_samples(self) -> List[int]:
        """获取失败样本列表（返回table_id，并过滤超长表格）"""
        # 获取所有输入的ID
        all_input_ids = set()
        for f in INPUT_TABLES_DIR.glob("*.json"):
            if f.stem.isdigit():
                all_input_ids.add(int(f.stem))
        
        # 获取所有已成功处理的ID
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
        
        # 计算未处理的ID
        failed_ids = all_input_ids - all_output_ids
        
        self.logger.info(f"输入表格数: {len(all_input_ids)}")
        self.logger.info(f"已成功处理: {len(all_output_ids)}")
        self.logger.info(f"未处理/失败（初步）: {len(failed_ids)}")
        
        # 过滤：移除超长表格
        filtered_failed_ids = []
        too_long_count = 0
        
        for table_id in failed_ids:
            table_content = get_table_content_from_json(table_id)
            if not table_content:
                continue
            
            if len(table_content) > MAX_TABLE_LENGTH:
                too_long_count += 1
                continue
            
            filtered_failed_ids.append(table_id)
        
        self.logger.info(f"过滤超长表格（>{MAX_TABLE_LENGTH}字符）: {too_long_count}")
        self.logger.info(f"最终需要重新生成: {len(filtered_failed_ids)}")
        
        return sorted(filtered_failed_ids)
    
    def _regenerate_single(self, table_id: int):
        """重新生成单个JSON"""
        # 从JSON读取表格内容
        table_content = get_table_content_from_json(table_id)
        if not table_content:
            self.logger.warning(f"找不到表格: ID={table_id}")
            self.stats['failed'] += 1
            return
        
        # 长度检查（双重保险，理论上已经预过滤）
        if len(table_content) > MAX_TABLE_LENGTH:
            self.stats['skipped_too_long'] += 1
            self.logger.warning(f"⚠️ 未预期的超长表格: ID={table_id} ({len(table_content)} chars)")
            return
        
        # 提取caption
        caption = extract_caption(table_content)
        
        # 构建提示词（使用字符串替换，避免与JSON大括号冲突）
        prompt = self.prompt_template
        prompt += f"\n\n## 表格标题\n{caption or 'null'}\n\n## 表格内容\n{table_content}\n\n请转换为JSON:"
        
        # 调用API
        response = call_qwen32b(prompt, temperature=0.05, max_tokens=4096, logger=self.logger)
        
        if not response:
            self.logger.error(f"API调用失败（返回空）: ID={table_id}")
            self.stats['failed'] += 1
            return
        
        # 后处理
        cleaned_json = clean_json_output(response)
        
        # 解析验证
        try:
            json_data = json.loads(cleaned_json)
            
            # 保存（使用structured_N.json格式）
            output_file = TASK2_OUTPUT / f"regenerated_{table_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            self.stats['success'] += 1
            self.stats['regenerated'] += 1
            self.logger.info(f"✓ 重生成成功: ID={table_id}")
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: ID={table_id} - {e}")
            self.stats['failed'] += 1
            
            # 保存原始响应用于调试
            debug_file = TASK2_OUTPUT / f"regenerated_{table_id}_raw.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response)
    
    def _print_summary(self):
        """打印汇总信息"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("任务2 重生成汇总")
        self.logger.info("=" * 80)
        self.logger.info(f"失败样本总数: {self.stats['total_failed']}")
        self.logger.info(f"跳过(太长): {self.stats['skipped_too_long']}")
        self.logger.info(f"尝试重生成: {self.stats['regenerated']}")
        self.logger.info(f"重生成成功: {self.stats['success']}")
        self.logger.info(f"重生成失败: {self.stats['failed']}")
        if self.stats['regenerated'] > 0:
            success_rate = (self.stats['success'] / self.stats['regenerated']) * 100
            self.logger.info(f"成功率: {success_rate:.1f}%")
        self.logger.info("=" * 80 + "\n")

# ============================================================================
# 任务3：领域相关性筛选
# ============================================================================

class Task3DomainFilter:
    def __init__(self):
        self.logger = setup_logger('task3', TASK3_LOG)
        self.prompt_template = load_prompt('task3_domain_filter')
        self.results = {
            'total': 0,
            'relevant': 0,
            'irrelevant': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'failed': 0,
            'details': []
        }
    
    def filter_all(self):
        """筛选所有论文"""
        self.logger.info("=" * 80)
        self.logger.info("开始任务3：领域相关性筛选")
        self.logger.info("=" * 80)
        
        # 获取所有表格文件（从文件名提取论文ID）
        papers = self._collect_papers()
        self.results['total'] = len(papers)
        
        self.logger.info(f"找到 {len(papers)} 篇论文")
        
        for idx, paper in enumerate(papers, 1):
            if idx % 50 == 0:
                self.logger.info(f"进度: {idx}/{len(papers)}")
            
            try:
                self._filter_single(paper)
            except Exception as e:
                self.logger.error(f"筛选失败 {paper['paper_id']}: {e}")
                self.results['failed'] += 1
            
            # 避免API过载
            time.sleep(0.5)
        
        # 保存结果
        self._save_results()
        self._print_summary()
    
    def _collect_papers(self) -> List[Dict]:
        """收集所有论文信息（从all_in的JSON元数据）"""
        papers = {}
        
        # 从JSON文件中提取论文信息
        for json_file in INPUT_TABLES_DIR.glob("*.json"):
            if not json_file.stem.isdigit():
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 从file_path提取paper_id（格式: path/to/paper_id/...）
                file_path = data.get('file_path', '')
                # 简化：使用table_id的前缀作为paper标识
                # 实际可能需要更复杂的逻辑
                table_id = data.get('table_id', '')
                
                # 暂时每个表格作为独立的"论文"进行筛选
                paper_id = json_file.stem
                papers[paper_id] = {
                    'paper_id': paper_id,
                    'table_id': int(json_file.stem),
                    'metadata': data
                }
            except Exception as e:
                continue
        
        return list(papers.values())
    
    def _filter_single(self, paper: Dict):
        """筛选单篇论文"""
        paper_id = paper['paper_id']
        metadata = paper.get('metadata', {})
        
        # 从markdown_content提取caption作为标题
        markdown_content = metadata.get('markdown_content', '')
        
        if not markdown_content:
            self.logger.warning(f"无法读取表格内容: {paper_id}")
            return
        
        # 从markdown内容提取标题和摘要
        title = self._extract_title_from_markdown(markdown_content)
        abstract = self._extract_abstract_from_markdown(markdown_content)
        
        if not title and not abstract:
            self.logger.warning(f"无法提取标题和摘要: {paper_id}")
            return
        
        # 构建提示词（拼接方式）
        prompt = self.prompt_template
        prompt += f"\n\n## 标题\n{title or 'N/A'}\n\n## 摘要/上下文\n{abstract or 'N/A'}\n\n请判断相关性:"
        
        # 调用API
        response = call_qwen32b(prompt, temperature=0.1, max_tokens=1024, logger=self.logger)
        
        if not response:
            self.logger.error(f"API调用失败（返回空）: {paper_id}")
            self.results['failed'] += 1
            return
        
        # 解析结果
        try:
            # 清理并解析JSON
            cleaned = clean_json_output(response)
            result = json.loads(cleaned)
            
            # 统计
            if result.get('relevant', False):
                self.results['relevant'] += 1
                
                confidence = result.get('confidence', 0)
                if confidence >= 0.8:
                    self.results['high_confidence'] += 1
                elif confidence >= 0.5:
                    self.results['medium_confidence'] += 1
                else:
                    self.results['low_confidence'] += 1
            else:
                self.results['irrelevant'] += 1
            
            # 保存详情
            self.results['details'].append({
                'paper_id': paper_id,
                'title': title,
                'relevant': result.get('relevant', False),
                'subdomain': result.get('subdomain', 'none'),
                'confidence': result.get('confidence', 0),
                'reasoning': result.get('reasoning', {})
            })
            
            self.logger.info(
                f"✓ {paper_id}: relevant={result.get('relevant')}, "
                f"subdomain={result.get('subdomain')}, "
                f"confidence={result.get('confidence', 0):.2f}"
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON解析失败: {paper_id} - {e}")
            self.results['failed'] += 1
    
    def _extract_title_from_markdown(self, markdown_content: str) -> Optional[str]:
        """从markdown内容提取标题（从Caption字段）"""
        lines = markdown_content.split('\n')
        in_caption = False
        caption_lines = []
        
        for line in lines:
            if line.startswith('## Caption'):
                in_caption = True
                continue
            elif line.startswith('##') and in_caption:
                break
            elif in_caption and line.strip():
                caption_lines.append(line.strip())
        
        caption = ' '.join(caption_lines)
        return caption[:200] if caption else None
    
    def _extract_abstract_from_markdown(self, markdown_content: str) -> Optional[str]:
        """从markdown内容提取摘要（暂无，返回表格内容前100字符）"""
        # 暂时使用表格内容的一部分作为上下文
        lines = markdown_content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('## Table Content'):
                # 获取表格内容的前几行
                content_lines = lines[i+1:i+5]
                return ' '.join([l.strip() for l in content_lines if l.strip()])[:200]
        return None
    
    def _save_results(self):
        """保存筛选结果"""
        with open(TASK3_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"筛选结果已保存到: {TASK3_OUTPUT}")
    
    def _print_summary(self):
        """打印汇总信息"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("任务3 领域筛选汇总")
        self.logger.info("=" * 80)
        self.logger.info(f"论文总数: {self.results['total']}")
        self.logger.info(f"相关论文: {self.results['relevant']}")
        self.logger.info(f"  - 高置信度(≥0.8): {self.results['high_confidence']}")
        self.logger.info(f"  - 中等置信度(0.5-0.8): {self.results['medium_confidence']}")
        self.logger.info(f"  - 低置信度(<0.5): {self.results['low_confidence']}")
        self.logger.info(f"不相关论文: {self.results['irrelevant']}")
        self.logger.info(f"筛选失败: {self.results['failed']}")
        if self.results['total'] > 0:
            relevance_rate = (self.results['relevant'] / self.results['total']) * 100
            self.logger.info(f"相关率: {relevance_rate:.1f}%")
        self.logger.info("=" * 80 + "\n")

# ============================================================================
# 主程序
# ============================================================================

def run_task1():
    """运行任务1：JSON符合性验证"""
    try:
        task1 = Task1Validator()
        task1.validate_all()
    except Exception as e:
        print(f"任务1异常: {e}")
        raise

def run_task2():
    """运行任务2：JSON重生成"""
    try:
        task2 = Task2Regenerator()
        task2.regenerate_all()
    except Exception as e:
        print(f"任务2异常: {e}")
        raise

def run_task3():
    """运行任务3：领域相关性筛选"""
    try:
        task3 = Task3DomainFilter()
        task3.filter_all()
    except Exception as e:
        print(f"任务3异常: {e}")
        raise

def main():
    """主函数：并行执行三个任务"""
    start_time = time.time()
    
    print("=" * 80)
    print("Step 4 后处理任务开始（并行模式）")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n【并行执行3个任务】")
    print("  - 任务1：验证已有的structured JSON（13,544个）")
    print("  - 任务2：重新生成失败样本（~7,000个）")
    print("  - 任务3：领域筛选all_in表格（25,166个）")
    print("=" * 80)
    
    try:
        # 创建三个进程
        p1 = mp.Process(target=run_task1, name="Task1-Validator")
        p2 = mp.Process(target=run_task2, name="Task2-Regenerator")
        p3 = mp.Process(target=run_task3, name="Task3-DomainFilter")
        
        # 启动所有进程
        p1.start()
        p2.start()
        p3.start()
        
        print(f"\n✓ 任务1已启动 (PID: {p1.pid})")
        print(f"✓ 任务2已启动 (PID: {p2.pid})")
        print(f"✓ 任务3已启动 (PID: {p3.pid})")
        print("\n查看实时日志:")
        print(f"  tail -f {TASK1_LOG}")
        print(f"  tail -f {TASK2_LOG}")
        print(f"  tail -f {TASK3_LOG}")
        print("=" * 80)
        
        # 等待所有进程完成
        p1.join()
        print(f"\n✓ 任务1完成")
        
        p2.join()
        print(f"✓ 任务2完成")
        
        p3.join()
        print(f"✓ 任务3完成")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断，正在停止所有任务...")
        p1.terminate()
        p2.terminate()
        p3.terminate()
        p1.join()
        p2.join()
        p3.join()
        print("所有任务已停止")
    except Exception as e:
        print(f"\n\n❌ 执行出错: {e}")
        raise
    finally:
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        print("\n" + "=" * 80)
        print("所有任务完成")
        print("=" * 80)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {hours}小时 {minutes}分钟 {seconds}秒")
        print("=" * 80)
        print("\n输出文件:")
        print(f"  - 任务1验证结果: {TASK1_OUTPUT}")
        print(f"  - 任务2重生成JSON: {TASK2_OUTPUT}/")
        print(f"  - 任务3筛选结果: {TASK3_OUTPUT}")
        print("\n日志文件:")
        print(f"  - 任务1日志: {TASK1_LOG}")
        print(f"  - 任务2日志: {TASK2_LOG}")
        print(f"  - 任务3日志: {TASK3_LOG}")
        print("=" * 80)

if __name__ == "__main__":
    main()
