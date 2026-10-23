#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四线程 API 知识链生成脚本

使用四个模型 API 并行生成知识链，每个模型处理全部样本：
1. gpt-5.1
2. gemini-3-flash-preview
3. claude-sonnet-4-6
4. DeepSeek-V3.2

用法：
python api_kc_generation.py --num_samples 2749 --workers 4
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ============== 线程安全的日志记录器 ==============
class ThreadSafeLogger:
    """线程安全的日志记录器，避免多线程写入冲突"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.lock = threading.Lock()
        self._init_log()
    
    def _init_log(self):
        """初始化日志文件"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("  API 知识链生成日志\n")
            f.write("=" * 80 + "\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Python PID: {os.getpid()}\n")
            f.write("\n")
    
    def log(self, message: str, level: str = "INFO"):
        """写入日志（线程安全）"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] [{level}] {message}\n"
        with self.lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line)
    
    def log_section(self, title: str):
        """写入分节标题"""
        with self.lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"  {title}\n")
                f.write("=" * 60 + "\n")
    
    def log_dict(self, data: Dict, title: str = ""):
        """写入字典数据"""
        with self.lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                if title:
                    f.write(f"\n--- {title} ---\n")
                json_str = json.dumps(data, ensure_ascii=False, indent=2)
                f.write(json_str + "\n")


# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============== API 配置 ==============
API_URL = os.environ.get("LLM_API_URL", "")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 四个模型
MODELS = {
    "gpt-5.1": "gpt-5.1",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "deepseek-v3.2": "DeepSeek-V3.2"
}

# ============== Step8 提示词（与训练一致） ==============
SYSTEM_PROMPT = (
    "你是一名基因组学知识抽取助手，任务是从表格内容(table_content)中生成知识挖掘结果。"
    "输出必须是 JSON，键名为 knowledge_mining，对应一个列表，每个元素包含："
    "knowledge_chain.condition、knowledge_chain.result、entity_recognition。"
    "仅输出 JSON，不要额外解释。"
)

INPUT_PROMPT = """任务：从给定 table_content 抽取知识链并完成实体识别，输出严格 JSON：
{
  "knowledge_mining": [
    {
      "knowledge_chain": {"condition": "...", "result": "..."},
      "entity_recognition": {"<entity_type>": ["实体1", "实体2"]}
    }
  ]
}

知识链规则（对齐 step5）：
1) 遍历表格每行：每个数值字段至少一条链；行无数值时 result 允许为空。
2) result 仅 1 个键值对，格式 "指标:数值"，不含多 KV。
3) condition 仅非数值/分组信息，使用 "key:value" 片段，片段间用 " | "，每片段最多 1 个冒号，禁止把 result 数值放入 condition。
4) 覆盖所有数值字段（含单位/百分号/科学计数），ID/编号优先放 condition。

实体识别规则（对齐 01_entity_type）：
1) 实体值必须在 condition 或 result 原文中**完整匹配出现**，不得拆分、不得补全。
2) 不拆分 items.text（逗号/分号等仍视为整体）；若含并列混杂则不标。
3) 仅允许 14 类实体：
   gene_symbol, regulator, target, functional_mechanism, species, disease,
   organ, tissue, cell, protein, compound, drug, experimental_method, phenotype
4) 同类去重，未识别则用空对象 {}，不得幻觉新增或虚构实体。

请直接输出合法 JSON，不要额外文本。"""


@dataclass
class GenerationResult:
    """单次生成结果"""
    sample_id: str
    model_name: str
    generated_kc: Dict = field(default_factory=lambda: {"knowledge_mining": []})
    raw_response: str = ""
    error: str = ""
    success: bool = False
    retry_count: int = 0
    api_latency: float = 0.0


# 全局日志记录器
logger: Optional[ThreadSafeLogger] = None

# 每个模型的文件锁字典（线程安全写入）
model_file_locks: Dict[str, threading.Lock] = {}


def get_model_lock(model_key: str) -> threading.Lock:
    """获取指定模型的文件锁"""
    if model_key not in model_file_locks:
        model_file_locks[model_key] = threading.Lock()
    return model_file_locks[model_key]


def load_existing_results(output_file: Path) -> List[Dict]:
    """加载已存在的结果文件，返回已处理的样本ID列表和已有数据"""
    if output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            processed_ids = {s.get("sample_id") for s in data.get("samples", [])}
            return processed_ids, data
        except Exception as e:
            logger.log(f"读取已存在文件 {output_file} 失败: {e}，将重新创建", "WARNING")
    return set(), None


def initialize_model_output_file(output_file: Path, model: str, model_key: str, total_samples: int) -> Dict:
    """初始化或读取模型输出文件"""
    processed_ids, existing_data = load_existing_results(output_file)
    
    if existing_data:
        logger.log(f"[{model_key}] 从已存在文件恢复: {output_file}", "INFO")
        logger.log(f"[{model_key}] 已处理样本数: {len(processed_ids)}", "INFO")
        return existing_data, processed_ids
    
    # 创建新文件
    initial_data = {
        "model": model_key,
        "model_full_name": model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total_samples": total_samples,
        "success_count": 0,
        "error_count": 0,
        "success_rate": 0.0,
        "total_latency": 0.0,
        "avg_latency": 0.0,
        "elapsed_time": 0.0,
        "retry_distribution": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
        "error_types": {},
        "samples": []
    }
    
    # 确保文件存在
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, ensure_ascii=False, indent=2)
    
    return initial_data, set()


def append_result_to_file(output_file: Path, model_key: str, result: Dict):
    """线程安全地追加单个结果到文件（增量写入）"""
    file_lock = get_model_lock(model_key)
    
    with file_lock:
        try:
            # 读取现有数据
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否已存在该样本
            sample_id = result.get("sample_id")
            existing_ids = {s.get("sample_id") for s in data.get("samples", [])}
            
            if sample_id in existing_ids:
                # 已存在则跳过
                return False
            
            # 追加新结果
            data["samples"].append(result)
            
            # 更新统计
            if result.get("success"):
                data["success_count"] += 1
                data["total_latency"] += result.get("api_latency", 0)
            else:
                error_type = result.get("error", "Unknown").split(":")[0]
                data["error_types"][error_type] = data["error_types"].get(error_type, 0) + 1
            
            # 更新重试分布
            retry_count = result.get("retry_count", 0)
            data["retry_distribution"][retry_count] = data["retry_distribution"].get(retry_count, 0) + 1
            
            # 更新成功率
            data["success_rate"] = data["success_count"] / len(data["samples"]) if data["samples"] else 0
            
            # 更新平均延迟
            if data["success_count"] > 0:
                data["avg_latency"] = data["total_latency"] / data["success_count"]
            
            # 写回文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.log(f"[{model_key}] 增量写入失败: {e}", "ERROR")
            return False


def call_api(model: str, messages: List[Dict], retry_count: int = 0) -> str:
    """调用 API，返回原始响应文本"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        latency = time.time() - start_time
        
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Invalid response format: {data}")
            
    except Exception as e:
        if retry_count < 3:
            time.sleep(2 ** retry_count)  # 指数退避
            return call_api(model, messages, retry_count + 1)
        else:
            raise e


def extract_json(text: str) -> Dict:
    """从响应中提取 JSON"""
    cleaned = (text or "").strip()
    
    if not cleaned:
        return {"knowledge_mining": []}
    
    # 移除换行符和多余空格（修复 gemini/deepseek 响应问题）
    cleaned_no_newline = cleaned.replace('\n', '').replace('\r', '')
    
    # 尝试提取代码块中的 JSON（优先）
    if "```" in cleaned:
        blocks = cleaned.split("```")
        for block in blocks:
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
        # 也尝试去除换行后解析
        cleaned = cleaned_no_newline
        if "```" in cleaned:
            blocks = cleaned.split("```")
            for block in blocks:
                candidate = block.strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    try:
                        return json.loads(candidate)
                    except Exception:
                        pass
    
    # 尝试从无换行版本中提取
    cleaned = cleaned_no_newline
    
    # 尝试直接解析（去掉所有空白）
    for i, c in enumerate(cleaned):
        if c == '{':
            try:
                depth = 0
                for j in range(i, len(cleaned)):
                    if cleaned[j] == '{':
                        depth += 1
                    elif cleaned[j] == '}':
                        depth -= 1
                        if depth == 0:
                            return json.loads(cleaned[i:j+1])
            except Exception:
                continue
        elif c == '[':
            try:
                depth = 0
                for j in range(i, len(cleaned)):
                    if cleaned[j] == '[':
                        depth += 1
                    elif cleaned[j] == ']':
                        depth -= 1
                        if depth == 0:
                            return json.loads(cleaned[i:j+1])
            except Exception:
                continue
    
    return {"knowledge_mining": []}


def generate_kc(model: str, sample: Dict, sample_idx: int, total: int) -> GenerationResult:
    """为单个样本生成知识链"""
    sample_id = sample.get("sample_id", str(sample_idx))
    table_content = sample.get("table_content", "")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INPUT_PROMPT}\n\n<table_content>\n{table_content}\n</table_content>"}
    ]
    
    retry_count = 0
    while retry_count <= 3:
        try:
            start_time = time.time()
            raw_response = call_api(model, messages)
            latency = time.time() - start_time
            generated_kc = extract_json(raw_response)
            
            return GenerationResult(
                sample_id=sample_id,
                model_name=model,
                generated_kc=generated_kc,
                raw_response=raw_response,
                success=True,
                retry_count=retry_count,
                api_latency=latency
            )
        except Exception as e:
            retry_count += 1
            if retry_count > 3:
                return GenerationResult(
                    sample_id=sample_id,
                    model_name=model,
                    error=str(e),
                    retry_count=retry_count
                )
            time.sleep(2 ** retry_count)
    
    return GenerationResult(
        sample_id=sample_id,
        model_name=model,
        error="Max retries exceeded",
        retry_count=3
    )


def process_samples_for_model(
    model: str,
    model_key: str,
    samples: List[Dict],
    output_dir: Path,
    total_samples: int = 0
) -> Dict:
    """处理一批样本（单个模型）- 每个模型处理全部样本，增量写入"""
    global logger
    
    # 每个模型使用固定的输出文件名（不含时间戳，便于增量）
    output_file = output_dir / f"api_kc_{model_key}.json"
    
    # 初始化/读取已有结果
    output_data, processed_ids = initialize_model_output_file(
        output_file, model, model_key, len(samples)
    )
    
    results = []
    success_count = output_data.get("success_count", 0)
    error_count = output_data.get("error_count", 0)
    total_latency = output_data.get("total_latency", 0.0)
    retry_distribution = output_data.get("retry_distribution", {0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    error_types = output_data.get("error_types", {})
    
    # 计算需要处理的样本
    samples_to_process = [
        (i, sample) for i, sample in enumerate(samples)
        if sample.get("sample_id", str(i)) not in processed_ids
    ]
    
    # 记录模型开始信息
    logger.log(f"[{model_key}] 开始处理 {len(samples)} 个样本", "INFO")
    logger.log(f"[{model_key}] 模型: {model}", "INFO")
    logger.log(f"[{model_key}] 输出文件: {output_file}", "INFO")
    logger.log(f"[{model_key}] 已处理: {len(processed_ids)} 个", "INFO")
    logger.log(f"[{model_key}] 需处理: {len(samples_to_process)} 个", "INFO")
    logger.log(f"[{model_key}] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    
    start_time = time.time()
    process_start_time = start_time
    
    # 过滤掉已处理的样本
    for i, sample in samples_to_process:
        result = generate_kc(model, sample, i, len(samples))
        
        result_dict = {
            "sample_id": result.sample_id,
            "generated_kc": result.generated_kc,
            "ground_truth": sample.get("ground_truth", []),
            "table_content": sample.get("table_content", ""),
            "raw_response": result.raw_response,
            "error": result.error,
            "success": result.success,
            "retry_count": result.retry_count,
            "api_latency": result.api_latency
        }
        
        # 增量写入到文件
        appended = append_result_to_file(output_file, model_key, result_dict)
        
        if appended:
            results.append(result_dict)
            
            if result.success:
                success_count += 1
                total_latency += result.api_latency
            else:
                error_count += 1
                error_type = result.error.split(":")[0] if result.error else "Unknown"
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            retry_distribution[result.retry_count] = retry_distribution.get(result.retry_count, 0) + 1
        
        # 每 20 个样本记录一次进度
        processed_idx = len(results) + len(processed_ids)
        if (len(results)) % 20 == 0 or len(results) == len(samples_to_process):
            elapsed = time.time() - start_time
            avg_time = elapsed / len(results) if results else 0
            eta = avg_time * (len(samples_to_process) - len(results))
            
            progress_msg = (
                f"[{model_key}] 进度: {processed_idx}/{len(samples)} "
                f"({100*processed_idx/len(samples):.1f}%) "
                f"| 本次新增成功: {success_count - output_data.get('success_count', 0)} "
                f"| 失败: {error_count - output_data.get('error_count', 0)} "
                f"| 本次耗时: {elapsed:.1f}s "
                f"| 预计剩余: {eta:.1f}s"
            )
            logger.log(progress_msg, "INFO")
            print(progress_msg)  # 同时输出到 stdout
    
    elapsed_time = time.time() - start_time
    total_elapsed = time.time() - process_start_time
    
    # 记录模型完成信息
    logger.log(f"[{model_key}] 完成!", "INFO")
    logger.log(f"[{model_key}] 本次处理: {len(results)} 个样本", "INFO")
    logger.log(f"[{model_key}] 累计成功: {success_count}/{len(samples)} ({100*success_count/len(samples):.2f}%)", "INFO")
    logger.log(f"[{model_key}] 累计失败: {error_count}", "INFO")
    logger.log(f"[{model_key}] 本次耗时: {elapsed_time:.2f}s", "INFO")
    logger.log(f"[{model_key}] 总耗时: {total_elapsed:.2f}s", "INFO")
    logger.log(f"[{model_key}] 平均延迟: {total_latency/success_count:.3f}s" if success_count else f"[{model_key}] 平均延迟: N/A", "INFO")
    
    if error_types:
        logger.log(f"[{model_key}] 错误类型分布: {error_types}", "INFO")
    
    logger.log(f"[{model_key}] 重试次数分布: {retry_distribution}", "INFO")
    
    print(f"\n✅ [{model_key}] 完成！本次处理: {len(results)} 个，累计成功: {success_count}, 失败: {error_count}, 耗时: {elapsed_time:.1f}s")
    
    # 更新最终输出数据
    output_data["success_count"] = success_count
    output_data["error_count"] = error_count
    output_data["success_rate"] = success_count / len(samples) if samples else 0
    output_data["total_latency"] = total_latency
    output_data["avg_latency"] = total_latency / success_count if success_count else 0
    output_data["elapsed_time"] = elapsed_time
    output_data["retry_distribution"] = retry_distribution
    output_data["error_types"] = error_types
    
    return output_data


def load_test_samples(num_samples: Optional[int] = None) -> List[Dict]:
    """加载测试样本（从 step6/Test/ 原始数据）"""
    global logger
    
    test_dir = PROJECT_ROOT / "step6" / "Test"
    
    samples = []
    test_files = sorted(test_dir.glob("*.json"))
    
    logger.log(f"从 {test_dir} 加载测试样本", "INFO")
    logger.log(f"找到 {len(test_files)} 个测试文件", "INFO")
    
    for f in test_files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                
            # 构造样本
            sample = {
                "sample_id": data.get("file_id", f.stem),
                "table_content": data.get("table_content", ""),
                "ground_truth": data.get("knowledge_mining", []),
                "table_title": data.get("table_title", ""),
                "description": data.get("description", ""),
                "rows": data.get("rows", 0),
            }
            samples.append(sample)
        except Exception as e:
            logger.log(f"加载 {f} 失败: {e}", "ERROR")
            print(f"⚠️ 加载 {f} 失败: {e}")
    
    if num_samples:
        samples = samples[:num_samples]
    
    logger.log(f"加载完成，共 {len(samples)} 个测试样本", "INFO")
    print(f"📊 从 step6/Test/ 加载了 {len(samples)} 个测试样本")
    return samples


def main():
    global logger
    
    parser = argparse.ArgumentParser(description="四线程 API 知识链生成")
    parser.add_argument("--num_samples", type=int, default=None, help="处理的样本数量")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数（等于模型数）")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "api_generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志目录
    log_dir = PROJECT_ROOT / "api_generation" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化日志记录器
    log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"api_kc_generation_{log_timestamp}.log"
    logger = ThreadSafeLogger(log_file)
    
    print("=" * 70)
    print("  四线程 API 知识链生成")
    print("  每个模型处理全部样本")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"日志文件: {log_file}")
    print(f"使用模型: {list(MODELS.keys())}")
    
    logger.log_section("任务配置")
    logger.log(f"输出目录: {output_dir}", "INFO")
    logger.log(f"日志文件: {log_file}", "INFO")
    logger.log(f"使用模型: {list(MODELS)}", "INFO")
    logger.log(f"并行线程数: {args.workers}", "INFO")
    logger.log(f"请求样本数: {args.num_samples or '全部'}", "INFO")
    
    # 加载样本
    logger.log_section("加载测试数据")
    samples = load_test_samples(args.num_samples)
    total_samples = len(samples)
    
    logger.log(f"每个模型将处理 {total_samples} 个样本", "INFO")
    print(f"\n📦 每个模型将处理 {total_samples} 个样本")
    
    # 并行处理 - 每个模型处理全部样本
    logger.log_section("开始并行处理")
    logger.log(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    
    model_start_times = {}
    model_end_times = {}
    
    def run_model(model_key: str, model_name: str):
        model_start_times[model_key] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result = process_samples_for_model(
            model_name,
            model_key,
            samples,  # 全部样本
            output_dir,
            total_samples=total_samples
        )
        model_end_times[model_key] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return result
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for model_key, model_name in MODELS.items():
            future = executor.submit(run_model, model_key, model_name)
            futures[model_key] = future
            logger.log(f"已提交 [{model_key}] 到线程池", "INFO")
        
        # 等待完成
        results = {}
        for model_key, future in futures.items():
            try:
                results[model_key] = future.result()
            except Exception as e:
                logger.log(f"[{model_key}] 处理失败: {e}", "ERROR")
                print(f"❌ [{model_key}] 处理失败: {e}")
    
    # 汇总统计
    logger.log_section("处理完成汇总")
    end_time = datetime.now()
    
    total_success = sum(r.get("success_count", 0) for r in results.values())
    total_error = sum(r.get("error_count", 0) for r in results.values())
    total_latency = sum(r.get("total_latency", 0) for r in results.values())
    
    logger.log(f"完成时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    logger.log(f"总成功: {total_success}", "INFO")
    logger.log(f"总失败: {total_error}", "INFO")
    logger.log(f"总 API 调用耗时: {total_latency:.2f}s", "INFO")
    
    # 各模型详细统计
    logger.log_section("各模型统计")
    for model_key, result in results.items():
        stats = {
            "model": model_key,
            "samples": result.get("total_samples", 0),
            "success": result.get("success_count", 0),
            "error": result.get("error_count", 0),
            "success_rate": f"{100*result.get('success_rate', 0):.2f}%",
            "avg_latency": f"{result.get('avg_latency', 0):.3f}s",
            "elapsed_time": f"{result.get('elapsed_time', 0):.2f}s",
            "retry_distribution": result.get("retry_distribution", {}),
            "start_time": model_start_times.get(model_key, "N/A"),
            "end_time": model_end_times.get(model_key, "N/A"),
        }
        logger.log_dict(stats, f"[{model_key}] 统计")
    
    print("\n" + "=" * 70)
    print("✅ 全部完成！")
    print("=" * 70)
    print(f"总成功: {total_success}")
    print(f"总失败: {total_error}")
    
    # 列出生成的文件
    logger.log_section("生成的文件")
    output_files = list(output_dir.glob("api_kc_*.json"))
    
    logger.log(f"找到 {len(output_files)} 个生成文件:", "INFO")
    for f in output_files:
        # 读取每个文件的基本信息
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            sample_count = len(data.get("samples", []))
            success_count = data.get("success_count", 0)
            logger.log(f"  {f.name} - 样本数: {sample_count}, 成功: {success_count}, 大小: {f.stat().st_size / 1024:.1f} KB", "INFO")
            print(f"   {f.name} (样本数: {sample_count}, 成功: {success_count})")
        except Exception as e:
            logger.log(f"  {f.name} - 读取失败: {e}", "WARNING")
            print(f"   {f.name}")
    
    print(f"\n📄 完整日志: {log_file}")


if __name__ == "__main__":
    main()
