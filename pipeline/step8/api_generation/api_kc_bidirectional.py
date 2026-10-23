#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向并行 API 知识链生成脚本

功能：
1. 同时开启正向（从头）和反向（从尾）两个进程处理同一模型
2. 使用多线程加速
3. 当检测到样本重合时停止
4. 最后合并两个进程的输出并去重

用法：
# 处理 DeepSeek-V3.2（双向，2线程）
python api_kc_bidirectional.py --model deepseek-v3.2 --workers 2

# 处理 Gemini（双向，2线程）
python api_kc_bidirectional.py --model gemini-3-flash-preview --workers 2
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import shutil

# ============== 线程安全的日志记录器 ==============
class ThreadSafeLogger:
    """线程安全的日志记录器"""
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.lock = threading.Lock()
        self._init_log()
    
    def _init_log(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("  双向 API 知识链生成日志\n")
            f.write("=" * 80 + "\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Python PID: {os.getpid()}\n")
            f.write("\n")
    
    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] [{level}] {message}\n"
        with self.lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line)
    
    def log_section(self, title: str):
        with self.lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"  {title}\n")
                f.write("=" * 60 + "\n")


# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============== API 配置 ==============
API_URL = os.environ.get("LLM_API_URL", "")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MODEL_NAMES = {
    "gpt-5.1": "gpt-5.1",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "deepseek-v3.2": "DeepSeek-V3.2"
}

# ============== Step8 提示词 ==============
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


# 全局变量
logger: Optional[ThreadSafeLogger] = None
file_locks: Dict[str, threading.Lock] = {}


def get_lock(key: str) -> threading.Lock:
    if key not in file_locks:
        file_locks[key] = threading.Lock()
    return file_locks[key]


# ============== API 调用 ==============
def call_api(model: str, messages: List[Dict], retry_count: int = 0) -> str:
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
            raise ValueError(f"Invalid response: {data}")
            
    except Exception as e:
        if retry_count < 3:
            time.sleep(2 ** retry_count)
            return call_api(model, messages, retry_count + 1)
        else:
            raise e


def extract_json(text: str) -> Dict:
    """从响应中提取 JSON"""
    cleaned = (text or "").strip()
    
    if not cleaned:
        return {"knowledge_mining": []}
    
    cleaned_no_newline = cleaned.replace('\n', '').replace('\r', '')
    
    # 尝试提取代码块中的 JSON
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
    
    # 尝试直接解析
    for i, c in enumerate(cleaned_no_newline):
        if c == '{':
            try:
                depth = 0
                for j in range(i, len(cleaned_no_newline)):
                    if cleaned_no_newline[j] == '{':
                        depth += 1
                    elif cleaned_no_newline[j] == '}':
                        depth -= 1
                        if depth == 0:
                            return json.loads(cleaned_no_newline[i:j+1])
            except Exception:
                continue
        elif c == '[':
            try:
                depth = 0
                for j in range(i, len(cleaned_no_newline)):
                    if cleaned_no_newline[j] == '[':
                        depth += 1
                    elif cleaned_no_newline[j] == ']':
                        depth -= 1
                        if depth == 0:
                            return json.loads(cleaned_no_newline[i:j+1])
            except Exception:
                continue
    
    return {"knowledge_mining": []}


def generate_kc(model: str, sample: Dict) -> Tuple[Dict, bool, float]:
    """为单个样本生成知识链"""
    sample_id = sample.get("sample_id", "")
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
            
            return {
                "sample_id": sample_id,
                "generated_kc": generated_kc,
                "ground_truth": sample.get("ground_truth", []),
                "raw_response": raw_response,
                "success": True,
                "retry_count": retry_count,
                "api_latency": latency
            }, True, latency
        except Exception as e:
            retry_count += 1
            if retry_count > 3:
                return {
                    "sample_id": sample_id,
                    "error": str(e),
                    "success": False,
                    "retry_count": retry_count
                }, False, 0
            time.sleep(2 ** retry_count)
    
    return {"sample_id": sample_id, "success": False}, False, 0


# ============== 双向处理 ==============
def load_test_samples(num_samples: Optional[int] = None) -> List[Dict]:
    """加载测试样本"""
    test_dir = PROJECT_ROOT / "step6" / "Test"
    samples = []
    
    for f in sorted(test_dir.glob("*.json")):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            samples.append({
                "sample_id": data.get("file_id", f.stem),
                "table_content": data.get("table_content", ""),
                "ground_truth": data.get("knowledge_mining", []),
            })
        except Exception as e:
            logger.log(f"加载 {f} 失败: {e}", "ERROR")
    
    if num_samples:
        samples = samples[:num_samples]
    
    return samples


def load_existing_ids(output_file: Path) -> Set[str]:
    """加载已存在的样本ID"""
    if output_file.exists():
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            return {s.get("sample_id") for s in data.get("samples", [])}
        except:
            pass
    return set()


def write_sample(output_file: Path, result: Dict):
    """线程安全地写入单个样本"""
    lock = get_lock(str(output_file))
    
    with lock:
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # 检查是否已存在
            sample_id = result.get("sample_id")
            existing_ids = {s.get("sample_id") for s in data.get("samples", [])}
            
            if sample_id in existing_ids:
                return False  # 已存在，跳过
            
            # 追加
            data["samples"].append(result)
            data["success_count"] += 1 if result.get("success") else 0
            
            with open(output_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.log(f"写入失败: {e}", "ERROR")
            return False


def merge_and_deduplicate(file1: Path, file2: Path, output: Path):
    """合并两个文件并去重"""
    merged = {}
    
    for f in [file1, file2]:
        if f.exists():
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                for sample in data.get("samples", []):
                    sid = sample.get("sample_id")
                    if sid not in merged:
                        merged[sid] = sample
                    elif sample.get("success") and not merged[sid].get("success"):
                        merged[sid] = sample  # 用成功的替换失败的
            except Exception as e:
                logger.log(f"读取 {f} 失败: {e}", "ERROR")
    
    # 写入合并结果
    output_data = {
        "model": output.stem.replace("api_kc_", ""),
        "merged_from": [str(file1), str(file2)] if file2.exists() else [str(file1)],
        "total_samples": len(merged),
        "success_count": sum(1 for s in merged.values() if s.get("success")),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "samples": list(merged.values())
    }
    
    with open(output, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    return output_data


def process_direction(
    model: str,
    model_key: str,
    samples: List[Dict],
    output_file: Path,
    direction: str,  # "forward" or "backward"
    workers: int = 2,
    stop_event: threading.Event = None,
    processed_ids: Dict[str, threading.Lock] = None,
    check_interval: int = 10
) -> Tuple[int, int]:
    """
    单向处理函数
    
    Args:
        model: API模型名
        model_key: 模型标识符
        samples: 样本列表
        output_file: 输出文件
        direction: "forward" 或 "backward"
        workers: 线程数
        stop_event: 停止事件
        processed_ids: 已处理ID字典（用于检测冲突）
        check_interval: 检查间隔
    
    Returns:
        (处理数量, 成功数量)
    """
    global logger
    
    # 根据方向确定处理顺序
    if direction == "backward":
        samples = list(reversed(samples))
    
    processed_count = 0
    success_count = 0
    
    logger.log(f"[{direction}] 开始处理，样本数: {len(samples)}", "INFO")
    
    def process_one(args):
        idx, sample = args
        
        # 检查是否需要停止
        if stop_event and stop_event.is_set():
            return None
        
        sample_id = sample.get("sample_id")
        
        # 检查是否已被其他进程处理
        if processed_ids is not None:
            with processed_ids.get("lock"):
                if sample_id in processed_ids.get("ids", set()):
                    logger.log(f"[{direction}] 样本 {sample_id} 已被处理，跳过", "INFO")
                    return None
        
        result, success, _ = generate_kc(model, sample)
        
        if success:
            write_sample(output_file, result)
            
            # 更新已处理ID
            if processed_ids is not None:
                with processed_ids.get("lock"):
                    processed_ids["ids"].add(sample_id)
            
            return 1
        return 0
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for idx, sample in enumerate(samples):
            if stop_event and stop_event.is_set():
                break
            futures.append(executor.submit(process_one, (idx, sample)))
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is None:
                    break  # 被其他进程处理或需要停止
                processed_count += 1
                success_count += result
                
                if processed_count % 20 == 0:
                    logger.log(f"[{direction}] 已处理: {processed_count}, 成功: {success_count}", "INFO")
            except Exception as e:
                logger.log(f"[{direction}] 处理异常: {e}", "ERROR")
    
    logger.log(f"[{direction}] 完成！处理: {processed_count}, 成功: {success_count}", "INFO")
    return processed_count, success_count


def run_bidirectional(
    model_key: str,
    model: str,
    samples: List[Dict],
    output_dir: Path,
    workers: int = 2
):
    """
    双向并行处理主函数
    
    1. 正向进程：从头开始处理
    2. 反向进程：从尾开始处理
    3. 使用共享的已处理ID集合检测冲突
    4. 检测到冲突时双方停止
    """
    global logger
    
    model_output_dir = output_dir / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 正向输出文件
    forward_file = model_output_dir / f"api_kc_{model_key}_forward.json"
    backward_file = model_output_dir / f"api_kc_{model_key}_backward.json"
    merged_file = model_output_dir / f"api_kc_{model_key}.json"
    
    # 初始化输出文件
    def init_file(fpath):
        with open(fpath, 'w') as fp:
            json.dump({
                "model": model_key,
                "direction": "forward" if "forward" in str(fpath) else "backward",
                "success_count": 0,
                "samples": []
            }, fp)
    
    # 如果已存在合并文件，直接加载已处理ID
    existing_ids = load_existing_ids(merged_file)
    
    if forward_file.exists():
        existing_forward = load_existing_ids(forward_file)
        existing_ids.update(existing_forward)
    
    if backward_file.exists():
        existing_backward = load_existing_ids(backward_file)
        existing_ids.update(existing_backward)
    
    if not forward_file.exists():
        init_file(forward_file)
    if not backward_file.exists():
        init_file(backward_file)
    
    logger.log(f"[{model_key}] 已有样本数: {len(existing_ids)}", "INFO")
    
    # 共享的已处理ID和锁
    shared_processed = {
        "ids": existing_ids,
        "lock": threading.Lock()
    }
    
    # 停止事件
    stop_event = threading.Event()
    
    # 计算分割点（用于检测冲突）
    # 当两个方向处理到中间区域时，可能产生冲突
    mid_point = len(samples) // 2
    
    def forward_worker():
        forward_samples = samples  # 正向：sample[0], sample[1], ...
        process_direction(
            model, model_key, forward_samples, forward_file,
            "forward", workers, stop_event, shared_processed
        )
    
    def backward_worker():
        backward_samples = list(reversed(samples))  # 反向：sample[-1], sample[-2], ...
        
        # 定期检查是否与正向处理冲突
        def check_and_stop():
            while not stop_event.is_set():
                time.sleep(check_interval)
                
                with shared_processed["lock"]:
                    forward_count = len([s for s in shared_processed["ids"] 
                                        if forward_file.exists() and 
                                        load_existing_ids(forward_file).__contains__(s)])
                    backward_count = len(shared_processed["ids"]) - forward_count
                    
                    # 如果正向和反向处理的ID开始重叠
                    if forward_count > 0 and backward_count > 0:
                        # 检查正向处理的最后一个ID和反向处理的最后一个ID是否接近
                        if forward_count >= mid_point or backward_count >= mid_point:
                            logger.log(f"[{model_key}] 检测到中间区域，停止反向处理", "INFO")
                            stop_event.set()
                            break
        
        # 启动冲突检测线程
        check_thread = threading.Thread(target=check_and_stop)
        check_thread.daemon = True
        check_thread.start()
        
        process_direction(
            model, model_key, backward_samples, backward_file,
            "backward", workers, stop_event, shared_processed
        )
    
    # 启动两个进程
    import multiprocessing as mp
    
    # 使用进程而不是线程（更好的并行性）
    forward_process = mp.Process(target=forward_worker)
    backward_process = mp.Process(target=backward_worker)
    
    forward_process.start()
    backward_process.start()
    
    # 等待完成
    forward_process.join()
    backward_process.join()
    
    # 合并结果
    logger.log(f"[{model_key}] 开始合并结果...", "INFO")
    merged = merge_and_deduplicate(forward_file, backward_file, merged_file)
    logger.log(f"[{model_key}] 合并完成！总样本: {merged['total_samples']}, 成功: {merged['success_count']}", "INFO")
    
    return merged


def main():
    global logger
    
    parser = argparse.ArgumentParser(description="双向 API 知识链生成")
    parser.add_argument("--model", type=str, required=True, 
                       choices=list(MODEL_NAMES.keys()),
                       help="选择的模型")
    parser.add_argument("--workers", type=int, default=2, help="每个方向的线程数")
    parser.add_argument("--num_samples", type=int, default=None, help="处理的样本数")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    args = parser.parse_args()
    
    # 设置
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "api_generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"api_kc_{args.model}_bidirectional_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = ThreadSafeLogger(log_file)
    
    print("=" * 70)
    print(f"  双向 API 知识链生成 - {args.model}")
    print("=" * 70)
    
    # 加载样本
    samples = load_test_samples(args.num_samples)
    print(f"加载了 {len(samples)} 个测试样本")
    
    # 运行双向处理
    model = MODEL_NAMES[args.model]
    result = run_bidirectional(args.model, model, samples, output_dir, args.workers)
    
    print(f"\n✅ {args.model} 完成！")
    print(f"   总样本: {result['total_samples']}")
    print(f"   成功: {result['success_count']}")
    print(f"   输出: {result.get('merged_from', [])}")


if __name__ == "__main__":
    main()
