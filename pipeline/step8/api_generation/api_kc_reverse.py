#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反向 API 知识链生成脚本（从后往前处理）

与正向进程并行运行，检测到样本重合时自动停止

用法：
python api_kc_reverse.py --model deepseek-v3.2 --workers 2
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

API_URL = os.environ.get("LLM_API_URL", "")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MODEL_NAMES = {
    "deepseek-v3.2": "DeepSeek-V3.2",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
}

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


def load_samples(num_samples: int = None) -> List[Dict]:
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
            print(f"加载 {f} 失败: {e}")
    
    if num_samples:
        samples = samples[:num_samples]
    
    return samples


def load_forward_ids(model_key: str) -> Set[str]:
    """加载正向进程已处理的ID"""
    forward_file = PROJECT_ROOT / "api_generation" / f"api_kc_{model_key}.json"
    if forward_file.exists():
        try:
            with open(forward_file, 'r') as f:
                data = json.load(f)
            return {s.get("sample_id") for s in data.get("samples", [])}
        except:
            pass
    return set()


def init_reverse_file(model_key: str):
    """初始化反向输出文件"""
    reverse_file = PROJECT_ROOT / "api_generation" / f"api_kc_{model_key}_reverse.json"
    if not reverse_file.exists():
        with open(reverse_file, 'w') as f:
            json.dump({
                "model": model_key,
                "direction": "backward",
                "success_count": 0,
                "samples": []
            }, f)
    return reverse_file


def get_reverse_ids(reverse_file: Path) -> Set[str]:
    """获取反向进程已处理的ID"""
    if reverse_file.exists():
        try:
            with open(reverse_file, 'r') as f:
                data = json.load(f)
            return {s.get("sample_id") for s in data.get("samples", [])}
        except:
            pass
    return set()


def write_sample(reverse_file: Path, sample_data: Dict):
    """写入样本"""
    with threading.Lock():
        try:
            with open(reverse_file, 'r') as f:
                data = json.load(f)
            
            sample_id = sample_data.get("sample_id")
            existing_ids = {s.get("sample_id") for s in data.get("samples", [])}
            
            if sample_id in existing_ids:
                return False
            
            data["samples"].append(sample_data)
            data["success_count"] += 1 if sample_data.get("success") else 0
            
            with open(reverse_file, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"写入失败: {e}")
            return False


def call_api(model: str, messages: List[Dict]) -> str:
    """调用API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {"model": model, "messages": messages, "temperature": 0.7}
    
    for retry in range(4):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            if retry < 3:
                time.sleep(2 ** retry)
            else:
                raise e
    return ""


def extract_json(text: str) -> Dict:
    """提取JSON"""
    cleaned = (text or "").strip().replace('\n', '').replace('\r', '')
    
    if "```" in cleaned:
        blocks = cleaned.split("```")
        for block in blocks:
            candidate = block.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                try:
                    return json.loads(candidate)
                except:
                    pass
    
    for i, c in enumerate(cleaned):
        if c == '{':
            try:
                for j in range(i, len(cleaned)):
                    if cleaned[j] == '}':
                        return json.loads(cleaned[i:j+1])
            except:
                continue
    return {"knowledge_mining": []}


def generate(sample: Dict, model: str) -> Dict:
    """生成单个样本"""
    sample_id = sample.get("sample_id", "")
    table_content = sample.get("table_content", "")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INPUT_PROMPT}\n\n<table_content>\n{table_content}\n</table_content>"}
    ]
    
    try:
        raw = call_api(model, messages)
        kc = extract_json(raw)
        return {
            "sample_id": sample_id,
            "generated_kc": kc,
            "ground_truth": sample.get("ground_truth", []),
            "table_content": table_content,
            "raw_response": raw,
            "success": True
        }
    except Exception as e:
        return {
            "sample_id": sample_id,
            "generated_kc": {"knowledge_mining": []},
            "error": str(e),
            "success": False
        }


def process_reverse(model_key: str, model: str, samples: List[Dict], workers: int = 2):
    """反向处理主函数"""
    print(f"\n{'='*60}")
    print(f"  {model_key} 反向处理（从后往前）")
    print(f"{'='*60}")
    
    # 初始化反向文件
    reverse_file = init_reverse_file(model_key)
    
    # 已处理的ID集合（正向+反向）
    processed_ids = load_forward_ids(model_key)
    all_reverse_ids = get_reverse_ids(reverse_file)
    processed_ids.update(all_reverse_ids)
    
    print(f"正向已处理: {len(load_forward_ids(model_key))}")
    print(f"反向已处理: {len(all_reverse_ids)}")
    print(f"输出文件: {reverse_file}")
    print(f"{'='*60}\n")
    
    # 反向样本（从后往前）
    reverse_samples = list(reversed(samples))
    
    # 需要处理的样本（排除已处理的）
    samples_to_process = [
        (idx, s) for idx, s in enumerate(reverse_samples)
        if s.get("sample_id") not in processed_ids
    ]
    
    print(f"需要处理: {len(samples_to_process)} 个样本")
    
    success_count = 0
    processed = 0
    local_lock = threading.Lock()
    
    def worker(args):
        nonlocal success_count, processed
        idx, sample = args
        
        # 每次处理前检查是否与正向重合
        current_forward_ids = load_forward_ids(model_key)
        
        # 如果正向处理到了后面的样本，可能有冲突
        # 保守策略：如果当前处理的样本在正向已处理范围之外不远，就停止
        sample_id = sample.get("sample_id")
        
        # 快速检查
        if sample_id in current_forward_ids:
            print(f"[反向] 样本 {sample_id} 已被正向处理，停止")
            return False
        
        result = generate(sample, model)
        
        if write_sample(reverse_file, result):
            with local_lock:
                processed_ids.add(sample_id)
                success_count += 1 if result.get("success") else 0
        
        return result.get("success")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for args in samples_to_process:
            futures.append(executor.submit(worker, args))
        
        for future in as_completed(futures):
            processed += 1
            try:
                future.result()
            except:
                pass
            
            if processed % 10 == 0:
                elapsed = time.time() - start_time
                pct = 100 * processed / len(samples_to_process) if samples_to_process else 100
                print(f"[反向] 进度: {processed}/{len(samples_to_process)} ({pct:.1f}%) | 成功: {success_count} | 耗时: {elapsed:.0f}s")
    
    print(f"\n✅ {model_key} 反向处理完成！")
    print(f"   处理: {processed} | 成功: {success_count}")


def main():
    parser = argparse.ArgumentParser(description="反向 API 生成")
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODEL_NAMES.keys()),
                       help="模型")
    parser.add_argument("--workers", type=int, default=2, help="线程数")
    args = parser.parse_args()
    
    samples = load_samples()
    print(f"加载了 {len(samples)} 个样本")
    
    model = MODEL_NAMES[args.model]
    process_reverse(args.model, model, samples, args.workers)


if __name__ == "__main__":
    main()
