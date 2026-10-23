#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向多线程 API 知识链生成脚本

功能：
1. 同时开启多线程处理前段和后段样本
2. 当检测到重合时自动停止
3. 最终合并去重

用法：
python api_kc_dual_direction.py --model deepseek-v3.2 --workers 4
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import fcntl

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

# ============== 提示词 ==============
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


class DualDirectionGenerator:
    """双向生成器"""
    
    def __init__(self, model: str, model_key: str, samples: List[Dict], 
                 output_file: Path, workers: int = 4):
        self.model = model
        self.model_key = model_key
        self.samples = samples
        self.output_file = output_file
        self.workers = workers
        
        # 已处理ID集合（线程安全）
        self.processed_ids: Set[str] = set()
        self.lock = threading.Lock()
        
        # 停止标志
        self.stop_flag = threading.Event()
        
        # 统计
        self.stats = {
            "total": len(samples),
            "forward_processed": 0,
            "backward_processed": 0,
            "forward_success": 0,
            "backward_success": 0,
            "duplicate_skipped": 0
        }
        
        # 初始化输出文件
        self._init_output_file()
        
        # 加载已处理的ID
        self._load_existing_ids()
    
    def _init_output_file(self):
        """初始化输出文件"""
        if not self.output_file.exists():
            with open(self.output_file, 'w') as f:
                json.dump({
                    "model": self.model_key,
                    "model_full_name": self.model,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "total_samples": len(self.samples),
                    "success_count": 0,
                    "samples": []
                }, f, ensure_ascii=False, indent=2)
    
    def _load_existing_ids(self):
        """加载已存在的样本ID"""
        try:
            with open(self.output_file, 'r') as f:
                data = json.load(f)
            self.processed_ids = {s.get("sample_id") for s in data.get("samples", [])}
            print(f"[{self.model_key}] 已有样本数: {len(self.processed_ids)}")
        except:
            self.processed_ids = set()
    
    def _is_processed(self, sample_id: str) -> bool:
        """检查样本是否已处理"""
        with self.lock:
            return sample_id in self.processed_ids
    
    def _mark_processed(self, sample_id: str):
        """标记样本已处理"""
        with self.lock:
            self.processed_ids.add(sample_id)
    
    def _should_stop(self) -> bool:
        """检查是否应该停止"""
        # 如果已处理超过80%，停止
        with self.lock:
            processed = len(self.processed_ids)
            total = self.stats["total"]
            return processed >= total * 0.8
    
    def _write_sample(self, sample_data: Dict, success: bool):
        """线程安全写入"""
        sample_id = sample_data.get("sample_id")
        
        with self.lock:
            try:
                with open(self.output_file, 'r') as f:
                    data = json.load(f)
                
                # 检查是否已存在
                existing_ids = {s.get("sample_id") for s in data.get("samples", [])}
                if sample_id in existing_ids:
                    self.stats["duplicate_skipped"] += 1
                    return False
                
                # 追加
                data["samples"].append(sample_data)
                if success:
                    data["success_count"] += 1
                
                with open(self.output_file, 'w') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                return True
            except Exception as e:
                print(f"写入失败: {e}")
                return False
    
    def _call_api(self, messages: List[Dict]) -> Tuple[str, float]:
        """调用API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        payload = {"model": self.model, "messages": messages, "temperature": 0.7}
        
        for retry in range(4):
            try:
                start = time.time()
                response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
                latency = time.time() - start
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"], latency
            except Exception as e:
                if retry < 3:
                    time.sleep(2 ** retry)
                else:
                    raise e
        return "", 0
    
    def _extract_json(self, text: str) -> Dict:
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
    
    def _generate(self, sample: Dict) -> Tuple[Dict, bool]:
        """生成单个样本"""
        sample_id = sample.get("sample_id", "")
        table_content = sample.get("table_content", "")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{INPUT_PROMPT}\n\n<table_content>\n{table_content}\n</table_content>"}
        ]
        
        try:
            raw_response, latency = self._call_api(messages)
            generated_kc = self._extract_json(raw_response)
            
            return {
                "sample_id": sample_id,
                "generated_kc": generated_kc,
                "ground_truth": sample.get("ground_truth", []),
                "table_content": table_content,
                "raw_response": raw_response,
                "success": True,
                "api_latency": latency
            }, True
        except Exception as e:
            return {
                "sample_id": sample_id,
                "generated_kc": {"knowledge_mining": []},
                "ground_truth": sample.get("ground_truth", []),
                "table_content": table_content,
                "error": str(e),
                "success": False
            }, False
    
    def process_forward(self, progress_callback=None):
        """正向处理（从头到尾）"""
        forward_samples = self.samples  # sample[0], sample[1], ...
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for sample in forward_samples:
                if self.stop_flag.is_set() or self._should_stop():
                    break
                futures.append(executor.submit(self._process_single, sample, "forward"))
            
            for future in as_completed(futures):
                if self.stop_flag.is_set() or self._should_stop():
                    break
                try:
                    success = future.result()
                    if success:
                        self.stats["forward_success"] += 1
                    self.stats["forward_processed"] += 1
                    
                    if progress_callback:
                        progress_callback("forward", self.stats)
                except Exception as e:
                    print(f"Forward error: {e}")
    
    def process_backward(self, progress_callback=None):
        """反向处理（从尾到头）"""
        backward_samples = list(reversed(self.samples))  # sample[-1], sample[-2], ...
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            for sample in backward_samples:
                if self.stop_flag.is_set() or self._should_stop():
                    break
                futures.append(executor.submit(self._process_single, sample, "backward"))
            
            for future in as_completed(futures):
                if self.stop_flag.is_set() or self._should_stop():
                    break
                try:
                    success = future.result()
                    if success:
                        self.stats["backward_success"] += 1
                    self.stats["backward_processed"] += 1
                    
                    if progress_callback:
                        progress_callback("backward", self.stats)
                except Exception as e:
                    print(f"Backward error: {e}")
    
    def _process_single(self, sample: Dict, direction: str) -> bool:
        """处理单个样本"""
        sample_id = sample.get("sample_id", "")
        
        # 检查是否已处理
        if self._is_processed(sample_id):
            with self.lock:
                self.stats["duplicate_skipped"] += 1
            return False
        
        # 生成
        sample_data, success = self._generate(sample)
        
        # 标记已处理
        self._mark_processed(sample_id)
        
        # 写入
        if self._write_sample(sample_data, success):
            return success
        
        return False
    
    def _progress_callback(self, direction: str, stats: Dict):
        """进度回调"""
        total = stats["total"]
        processed = len(self.processed_ids)
        pct = 100 * processed / total if total > 0 else 0
        
        msg = f"[{direction}] 已处理: {processed}/{total} ({pct:.1f}%) "
        msg += f"| 正向: {stats['forward_processed']} 成功:{stats['forward_success']} "
        msg += f"| 反向: {stats['backward_processed']} 成功:{stats['backward_success']} "
        msg += f"| 跳过重复: {stats['duplicate_skipped']}"
        
        print(msg)
    
    def run(self):
        """运行双向处理"""
        print(f"\n{'='*60}")
        print(f"  {self.model_key} 双向多线程生成")
        print(f"{'='*60}")
        print(f"总样本数: {len(self.samples)}")
        print(f"线程数: {self.workers} (正向) + {self.workers} (反向)")
        print(f"输出文件: {self.output_file}")
        print(f"{'='*60}\n")
        
        import multiprocessing as mp
        
        # 创建进程
        p1 = mp.Process(target=self.process_forward, args=(self._progress_callback,))
        p2 = mp.Process(target=self.process_backward, args=(self._progress_callback,))
        
        # 启动
        print("启动正向处理...")
        p1.start()
        
        print("启动反向处理...")
        p2.start()
        
        # 等待
        p1.join()
        p2.join()
        
        # 最终统计
        print(f"\n{'='*60}")
        print(f"  完成统计")
        print(f"{'='*60}")
        print(f"总样本: {self.stats['total']}")
        print(f"已处理: {len(self.processed_ids)}")
        print(f"正向成功: {self.stats['forward_success']}")
        print(f"反向成功: {self.stats['backward_success']}")
        print(f"跳过重复: {self.stats['duplicate_skipped']}")
        
        return self.stats


def load_test_samples(num_samples: int = None) -> List[Dict]:
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


def main():
    parser = argparse.ArgumentParser(description="双向多线程 API 生成")
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODEL_NAMES.keys()),
                       help="模型")
    parser.add_argument("--workers", type=int, default=4, help="每个方向线程数")
    parser.add_argument("--num_samples", type=int, default=None, help="样本数")
    args = parser.parse_args()
    
    output_dir = PROJECT_ROOT / "api_generation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model = MODEL_NAMES[args.model]
    output_file = output_dir / f"api_kc_{args.model}.json"
    
    samples = load_test_samples(args.num_samples)
    print(f"加载了 {len(samples)} 个测试样本")
    
    generator = DualDirectionGenerator(
        model=model,
        model_key=args.model,
        samples=samples,
        output_file=output_file,
        workers=args.workers
    )
    
    stats = generator.run()
    
    print(f"\n✅ {args.model} 完成！")


if __name__ == "__main__":
    main()
