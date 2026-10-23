#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据提取脚本 - 从 closed_loop_results 中提取完整轨迹训练数据

训练目标（完整轨迹）：
- 输入：表格 + 初始知识链
- 输出：所有修正轮次的完整序列（action, strategy, chain_indices）

优势：
- 训练样本数 = 原始样本数（约3k），而不是拆分的1万多
- 模型学会完整的多轮修正过程
- 更好对齐实际使用场景
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent

CLOSED_LOOP_DIR = PROJECT_ROOT / "step7" / "closed_loop_results"
OUTPUT_DIR = PROJECT_ROOT / "step8" / "training_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_STRATEGIES = [
    "Modify/condition_kv", "Modify/entity_type", "Modify/entity_value",
    "Modify/result", "Modify/entity_recognition",
    "Add/chain", "Add/entity_type", "Add/entity_value",
    "Delete/chain", "Delete/condition_kv", "Delete/entity_type", "Delete/entity_value",
    "STOP"
]

# 是否只保留分数单调递增的样本
FILTER_SCORE_INCREASING = True


def check_score_increasing(correction_history: List[Dict]) -> bool:
    """
    检查分数是否单调递增（允许相等）
    """
    scores = []
    for ch in correction_history:
        score = ch.get("r_rule_score", 0)
        if score is None:
            score = 0
        scores.append(score)
    
    if all(s == 0 for s in scores):
        return False
    
    for i in range(len(scores) - 1):
        if scores[i] > 0 and scores[i+1] > 0:
            if scores[i] > scores[i+1]:
                return False
    
    return True


def load_closed_loop_data() -> List[Dict]:
    print("📁 加载 closed loop 数据...")
    all_data = []
    
    for fname in [
        "closed_loop_temp.json",
        "closed_loop_temp copy.json",
        "closed_loop_temp copy 2.json",
        "closed_loop_temp copy 3.json",
        "closed_loop_raw_deepseek_20260211_005130.json"
    ]:
        fpath = CLOSED_LOOP_DIR / fname
        if fpath.exists():
            print(f"  加载: {fname}")
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
    
    seen_ids = set()
    unique_data = []
    for sample in all_data:
        sid = sample.get('sample_id')
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_data.append(sample)
    
    print(f"✅ 总样本数: {len(unique_data)}")
    return unique_data


def extract_full_trajectory_samples(data: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    提取完整轨迹训练样本
    
    每个样本包含：
    - 输入: 表格 + 初始知识链
    - 输出: 所有修正轮次的完整序列
    """
    print("\n🔄 提取完整轨迹训练样本...")
    
    training_samples = []
    statistics = {
        "total_samples": len(data),
        "valid_samples": 0,
        "score_increasing_filtered": 0,
        "rounds_distribution": Counter()
    }
    
    # 筛选分数单调递增的样本
    if FILTER_SCORE_INCREASING:
        print("\n🔍 筛选分数单调递增的样本...")
        filtered_data = []
        for sample in data:
            correction_history = sample.get("correction_history", [])
            if not correction_history:
                continue
            if check_score_increasing(correction_history):
                filtered_data.append(sample)
            else:
                statistics["score_increasing_filtered"] += 1
        print(f"  筛选后: {len(filtered_data)} / {len(data)} 样本")
        data = filtered_data
    
    for sample in data:
        sample_id = sample.get("sample_id", "unknown")
        table_content = sample.get("table_content", "")
        correction_history = sample.get("correction_history", [])
        
        if not correction_history:
            continue
        
        # 验证所有轮次都有效
        valid_rounds = []
        for i, correction in enumerate(correction_history):
            action = correction.get("action", "CORRECT")
            strategy = correction.get("strategy", "")
            
            student_op = correction.get("student_operation", {})
            chain_indices = student_op.get("chain_indices", []) if student_op else []
            
            if chain_indices is None:
                chain_indices = []
            
            # 筛选有效策略
            is_stop = (action == "STOP")
            if not is_stop and strategy not in VALID_STRATEGIES:
                continue
            if not is_stop and len(chain_indices) > 7:
                continue
            
            valid_rounds.append({
                "round": i,
                "action": action,
                "strategy": strategy if not is_stop else "STOP",
                "chain_indices": chain_indices
            })
        
        # 至少需要有一轮有效修正
        if len(valid_rounds) == 0:
            continue
        
        statistics["valid_samples"] += 1
        statistics["rounds_distribution"][len(valid_rounds)] += 1
        
        train_sample = {
            "id": sample_id,
            "table_content": table_content,
            "initial_knowledge_chain": sample.get("initial_generated", {}),
            "correction_rounds": valid_rounds,
            "num_rounds": len(valid_rounds)
        }
        
        training_samples.append(train_sample)
    
    return training_samples, statistics


def format_for_sft_full_trajectory(sample: Dict) -> Dict:
    """
    格式化为完整轨迹 SFT 训练数据
    
    对话格式：
    - user: 表格 + 初始知识链
    - assistant: 所有修正轮次的完整序列（JSON数组格式）
    """
    table_content = sample["table_content"]
    initial_kc = sample.get("initial_knowledge_chain", {})
    correction_rounds = sample["correction_rounds"]
    
    # 将知识链格式化为可读文本
    kc_text = json.dumps(initial_kc, ensure_ascii=False, indent=2)
    
    # 构建用户输入（包含表格和初始知识链）
    user_content = f"""## 任务
你是一个知识链修正助手。根据以下表格内容和当前知识链，选择修正策略进行迭代修正，直到知识链正确无误。

## 表格内容
{table_content}

## 当前知识链
{kc_text}

请根据上述表格内容和当前知识链，进行修正。
输出格式为JSON数组，每一轮修正包含：
- action: "CORRECT" 或 "STOP"
- strategy: 修正策略类型
- chain_indices: 需要修正的知识链索引

开始修正："""
    
    # 构建assistant输出（完整修正序列）
    output_rounds = []
    for round_data in correction_rounds:
        output_rounds.append({
            "round": round_data["round"],
            "action": round_data["action"],
            "strategy": round_data["strategy"],
            "chain_indices": round_data["chain_indices"]
        })
    
    assistant_content = json.dumps(output_rounds, ensure_ascii=False, indent=2)
    
    return {
        "id": sample["id"],
        "conversations": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def main():
    print("=" * 80)
    print("Step 8: 提取完整轨迹训练数据")
    print("=" * 80)
    
    # 加载数据
    data = load_closed_loop_data()
    
    # 提取样本
    training_samples, statistics = extract_full_trajectory_samples(data)
    
    print("\n📊 统计信息:")
    print(f"  有效样本数: {statistics['valid_samples']}")
    print(f"  分数递增筛选: {statistics['score_increasing_filtered']}")
    print(f"  轮次分布: {dict(statistics['rounds_distribution'])}")
    
    # 格式化为 SFT 数据
    print("\n🔄 格式化为 SFT 数据...")
    sft_data = [format_for_sft_full_trajectory(s) for s in training_samples]
    
    # 保存
    output_path = OUTPUT_DIR / "sft_training_data_full_trajectory.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 训练数据已保存: {output_path}")
    print(f"  总样本数: {len(sft_data)}")
    
    # 保存统计信息
    stats_path = OUTPUT_DIR / "statistics_full_trajectory.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 统计信息已保存: {stats_path}")


if __name__ == "__main__":
    main()

