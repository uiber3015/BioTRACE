#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 closed_loop_temp.json 构建多轮训练数据 (正确筛选版本)

基于 extract_training_data_multi_turn.py 的筛选逻辑
"""

import json
import os
from collections import Counter

# 配置
INPUT_PATH = "/home/weijie_yu/yjw/ECLO-RL/step7/closed_loop_results/closed_loop_temp.json"
OUTPUT_DIR = "/home/weijie_yu/yjw/ECLO-RL/step8/training_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_training_data_multi_turn_v2.json")

VALID_STRATEGIES = [
    "Modify/condition_kv", "Modify/entity_type", "Modify/entity_value",
    "Modify/result", "Modify/entity_recognition",
    "Add/chain", "Add/entity_type", "Add/entity_value",
    "Delete/chain", "Delete/condition_kv", "Delete/entity_type", "Delete/entity_value",
    "STOP"
]


def check_score_improving(correction_history):
    """检查分数是否持续改善（允许相等）"""
    scores = []
    for ch in correction_history:
        score = ch.get("r_rule_score", 0)
        if score is None:
            score = 0
        scores.append(score)
    
    # 过滤掉全0的样本
    if all(s == 0 for s in scores):
        return False
    
    # 检查是否单调递增（允许相等）
    for i in range(len(scores) - 1):
        if scores[i] > 0 and scores[i+1] > 0:
            if scores[i] > scores[i+1]:
                return False
    
    return True


# Task prompt (与原始脚本一致)
TASK_PROMPT = """## Table
{table_content}

## Current Knowledge Chain
```json
{knowledge_chain}
```
{history_text}
{summary}

## Task
Compare the table and current knowledge chain, select correction strategy:
- If knowledge chain is consistent with table, output STOP
- If there are issues, output correction strategy

Output format (JSON):
{{
  "action": "CORRECT" or "STOP",
  "strategy": "correction strategy type",
  "chain_indices": [knowledge chain indices to correct]
}}

Start correction:"""


def build_multi_turn_data():
    """构建多轮训练数据"""
    print(f"📂 加载原始数据: {INPUT_PATH}")
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"✅ 加载了 {len(raw_data)} 个样本")
    
    # 筛选：分数递增
    print("\n🔍 筛选分数递增的样本...")
    filtered_data = []
    score_filtered = 0
    
    for sample in raw_data:
        correction_history = sample.get("correction_history", [])
        if not correction_history:
            score_filtered += 1
            continue
        if check_score_improving(correction_history):
            filtered_data.append(sample)
        else:
            score_filtered += 1
    
    print(f"   筛选后: {len(filtered_data)} / {len(raw_data)} 样本")
    print(f"   被过滤: {score_filtered}")
    
    # 构建训练数据
    training_data = []
    statistics = {
        "total_samples": len(raw_data),
        "valid_samples": 0,
        "score_filtered": score_filtered,
        "rounds_distribution": Counter(),
        "action_distribution": Counter(),
        "strategy_distribution": Counter()
    }
    
    for sample in filtered_data:
        sample_id = sample['sample_id']
        table_content = sample['table_content']
        initial_kc = sample.get('initial_generated', {})
        correction_history = sample.get('correction_history', [])
        
        if not correction_history:
            continue
        
        # 构建知识链历史
        knowledge_chains = [initial_kc]
        for correction in correction_history:
            corrected = correction.get("corrected_json", {})
            if corrected:
                knowledge_chains.append(corrected)
        
        # 每一轮生成一个训练样本
        for round_idx, h in enumerate(correction_history):
            action = h.get('action', 'CORRECT')
            strategy = h.get('strategy', '')
            student_op = h.get('student_operation', {})
            chain_indices = student_op.get('chain_indices', [])
            teacher_summary = h.get('teacher_summary', '')
            
            # 过滤无效策略
            if action != "STOP" and strategy not in VALID_STRATEGIES:
                continue
            
            # 过滤chain_indices过长的情况
            if chain_indices is None:
                chain_indices = []
            if len(chain_indices) > 7:
                continue
            
            # 构建历史修正记录
            history_text = ""
            if round_idx > 0:
                history_text = "\n## History Corrections\n"
                for h_idx in range(round_idx):
                    h_action = correction_history[h_idx].get("action", "CORRECT")
                    h_strategy = correction_history[h_idx].get("strategy", "")
                    h_chain_indices = correction_history[h_idx].get("student_operation", {}).get("chain_indices", [])
                    h_reason = correction_history[h_idx].get("student_operation", {}).get("reason", "")
                    
                    if h_reason:
                        history_text += f"- Round {h_idx}: {h_action} - {h_strategy}, indices: {h_chain_indices}, reason: {h_reason}\n"
                    else:
                        history_text += f"- Round {h_idx}: {h_action} - {h_strategy}, indices: {h_chain_indices}\n"
            
            # 获取上一轮的修正总结
            if round_idx == 0:
                summary = ""
            else:
                summary = f"\n## Previous Correction Summary\n{correction_history[round_idx - 1].get('teacher_summary', '')}"
            
            # 构建 prompt
            current_kc = knowledge_chains[round_idx] if round_idx < len(knowledge_chains) else knowledge_chains[-1]
            kc_text = json.dumps(current_kc, ensure_ascii=False, indent=2)
            
            prompt = TASK_PROMPT.format(
                table_content=table_content,
                knowledge_chain=kc_text,
                history_text=history_text,
                summary=summary
            )
            
            # 构建 response
            if action in ['STOP', 'TEACHER_STOP']:
                response = json.dumps({
                    "action": "STOP",
                    "strategy": "STOP",
                    "chain_indices": []
                }, ensure_ascii=False)
            elif action in ['CORRECT', 'EMPTY_OPERATION', 'EMPTY_RESULT']:
                response = json.dumps({
                    "action": action,
                    "strategy": strategy,
                    "chain_indices": chain_indices
                }, ensure_ascii=False)
            else:
                continue
            
            # 添加到训练数据
            training_data.append({
                "id": f"{sample_id}_round_{round_idx}",
                "conversations": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "metadata": {
                    "sample_id": sample_id,
                    "round": round_idx,
                    "action": action,
                    "strategy": strategy,
                    "chains_before": h.get('chains_before', 0),
                    "chains_after": h.get('chains_after', 0)
                }
            })
            
            # 统计
            statistics["action_distribution"][action] += 1
            statistics["strategy_distribution"][strategy if action != "STOP" else "STOP"] += 1
        
        statistics["valid_samples"] += 1
        statistics["rounds_distribution"][len(correction_history)] += 1
    
    # 统计
    print(f"\n✅ 构建完成!")
    print(f"   总训练样本: {len(training_data)}")
    
    # 统计轮数分布
    print(f"   轮数分布:")
    for r in sorted(statistics["rounds_distribution"].keys()):
        print(f"     Round {r}: {statistics['rounds_distribution'][r]}")
    
    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 已保存到: {OUTPUT_FILE}")
    
    # 保存统计信息
    stats_output = OUTPUT_FILE.replace('.json', '_statistics.json')
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": statistics["total_samples"],
            "valid_samples": statistics["valid_samples"],
            "score_filtered": statistics["score_filtered"],
            "rounds_distribution": dict(statistics["rounds_distribution"]),
            "action_distribution": dict(statistics["action_distribution"]),
            "strategy_distribution": dict(statistics["strategy_distribution"])
        }, f, ensure_ascii=False, indent=2)
    
    # 显示一个样本示例
    print("\n=== 示例样本 ===")
    sample = training_data[0]
    print(f"ID: {sample['id']}")
    print(f"轮数: {sample['metadata']['round']}")
    print(f"\nUser (前800字):")
    print(sample['conversations'][0]['content'][:800])
    print(f"\nAssistant:")
    print(sample['conversations'][1]['content'])


if __name__ == "__main__":
    build_multi_turn_data()
