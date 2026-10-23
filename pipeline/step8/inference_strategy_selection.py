#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8: 策略选择推理脚本

功能：
- 加载训练好的模型
- 对输入的表格和知识链进行策略选择
- 输出修正指令 {action, strategy, chain_indices, reason}
- 调用修正器执行修正
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 配置
MODEL_PATH = PROJECT_ROOT / "step8" / "output" / "final"
DEFAULT_MODEL_PATH = "/home/new_disk/yjw/checkpoints/qwen3_sft1"

# 修正器 prompt 路径
CORRECTOR_PROMPT_PATH = PROJECT_ROOT / "step8" / "prompts" / "corrector.txt"

# 策略到 operation 的映射表
STRATEGY_TO_OPERATION = {
    "STOP": "无需修正",
    "Modify/condition_kv": "修改指定链的 condition 字段",
    "Modify/result": "修改指定链的 result 字段",
    "Modify/entity_type": "修改实体类型标签",
    "Modify/entity_value": "修改实体值拼写",
    "Modify/entity_recognition": "修改实体识别结果",
    "Add/chain": "添加新知识链",
    "Add/entity_type": "添加实体类型标签",
    "Add/entity_value": "添加实体值",
    "Delete/chain": "删除指定知识链",
    "Delete/condition_kv": "删除 condition 字段",
    "Delete/entity_type": "删除实体类型标签",
    "Delete/entity_value": "删除实体值"
}


def load_model_and_tokenizer(model_path: Optional[str] = None):
    """加载模型和 tokenizer"""
    if model_path is None:
        model_path = str(MODEL_PATH) if MODEL_PATH.exists() else DEFAULT_MODEL_PATH
    
    print(f"📁 加载模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("✅ 模型加载完成")
    return model, tokenizer


def build_prompt(
    table_content: str, 
    current_knowledge_chain: Dict,
    history: List[Dict] = [],
    previous_summary: str = ""
) -> str:
    """构建输入 prompt - 与训练数据格式一致"""
    user_parts = [f"""## Table
{table_content}

## Current Knowledge Chain
```json
{json.dumps(current_knowledge_chain, ensure_ascii=False)}
```"""]
    
    # 添加历史修正记录
    if history:
        history_text = "\n## History Corrections\n"
        for h in history:
            reason = h.get("reason", "")
            if reason:
                history_text += f"- Round {h['round']}: {h['action']} - {h['strategy']}, indices: {h['chain_indices']}, reason: {reason}\n"
            else:
                history_text += f"- Round {h['round']}: {h['action']} - {h['strategy']}, indices: {h['chain_indices']}\n"
        user_parts.append(history_text)
    
    # 添加上一轮修正总结
    if previous_summary:
        user_parts.append(f"""## Previous Correction Summary
{previous_summary}""")
    
    user_parts.append("""## Task
Compare the table and current knowledge chain, select correction strategy:
- If knowledge chain is consistent with table, output STOP
- If there are issues, output correction strategy

Output format (JSON):
{
  "action": "CORRECT" or "STOP",
  "strategy": "correction strategy type",
  "chain_indices": [knowledge chain indices to correct]
}

Start correction:""")
    
    return "\n".join(user_parts)


def parse_response(response_text: str) -> Dict:
    """解析模型输出"""
    try:
        # 尝试提取 JSON
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        result = json.loads(response_text)
        return result
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[^{}]*\}', response_text)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        
        return {
            "action": "ERROR",
            "strategy": "PARSE_ERROR",
            "chain_indices": [],
            "reason": "",
            "raw_response": response_text
        }


def select_strategy(
    model,
    tokenizer,
    table_content: str,
    current_knowledge_chain: Dict,
    history: List[Dict] = [],
    previous_summary: str = "",
    max_length: int = 8192
) -> Dict:
    """执行策略选择"""
    prompt = build_prompt(table_content, current_knowledge_chain, history, previous_summary)
    
    # 构建输入格式
    input_text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,  # 增加长度以容纳 reason
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # 解析
    result = parse_response(response_text)
    
    # 添加 operation 字段
    if "strategy" in result and result["strategy"] in STRATEGY_TO_OPERATION:
        result["operation"] = STRATEGY_TO_OPERATION[result["strategy"]]
    else:
        result["operation"] = "未知操作"
    
    return result


def batch_select_strategy(
    model,
    tokenizer,
    samples: List[Dict],
    max_length: int = 8192
) -> List[Dict]:
    """批量执行策略选择"""
    results = []
    
    for i, sample in enumerate(samples):
        print(f"处理样本 {i+1}/{len(samples)}: {sample.get('id', 'unknown')}")
        
        history = sample.get("history", [])
        previous_summary = sample.get("previous_summary", "")
        
        result = select_strategy(
            model,
            tokenizer,
            sample["table_content"],
            sample["current_knowledge_chain"],
            history,
            previous_summary,
            max_length
        )
        
        result["sample_id"] = sample.get("id", i)
        
        # 添加真实值用于评估
        if "action" in sample:
            result["ground_truth"] = {
                "action": sample["action"],
                "strategy": sample["strategy"],
                "chain_indices": sample["chain_indices"],
                "reason": sample.get("reason", "")
            }
        
        results.append(result)
    
    return results


def evaluate_results(results: List[Dict]) -> Dict:
    """评估结果"""
    total = len(results)
    if total == 0:
        return {"error": "No results to evaluate"}
    
    action_correct = 0
    strategy_correct = 0
    chain_indices_correct = 0
    exact_match = 0
    
    for result in results:
        if "ground_truth" not in result:
            continue
        
        gt = result["ground_truth"]
        
        if result.get("action") == gt.get("action"):
            action_correct += 1
        
        if result.get("strategy") == gt.get("strategy"):
            strategy_correct += 1
        
        pred_indices = set(result.get("chain_indices", []))
        gt_indices = set(gt.get("chain_indices", []))
        if pred_indices == gt_indices:
            chain_indices_correct += 1
        
        if (result.get("action") == gt.get("action") and 
            result.get("strategy") == gt.get("strategy") and
            pred_indices == gt_indices):
            exact_match += 1
    
    return {
        "total": total,
        "action_accuracy": action_correct / total,
        "strategy_accuracy": strategy_correct / total,
        "chain_indices_accuracy": chain_indices_correct / total,
        "exact_match_accuracy": exact_match / total,
        "action_correct": action_correct,
        "strategy_correct": strategy_correct,
        "chain_indices_correct": chain_indices_correct,
        "exact_match": exact_match
    }


def load_corrector_prompt() -> str:
    """加载修正器 prompt"""
    if CORRECTOR_PROMPT_PATH.exists():
        with open(CORRECTOR_PROMPT_PATH, 'r', encoding='utf-8') as f:
            return f.read()
    return ""


def build_corrector_input(
    table_content: str,
    current_knowledge_chain: Dict,
    strategy: str,
    chain_indices: List[int],
    reason: str,
    previous_summary: str = ""
) -> str:
    """构建修正器的输入 prompt"""
    
    prompt = f"""## 表格
{table_content}

## 当前知识链
```json
{json.dumps(current_knowledge_chain, ensure_ascii=False)}
```

## 修正指令
- 策略: {strategy}
- 修正链索引: {chain_indices}
- 修正原因: {reason}
"""
    
    if previous_summary:
        prompt += f"""
## 上一轮修正总结
{previous_summary}
"""
    
    prompt += """

请根据上述修正指令执行修正，输出修正后的完整知识链（JSON格式）。

注意：
1. 只修改指定的知识链（根据 chain_indices）
2. 按照 strategy 描述的方式进行修改
3. 保持其他知识链不变
4. 直接输出 JSON，不要其他内容。"""
    
    return prompt


def main():
    """主函数 - 测试推理"""
    print("=" * 80)
    print("Step 8: 策略选择推理测试")
    print("=" * 80)
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer()
    
    # 加载测试数据
    test_data_path = PROJECT_ROOT / "step8" / "training_data" / "sft_training_data_multi_turn.json"
    print(f"\n📁 加载测试数据: {test_data_path}")
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 取前 10 个样本测试
    test_samples = test_data[:10]
    print(f"测试样本数: {len(test_samples)}")
    
    # 执行推理
    results = batch_select_strategy(model, tokenizer, test_samples)
    
    # 评估
    evaluation = evaluate_results(results)
    
    print("\n📊 评估结果:")
    print(f"  - Action 准确率: {evaluation['action_accuracy']:.2%}")
    print(f"  - Strategy 准确率: {evaluation['strategy_accuracy']:.2%}")
    print(f"  - Chain Indices 准确率: {evaluation['chain_indices_accuracy']:.2%}")
    print(f"  - 完全匹配准确率: {evaluation['exact_match_accuracy']:.2%}")
    
    # 保存结果
    output_path = PROJECT_ROOT / "step8" / "output" / "inference_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "evaluation": evaluation,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
