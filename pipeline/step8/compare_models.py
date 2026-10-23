#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step8 模型对比测试脚本
测试两个模型：
1. LoRA训练模型 - step8_multi_turn_lora
2. 无混合数据训练模型 - step8_sft_multi_turn
"""

import json
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 配置
BASE_MODEL_PATH = "/home/new_disk/yjw/checkpoints/qwen3_sft1_20260225_183025"

# 两个模型的路径
MODEL_PATHS = {
    "lora": "/home/new_disk/yjw/checkpoints/step8_merged",  # 已合并的LoRA模型
    "sft": "/home/new_disk/yjw/checkpoints/step8_sft_multi_turn/final",  # 直接SFT训练的模型
}

# Step6 提示词
SYSTEM_PROMPT_STEP6 = """你是一名基因组学知识抽取助手，任务是从表格内容(table_content)中生成知识挖掘结果。
输出必须是 JSON，键名为 knowledge_mining，对应一个列表，每个元素包含：
knowledge_chain.condition、knowledge_chain.result、entity_recognition。
仅输出 JSON，不要额外解释。"""

INPUT_PROMPT_STEP6 = """任务：从给定 table_content 抽取知识链并完成实体识别，输出严格 JSON：
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


def load_strategy_prompt():
    prompt_path = Path(__file__).parent / "prompts" / "strategy_selection.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_test_sample():
    """加载一个测试样本"""
    test_dir = Path("/home/weijie_yu/yjw/ECLO-RL/step6/SFT1")
    test_files = sorted(test_dir.glob("*.json"), key=lambda x: int(x.stem))
    if test_files:
        with open(test_files[0], 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def extract_json(text: str):
    """从模型输出中提取 JSON"""
    cleaned = text.strip()
    
    # 尝试从代码块中提取
    if "```" in cleaned:
        blocks = cleaned.split("```")
        for block in blocks:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{") or block.startswith("["):
                try:
                    return json.loads(block)
                except:
                    pass
    
    # 处理以 { 或 [ 开头
    if cleaned.startswith("{"):
        end = cleaned.rfind("}")
        if end > 0:
            try:
                return json.loads(cleaned[:end+1])
            except:
                pass
    
    if cleaned.startswith("["):
        end = cleaned.rfind("]")
        if end > 0:
            try:
                return json.loads(cleaned[:end+1])
            except:
                pass
    
    try:
        return json.loads(cleaned)
    except:
        pass
    
    return None


def load_model(model_path: str):
    """加载模型"""
    print(f"📂 加载模型: {model_path}")
    
    # 尝试从基座模型加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 检查是否是LoRA模型
    is_lora = Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists()
    
    if is_lora:
        print(f"   检测到LoRA模型，合并权重...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    model.eval()
    print("✅ 模型加载完成")
    return model, tokenizer


def test_initial_kc_generation(model, tokenizer, table_content: str):
    """测试能力1: 初始知识链生成"""
    print("\n" + "="*50)
    print("🧪 测试能力1: 初始知识链生成")
    print("="*50)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_STEP6},
        {"role": "user", "content": f"{INPUT_PROMPT_STEP6}\n\n<table_content>\n{table_content}\n</table_content>"},
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=6000)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    print(f"📤 输出 (前1000字符):")
    print("-"*40)
    print(generated_text[:1000])
    print("-"*40)
    
    result = extract_json(generated_text)
    if result and isinstance(result, dict) and 'knowledge_mining' in result:
        kc_count = len(result.get('knowledge_mining', []))
        print(f"\n✅ 成功! 包含 {kc_count} 条知识链")
        return True, result
    else:
        print(f"\n❌ 解析失败")
        return False, None


def test_strategy_selection(model, tokenizer, table_content: str, current_kc: dict):
    """测试能力2: 策略选择"""
    print("\n" + "="*50)
    print("🧪 测试能力2: 策略选择")
    print("="*50)
    
    prompt_template = load_strategy_prompt()
    prompt = prompt_template.replace("{table_content}", table_content)
    prompt = prompt.replace("{current_knowledge_chain}", json.dumps(current_kc, ensure_ascii=False))
    prompt = prompt.replace("{previous_summary_block}", "")
    
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=6000)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    print(f"📤 输出 (前1000字符):")
    print("-"*40)
    print(generated_text[:1000])
    print("-"*40)
    
    result = extract_json(generated_text)
    if result and isinstance(result, list) and len(result) > 0:
        first = result[0]
        if 'action' in first:
            print(f"\n✅ 成功! Action: {first.get('action')}, Strategy: {first.get('strategy')}")
            return True, result
    elif result and isinstance(result, dict) and 'action' in result:
        print(f"\n✅ 成功! Action: {result.get('action')}, Strategy: {result.get('strategy')}")
        return True, result
    
    print(f"\n❌ 解析失败")
    return False, None


def test_model(model_name: str, model_path: str):
    """测试单个模型"""
    print(f"\n{'#'*60}")
    print(f"# 测试模型: {model_name}")
    print(f"# 路径: {model_path}")
    print(f"{'#'*60}")
    
    # 加载模型
    model, tokenizer = load_model(model_path)
    
    # 加载测试样本
    test_data = load_test_sample()
    if test_data is None:
        print("❌ 无法加载测试样本")
        return None, None
    
    table_content = test_data.get("table_content", "")
    print(f"\n📋 表格内容长度: {len(table_content)} 字符")
    
    # 测试1: 初始生成
    success1, initial_kc = test_initial_kc_generation(model, tokenizer, table_content)
    
    # 测试2: 策略选择
    if initial_kc and isinstance(initial_kc, dict) and initial_kc.get("knowledge_mining"):
        test_kc = initial_kc
    else:
        test_kc = {"knowledge_mining": [{"knowledge_chain": {"condition": "Test", "result": "wrong"}, "entity_recognition": {}}]}
    
    success2, strategy_result = test_strategy_selection(model, tokenizer, table_content, test_kc)
    
    return success1, success2


def main():
    print("="*60)
    print("🔬 Step8 模型对比测试")
    print("="*60)
    
    results = {}
    
    # 测试 LoRA 模型
    results["lora"] = test_model("LoRA训练模型 (有SFT1混合)", MODEL_PATHS["lora"])
    
    # 测试 SFT 模型
    results["sft"] = test_model("直接SFT模型 (无SFT1混合)", MODEL_PATHS["sft"])
    
    # 总结
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    
    print(f"\n{'模型':<25} {'初始生成':<15} {'策略选择':<15}")
    print("-"*55)
    
    labels = {"lora": "LoRA模型(有混合)", "sft": "SFT模型(无混合)"}
    for key, (s1, s2) in results.items():
        status1 = "✅ 通过" if s1 else "❌ 失败"
        status2 = "✅ 通过" if s2 else "❌ 失败"
        print(f"{labels[key]:<25} {status1:<15} {status2:<15}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

