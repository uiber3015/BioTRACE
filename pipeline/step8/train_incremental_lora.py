#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8 增量学习训练脚本 - LoRA + 混合训练

目标：保留 Step6 的初始生成能力，同时学习 Step8 的策略选择能力

数据混合策略：
1. 初始生成数据来自 step6/SFT1/
2. 策略选择数据来自 step8/training_data/sft_training_data_multi_turn_v2.json

使用方法:
conda activate openr1
python step8/train_incremental_lora.py
"""

import os
import json
import sys
import random
from pathlib import Path
from typing import Dict, List
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 配置
PROJECT_ROOT = Path(__file__).parent.parent

# 基座模型 (Step6 训练后的模型)
BASE_MODEL_PATH = "/home/new_disk/yjw/checkpoints/qwen3_sft1_20260225_183025"

# 数据路径
SFT1_DATA_DIR = PROJECT_ROOT / "step6" / "SFT1"
STRATEGY_DATA_PATH = PROJECT_ROOT / "step8" / "training_data" / "sft_training_data_multi_turn_v2.json"

# 输出路径
OUTPUT_DIR = Path("/home/new_disk/yjw/checkpoints/step8_incremental_lora")
LOGS_DIR = PROJECT_ROOT / "step8" / "logs"

# 训练超参数
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 3e-4  # LoRA 可以用稍大的学习率
NUM_EPOCHS = 2
MAX_SEQ_LENGTH = 8192

# LoRA 配置
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Step6 系统提示词 (初始生成)
SYSTEM_PROMPT_SFT1 = (
    "你是一名生物医学知识抽取助手，任务是从表格内容(table_content)中生成知识挖掘结果。"
    "输出必须是 JSON，键名为 knowledge_mining，对应一个列表，每个元素包含："
    "knowledge_chain.condition、knowledge_chain.result、entity_recognition。"
    "仅输出 JSON，不要额外解释。"
)

INPUT_PROMPT_SFT1 = """任务：从给定 table_content 抽取知识链并完成实体识别，输出严格 JSON：
{
  "knowledge_mining": [
    {
      "knowledge_chain": {"condition": "...", "result": "..."},
      "entity_recognition": {"<entity_type>": ["实体1", "实体2"]}
    }
  ]
}

知识链规则：
1) 遍历表格每行：每个数值字段至少一条链
2) result 仅 1 个键值对，格式 "指标:数值"
3) condition 仅非数值/分组信息

实体识别规则：
1) 实体值必须在 condition 或 result 原文中完整匹配
2) 仅允许 14 类实体

请直接输出合法 JSON，不要额外文本。"""


def load_sft1_data(data_dir: Path) -> List[Dict]:
    """加载 Step6 SFT1 初始生成数据"""
    print(f"📂 加载 SFT1 数据: {data_dir}")
    samples = []
    
    for p in sorted(data_dir.glob("*.json"), key=lambda x: int(x.stem)):
        try:
            data = json.load(open(p))
        except Exception:
            continue
        
        table_content = data.get("table_content", "").strip()
        km = data.get("knowledge_mining", [])
        if not table_content or not km:
            continue
        
        target_json = json.dumps({"knowledge_mining": km}, ensure_ascii=False)
        
        # 构建对话格式
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SFT1},
            {"role": "user", "content": f"{INPUT_PROMPT_SFT1}\n\n<table_content>\n{table_content}\n</table_content>"},
            {"role": "assistant", "content": target_json}
        ]
        
        samples.append({
            "id": f"sft1_{p.stem}",
            "task_type": "initial_generation",
            "conversations": messages,
            "source": "sft1"
        })
    
    print(f"   加载了 {len(samples)} 条初始生成数据")
    return samples


def load_strategy_data(data_path: Path) -> List[Dict]:
    """加载 Step8 策略选择数据"""
    print(f"📂 加载策略选择数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 转换为统一格式
    samples = []
    for item in data:
        # 构建对话格式 - 英文版提示词
        user_content = ""
        assistant_content = ""
        
        for conv in item.get('conversations', []):
            if conv.get('role') == 'user':
                user_content = conv.get('content', '')
            elif conv.get('role') == 'assistant':
                assistant_content = conv.get('content', '')
        
        # 转换为英文版 prompt (与训练数据一致)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        samples.append({
            "id": item.get('id', ''),
            "task_type": "strategy_selection",
            "conversations": messages,
            "source": "strategy"
        })
    
    print(f"   加载了 {len(samples)} 条策略选择数据")
    return samples


def mix_training_data(sft1_data: List[Dict], strategy_data: List[Dict], mix_ratio: float = 1.0) -> List[Dict]:
    """
    混合训练数据
    
    Args:
        sft1_data: 初始生成数据
        strategy_data: 策略选择数据
        mix_ratio: 混合比例，1.0 表示 1:1 混合
    
    Returns:
        混合后的训练数据
    """
    # 为了平衡，可以适当重复较少的数据类
    min_len = min(len(sft1_data), len(strategy_data))
    
    # 使用全部策略选择数据
    mixed = []
    
    # 策略选择数据全部使用
    mixed.extend(strategy_data)
    
    # 初始生成数据使用 mix_ratio 比例
    sft1_sample = sft1_data[:int(len(sft1_data) * mix_ratio)]
    mixed.extend(sft1_sample)
    
    # 随机打乱
    random.shuffle(mixed)
    
    print(f"\n📊 混合后数据统计:")
    print(f"   总数据量: {len(mixed)}")
    print(f"   初始生成: {sum(1 for d in mixed if d['task_type'] == 'initial_generation')}")
    print(f"   策略选择: {sum(1 for d in mixed if d['task_type'] == 'strategy_selection')}")
    
    return mixed


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize 数据，根据任务类型构建 labels"""
    full_texts = []
    labels_list = []
    
    for conv in examples['conversations']:
        text = ""
        for msg in conv:
            if msg['role'] == 'user':
                text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'assistant':
                text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
        full_texts.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    # 根据任务类型构建 labels
    # 两种任务都是只训练 assistant 部分
    for i, text in enumerate(full_texts):
        full_ids = tokenized["input_ids"][i]
        assistant_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        
        assistant_start = -1
        for j in range(len(full_ids) - len(assistant_token) + 1):
            if full_ids[j:j+len(assistant_token)] == assistant_token:
                assistant_start = j + len(assistant_token) + 1
                break
        
        if assistant_start == -1 or assistant_start >= len(full_ids):
            labels_list.append([-100] * len(full_ids))
            continue
        
        label = [-100] * assistant_start + full_ids[assistant_start:]
        
        if len(label) < len(full_ids):
            label = label + [-100] * (len(full_ids) - len(label))
        elif len(label) > len(full_ids):
            label = label[:len(full_ids)]
        
        labels_list.append(label)
    
    tokenized["labels"] = labels_list
    return tokenized


def prepare_dataset(training_data: List[Dict], tokenizer) -> Dataset:
    """准备数据集"""
    print("🔄 准备数据集...")
    
    dataset = Dataset.from_list(training_data)
    
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, MAX_SEQ_LENGTH),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ 数据集准备完成，共 {len(tokenized_dataset)} 个样本")
    return tokenized_dataset


def main():
    print("=" * 80)
    print("Step 8 增量学习 - LoRA + 混合训练")
    print("=" * 80)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 设置随机种子
    random.seed(42)
    torch.manual_seed(42)
    
    # 加载 tokenizer
    print(f"\n🔄 加载 tokenizer: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    print(f"🔄 加载模型: {BASE_MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 配置 LoRA
    print("\n🔧 配置 LoRA 微调...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        inference_mode=False
    )
    
    # 包装模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载并混合数据
    sft1_data = load_sft1_data(SFT1_DATA_DIR)
    strategy_data = load_strategy_data(STRATEGY_DATA_PATH)
    training_data = mix_training_data(sft1_data, strategy_data, mix_ratio=1.0)
    
    # 准备数据集
    tokenized_dataset = prepare_dataset(training_data, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # 开始训练
    print("\n🚀 开始增量学习训练...")
    print(f"   - 基座模型: {BASE_MODEL_PATH}")
    print(f"   - LoRA R: {LORA_R}")
    print(f"   - 学习率: {LEARNING_RATE}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - 样本数: {len(tokenized_dataset)}")
    
    trainer.train()
    
    # 保存模型
    print("\n💾 保存 LoRA 权重...")
    model.save_pretrained(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    
    # 保存训练配置
    config = {
        "base_model": BASE_MODEL_PATH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "num_samples": len(training_data),
        "sft1_samples": len(sft1_data),
        "strategy_samples": len(strategy_data),
    }
    
    with open(OUTPUT_DIR / "training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 增量学习完成!")
    print(f"模型保存在: {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()

