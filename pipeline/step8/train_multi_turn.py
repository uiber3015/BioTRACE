#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 8 增量学习训练脚本 - 多轮修正训练

目标：训练模型能够进行多轮修正，而不是只输出单轮策略

训练数据：
1. Step6 SFT1 数据 (初始生成能力) - 保留原有能力
2. Step7 多轮修正完整轨迹数据 - 学习多轮修正

核心思路：
- 训练模型自主决定何时停止（输出 STOP）
- 学会在每一轮修正后，根据修正结果决定是否继续
- 使用 Step6 和 Step8 原始提示词格式
- 统一消息格式：user + assistant（去掉 system）
- 添加任务前缀区分两种任务
- 增强 STOP 信号
"""

import os
import json
import sys
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import Dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# 配置
PROJECT_ROOT = Path(__file__).parent.parent

# 基座模型 (Step6 训练后的模型)
BASE_MODEL_PATH = "/home/new_disk/yjw/checkpoints/qwen3_sft1_20260225_183025"

# 数据路径
SFT1_DATA_DIR = PROJECT_ROOT / "step6" / "SFT1"
MULTI_TURN_DATA_PATH = PROJECT_ROOT / "step8" / "training_data" / "sft_training_data_full_trajectory.json"

# 输出路径
OUTPUT_DIR = Path("/home/new_disk/yjw/checkpoints/step8_multi_turn_lora")
LOGS_DIR = PROJECT_ROOT / "step8" / "logs"

# 训练超参数
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5  # 降低学习率，减少灾难性遗忘
NUM_EPOCHS = 1  # 减少训练轮数
MAX_SEQ_LENGTH = 8192

# LoRA 配置 - 使用更保守的LoRA配置
LORA_R = 16  # 从64降到16，更保守的微调
LORA_ALPHA = 32  # 从128降到32
LORA_DROPOUT = 0.1  # 从0.05提高到0.1，增加正则化

# STOP 信号增强倍数
STOP_OVERSAMPLE_RATIO = 5  # 从3提高到5，增强停止信号

# LoRA 目标模块
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]


# ============== Step6 原始提示词 ==============
# 注意：必须与Step6原始训练格式完全一致，一个字符都不能改！
# 关键：使用SYSTEM_PROMPT，不添加任务前缀
SYSTEM_PROMPT_STEP6 = (
    "你是一名生物医学知识抽取助手，任务是从表格内容(table_content)中生成知识挖掘结果。"
    "输出必须是 JSON，键名为 knowledge_mining，对应一个列表，每个元素包含："
    "knowledge_chain.condition、knowledge_chain.result、entity_recognition。"
    "仅输出 JSON，不要额外解释。"
)

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


# ============== Step8 原始提示词 (策略选择) ==============
# 使用完整模板，与原始训练数据保持一致
INPUT_PROMPT_STEP8 = """## 任务
你是一个知识链修正助手。根据以下表格内容和当前知识链，选择修正策略进行迭代修正，直到知识链正确无误。

## 表格内容
{table_content}

## 当前知识链
```json
{current_knowledge_chain}
```

{previous_summary_block}

## 修正策略说明

| 策略 | 说明 |
|------|------|
| STOP | 知识链已正确，无需修正 |
| Modify/condition_kv | 修改指定链的 condition 字段 |
| Modify/result | 修改指定链的 result 字段 |
| Modify/entity_type | 修改实体类型标签 |
| Modify/entity_value | 修改实体值拼写 |
| Modify/entity_recognition | 修改实体识别结果 |
| Add/chain | 添加新知识链 |
| Add/entity_type | 添加实体类型标签 |
| Add/entity_value | 添加实体值 |
| Delete/chain | 删除指定知识链 |
| Delete/condition_kv | 删除 condition 字段 |
| Delete/entity_type | 删除实体类型标签 |
| Delete/entity_value | 删除实体值 |

请输出 JSON 格式的修正指令，包含以下字段：
- action: "STOP" 或 "CORRECT"
- strategy: 修正策略（如 "Modify/condition_kv"）
- chain_indices: 需要修正的知识链索引列表（从0开始）

直接输出 JSON，不要其他内容。"""


# ============== 任务前缀 ==============
# 去掉任务前缀，保持与Step6原始训练格式完全一致！
# 原始能力必须保留，所以不能添加任何额外的前缀或标签


def load_sft1_data(data_dir: Path) -> List[Dict]:
    """
    加载 Step6 SFT1 初始生成数据

    关键：必须与Step6原始训练格式完全一致！
    - 使用SYSTEM_PROMPT
    - 不添加任何任务前缀
    - 使用<table_content>标签
    """
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

        # 关键：与Step6原始训练格式完全一致！
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STEP6},
            {"role": "user", "content": f"{INPUT_PROMPT_STEP6}\n\n<table_content>\n{table_content}\n</table_content>"},
            {"role": "assistant", "content": target_json}
        ]

        samples.append({
            "id": f"sft1_{p.stem}",
            "task_type": "initial_generation",
            "conversations": messages,
        })

    print(f"   加载了 {len(samples)} 条初始生成数据")
    return samples


def load_multi_turn_data(data_path: Path, stop_oversample_ratio: int = 3) -> List[Dict]:
    """
    加载 Step7 多轮修正完整轨迹数据
    
    改进：
    - 去掉 system 消息
    - 添加任务前缀
    - 增强 STOP 信号（复制更多 STOP 样本）
    """
    print(f"📂 加载多轮修正轨迹数据: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    stop_samples = []  # 收集 STOP 样本
    
    for item in data:
        # 解析原始轨迹数据
        user_content = item['conversations'][0]['content']
        assistant_content = item['conversations'][1]['content']
        
        # 提取表格内容
        lines = user_content.split('\n')
        in_table = False
        table_lines = []
        for line in lines:
            if '## 表格内容' in line:
                in_table = True
                continue
            if in_table and '## 当前知识链' in line:
                break
            if in_table and line.strip():
                table_lines.append(line)
        
        table_content = '\n'.join(table_lines)
        
        # 提取知识链
        import re
        kc_match = re.search(r'```json\n(.*?)\n```', user_content, re.DOTALL)
        if kc_match:
            initial_kc = json.loads(kc_match.group(1))
        else:
            kc_match = re.search(r'## 当前知识链\s*\n(.*?)$', user_content, re.DOTALL)
            if kc_match:
                kc_text = kc_match.group(1)
                json_match = re.search(r'\{.*\}', kc_text, re.DOTALL)
                if json_match:
                    initial_kc = json.loads(json_match.group(0))
                else:
                    continue
            else:
                continue
        
        if not table_content or not initial_kc:
            continue

        # 去掉任务前缀，使用Step8原始提示词格式
        prompt = INPUT_PROMPT_STEP8.format(
            table_content=table_content,
            current_knowledge_chain=json.dumps(initial_kc, ensure_ascii=False),
            previous_summary_block=""
        )
        
        sample = {
            "id": f"mt_{item.get('id', '')}",
            "task_type": "multi_turn_correction",
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": assistant_content}
            ],
        }
        
        samples.append(sample)
        
        # 检查是否是 STOP 样本
        try:
            rounds = json.loads(assistant_content)
            if rounds and rounds[-1].get('action') == 'STOP':
                stop_samples.append(sample)
        except:
            pass
    
    # 增强 STOP 信号：复制 STOP 样本
    stop_count_before = len(stop_samples)
    for _ in range(stop_oversample_ratio - 1):
        samples.extend(stop_samples)
    
    # 统计轮次分布
    rounds_dist = {}
    for s in samples[:len(samples) // stop_oversample_ratio]:  # 只统计原始数据
        try:
            rounds = len(json.loads(s["conversations"][1]["content"]))
            rounds_dist[rounds] = rounds_dist.get(rounds, 0) + 1
        except:
            pass
    
    print(f"   加载了 {len(samples)} 条多轮修正数据 (增强 STOP 后)")
    print(f"   原始 STOP 样本: {stop_count_before}")
    print(f"   增强后 STOP 样本: {len(stop_samples) * stop_oversample_ratio}")
    print(f"   轮次分布: {rounds_dist}")
    
    return samples


def mix_training_data(sft1_data: List[Dict], multi_turn_data: List[Dict], mix_ratio: float = 0.8) -> List[Dict]:
    """
    混合训练数据

    关键：提高SFT1数据比例，减少灾难性遗忘
    - 使用80%的SFT1数据，确保原始能力得到充分复习
    - 多轮修正数据全部使用
    """
    mixed = []

    # 多轮修正数据全部使用（重点学习多轮修正）
    mixed.extend(multi_turn_data)

    # 初始生成数据使用 mix_ratio 比例（提高到80%）
    sft1_sample = sft1_data[:int(len(sft1_data) * mix_ratio)]
    mixed.extend(sft1_sample)

    # 随机打乱
    random.shuffle(mixed)

    # 统计
    sft1_count = len(sft1_sample)
    mt_count = len(multi_turn_data)
    total = len(mixed)

    print(f"\n📊 混合后数据统计:")
    print(f"   总数据量: {total}")
    print(f"   初始生成: {sft1_count} ({sft1_count/total*100:.1f}%)")
    print(f"   多轮修正: {mt_count} ({mt_count/total*100:.1f}%)")

    return mixed


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize 数据"""
    full_texts = []
    labels_list = []
    
    for conv in examples['conversations']:
        text = ""
        for msg in conv:
            if msg['role'] == 'system':
                text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
            elif msg['role'] == 'user':
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
    
    # 只训练 assistant 部分
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Step8 多轮修正增量学习训练")
    
    # 模型路径
    parser.add_argument("--base_model_path", type=str, default=BASE_MODEL_PATH,
                        help="基座模型路径")
    
    # 数据路径
    parser.add_argument("--sft1_data_dir", type=str, default=str(SFT1_DATA_DIR),
                        help="SFT1 数据目录")
    parser.add_argument("--multi_turn_data", type=str, default=str(MULTI_TURN_DATA_PATH),
                        help="多轮修正数据路径")
    
    # 输出路径
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="输出目录")
    
    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="每设备 batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="学习率")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS,
                        help="训练轮数")
    parser.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH,
                        help="最大序列长度")
    parser.add_argument("--mix_ratio", type=float, default=0.8,
                        help="SFT1 数据混合比例 (0.0-1.0)，建议0.8保留原始能力")
    parser.add_argument("--stop_oversample_ratio", type=int, default=STOP_OVERSAMPLE_RATIO,
                        help="STOP 信号过采样倍数")
    
    # LoRA 配置
    parser.add_argument("--lora_r", type=int, default=LORA_R,
                        help="LoRA r")
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT,
                        help="LoRA dropout")
    
    return parser.parse_args()


def main():
    # 解析参数
    args = parse_args()
    
    # 使用命令行参数覆盖默认值
    global BASE_MODEL_PATH, SFT1_DATA_DIR, MULTI_TURN_DATA_PATH
    global OUTPUT_DIR, LOGS_DIR, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
    global LEARNING_RATE, NUM_EPOCHS, MAX_SEQ_LENGTH
    global LORA_R, LORA_ALPHA, LORA_DROPOUT
    
    BASE_MODEL_PATH = args.base_model_path
    SFT1_DATA_DIR = Path(args.sft1_data_dir)
    MULTI_TURN_DATA_PATH = Path(args.multi_turn_data)
    OUTPUT_DIR = Path(args.output_dir)
    LOGS_DIR = PROJECT_ROOT / "step8" / "logs"
    BATCH_SIZE = args.batch_size
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    MAX_SEQ_LENGTH = args.max_seq_length
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    
    print("=" * 80)
    print("Step 8 增量学习 - 多轮修正训练 (LoRA)")
    print("=" * 80)
    print("改进点（防遗忘策略）：")
    print("  1. 保留SYSTEM_PROMPT：与Step6原始训练格式完全一致")
    print("  2. 去掉任务前缀：避免格式不一致导致遗忘")
    print("  3. 降低学习率：3e-4 -> 5e-5，减少灾难性遗忘")
    print("  4. 保守LoRA配置：R=64->16, Alpha=128->32")
    print("  5. 提高SFT1比例：50% -> 80%，加强原始能力复习")
    print("  6. 增强STOP信号：3x -> 5x")
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
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 加载数据
    sft1_data = load_sft1_data(SFT1_DATA_DIR)
    multi_turn_data = load_multi_turn_data(MULTI_TURN_DATA_PATH, args.stop_oversample_ratio)
    training_data = mix_training_data(sft1_data, multi_turn_data, args.mix_ratio)
    
    # 准备数据集
    tokenized_dataset = prepare_dataset(training_data, tokenizer)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # 添加梯度累积
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        save_steps=1000,  # 每1000步保存一次（约一半steps）
        save_total_limit=2,  # 最多保留2个checkpoint
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("\n🚀 开始多轮修正训练...")
    print(f"   - 基座模型: {BASE_MODEL_PATH}")
    print(f"   - LoRA R: {LORA_R}")
    print(f"   - 学习率: {LEARNING_RATE}")
    print(f"   - Epochs: {NUM_EPOCHS}")
    print(f"   - 样本数: {len(tokenized_dataset)}")
    print(f"   - SFT1 混合比例: {args.mix_ratio}")
    print(f"   - STOP 过采样: {args.stop_oversample_ratio}x")
    
    trainer.train()
    
    # 保存模型
    print("\n💾 保存 LoRA 权重...")
    model.save_pretrained(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    
    # 保存配置
    config = {
        "base_model": BASE_MODEL_PATH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "num_samples": len(training_data),
        "sft1_samples": len(sft1_data),
        "multi_turn_samples": len(multi_turn_data),
        "mix_ratio": args.mix_ratio,
        "stop_oversample_ratio": args.stop_oversample_ratio,
    }
    
    with open(OUTPUT_DIR / "training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("\n✅ 多轮修正训练完成!")
    print(f"模型保存在: {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
