"""
SFT 第一阶段：用 SFT1 数据集微调 Qwen3-8B（本地已下载）

输入：table_content
输出：knowledge_mining（含知识链 + 实体识别的 JSON）

用法示例（已假设 conda activate yjw）:
python train_sft1.py \
  --data_dir /home/weijie_yu/yjw/ECLO-RL/step6/SFT1 \
  --model_name_or_path /home/new_disk/yjw/Model/Qwen3-8B \
  --output_dir /home/new_disk/yjw/checkpoints/qwen3_sft1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --max_length 30000
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig,
)

# 优化 PyTorch 内存分配，减少碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


SYSTEM_PROMPT = (
    "你是一名生物医学知识抽取助手，任务是从表格内容(table_content)中生成知识挖掘结果。"
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


def load_jsonl_dir(data_dir: Path) -> List[Dict]:
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
        samples.append(
            {
                "table_content": table_content,
                "target": target_json,
                "file": p.name,
            }
        )
    return samples


def build_messages(table_content: str, target_json: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{INPUT_PROMPT}\n\n<table_content>\n{table_content}\n</table_content>",
        },
        {"role": "assistant", "content": target_json},
    ]


def tokenize_sample(tokenizer, sample, max_length: int):
    messages = build_messages(sample["table_content"], sample["target"])
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    encoded = tokenizer(
        prompt_text,
        max_length=max_length,
            truncation=True,
        padding=False,
        return_tensors=None,
    )

    input_ids = encoded["input_ids"]
    labels = input_ids.copy()

    # 仅训练助手输出：将 assistant 段之前的 tokens 屏蔽为 -100
    # 通过重新生成到助手前的模板，得到前缀长度
    prefix_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=False
    )
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    assist_start = len(prefix_ids)
    labels[:assist_start] = [-100] * assist_start

    encoded["labels"] = labels
    return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/new_disk/yjw/Model/Qwen3-8B",
        help="本地已下载的基座模型路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/new_disk/yjw/checkpoints/qwen3_sft1_new",  # 将被脚本覆盖
    )
    parser.add_argument("--max_length", type=int, default=30000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="可选：DeepSpeed 配置 JSON 路径（例如 zero-2），不需要则留空",
    )
    # 兼容 deepspeed 启动的 local_rank 参数
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    samples = load_jsonl_dir(data_dir)
    if not samples:
        raise ValueError("数据为空，请检查 data_dir")

    random.shuffle(samples)
    eval_size = max(1, int(len(samples) * args.eval_ratio))
    eval_samples = samples[:eval_size]
    train_samples = samples[eval_size:]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    def _tokenize(batch):
        return tokenize_sample(tokenizer, batch, args.max_length)

    train_ds = Dataset.from_list(train_samples).map(
        _tokenize, remove_columns=["table_content", "target", "file"]
    )
    eval_ds = Dataset.from_list(eval_samples).map(
        _tokenize, remove_columns=["table_content", "target", "file"]
    )

    # 优化内存使用：启用低 CPU 内存模式，限制单个 GPU 的最大内存
    max_memory = {0: "70GB"}  # 单卡预留更多内存缓冲
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
    )
    
    # 添加generation_config确保推理一致性
    model.generation_config = GenerationConfig(
        do_sample=False,  # 贪婪解码，不使用temperature
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=2048,
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args.max_length,
    )
    
    # 修正warmup_steps计算（基于实际训练步数）
    num_gpus = 1  # 单卡训练
    actual_train_steps = len(train_ds) // (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * num_gpus
    ) * args.num_train_epochs
    warmup_steps = max(1, int(actual_train_steps * args.warmup_ratio))
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_first_step=True,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,  # 关键修正：保留labels字段
        report_to="none",
        seed=args.seed,
        deepspeed=args.deepspeed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

