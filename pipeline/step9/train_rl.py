#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 9: 强化学习优化 (RL)

输入：step5/output/train_rl/ - 集合3的Markdown
输出：qwen3-7b-rl-final模型

方法：使用DPO (Direct Preference Optimization)
- 基于R_rule评分构建偏好对
- 不需要显式奖励模型，直接优化策略
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
import torch
import torch.nn.functional as F
from tqdm import tqdm

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 配置
SFT2_MODEL_PATH = project_root / "step8" / "output" / "sft2_model" / "final"
TRAIN_RL_DIR = project_root / "step5" / "output" / "train_rl"
OUTPUT_DIR = project_root / "step9" / "output" / "rl_model"
LOGS_DIR = project_root / "step9" / "training_logs"
PREFERENCE_DATA_DIR = project_root / "step9" / "preference_data"

# 训练超参数
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-6
NUM_EPOCHS = 1
MAX_SEQ_LENGTH = 2048
BETA = 0.1  # DPO温度参数

# 采样参数
NUM_SAMPLES_PER_TABLE = 4  # 每个表格生成的候选数
MIN_R_RULE_DIFF = 0.1  # 偏好对的最小R_rule差异


def calculate_r_rule_simple(chain: List[dict], teacher_chain: List[dict]) -> float:
    """
    简化版R_rule评分（用于快速评估）
    
    与Step 7的评分保持一致
    """
    if not chain or not teacher_chain:
        return 0.0
    
    score = 0.0
    
    # 1. 结构一致性 (25%)
    valid_chains = sum(
        1 for c in chain
        if all(k in c for k in ['condition', 'result', 'entity'])
        and c['condition'] and c['result'] and ':' in c['result']
    )
    if len(chain) > 0:
        score += 0.25 * (valid_chains / len(chain))
    
    # 2. 实体覆盖度 (25%)
    chain_entities = set()
    teacher_entities = set()
    
    for c in chain:
        if 'entity' in c and isinstance(c['entity'], dict):
            for entity_list in c['entity'].values():
                if isinstance(entity_list, list):
                    chain_entities.update(entity_list)
    
    for c in teacher_chain:
        if 'entity' in c and isinstance(c['entity'], dict):
            for entity_list in c['entity'].values():
                if isinstance(entity_list, list):
                    teacher_entities.update(entity_list)
    
    if teacher_entities:
        intersection = len(chain_entities & teacher_entities)
        union = len(chain_entities | teacher_entities)
        if union > 0:
            score += 0.25 * (intersection / union)
    
    # 3. 条件-结果匹配度 (30%)
    valid_results = sum(
        1 for c in chain
        if c.get('result') and ':' in c['result']
    )
    if len(chain) > 0:
        score += 0.30 * (valid_results / len(chain))
    
    # 4. 链数量接近度 (20%)
    if len(teacher_chain) > 0:
        ratio = min(len(chain), len(teacher_chain)) / max(len(chain), len(teacher_chain))
        score += 0.20 * ratio
    
    return min(1.0, score)


def generate_chain_from_model(
    tokenizer,
    model,
    caption: str,
    table_content: str,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Tuple[str, List[dict]]:
    """使用模型生成知识链"""
    
    input_text = f"""## Caption
{caption}

## Table Content
{table_content}

请抽取知识链（JSON格式）:"""
    
    prompt = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # 解析JSON
    try:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        chains = json.loads(cleaned)
        
        if isinstance(chains, list):
            return response, chains
    except:
        pass
    
    return response, []


def extract_caption_and_table(markdown_content: str) -> Tuple[str, str]:
    """从Markdown内容中提取Caption和Table Content"""
    lines = markdown_content.split('\n')
    caption = ""
    table_content = ""
    
    in_caption = False
    in_table = False
    
    for line in lines:
        if line.strip() == "## Caption":
            in_caption = True
            in_table = False
        elif line.strip() == "## Table Content":
            in_caption = False
            in_table = True
        elif line.startswith("##"):
            in_caption = False
            in_table = False
        elif in_caption:
            caption += line + "\n"
        elif in_table:
            table_content += line + "\n"
    
    return caption.strip(), table_content.strip()


def generate_preference_pairs(
    tokenizer,
    model,
    table_id: str,
    markdown_path: Path,
    teacher_chain: List[dict],
    num_samples: int = 4
) -> List[Dict]:
    """
    为单个表格生成偏好对
    
    策略：
    1. 用不同的温度生成多个候选
    2. 计算每个候选的R_rule分数
    3. 选择分数差异明显的作为偏好对
    """
    
    # 读取markdown
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    caption, table_content = extract_caption_and_table(markdown_content)
    
    # 生成多个候选（使用不同温度）
    candidates = []
    temperatures = [0.3, 0.5, 0.7, 0.9]
    
    for temp in temperatures:
        response, chains = generate_chain_from_model(
            tokenizer, model, caption, table_content,
            temperature=temp, top_p=0.9
        )
        
        if chains:
            r_rule = calculate_r_rule_simple(chains, teacher_chain)
            candidates.append({
                "response": response,
                "chains": chains,
                "r_rule": r_rule,
                "temperature": temp
            })
    
    # 排序候选
    candidates.sort(key=lambda x: x['r_rule'], reverse=True)
    
    # 构建偏好对
    preference_pairs = []
    
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            chosen = candidates[i]
            rejected = candidates[j]
            
            # 只保留R_rule差异明显的对
            if chosen['r_rule'] - rejected['r_rule'] >= MIN_R_RULE_DIFF:
                preference_pairs.append({
                    "table_id": table_id,
                    "prompt": f"""## Caption
{caption}

## Table Content
{table_content}

请抽取知识链（JSON格式）:""",
                    "chosen": chosen['response'],
                    "rejected": rejected['response'],
                    "chosen_r_rule": chosen['r_rule'],
                    "rejected_r_rule": rejected['r_rule'],
                    "score_diff": chosen['r_rule'] - rejected['r_rule']
                })
    
    return preference_pairs


class DPOTrainer(Trainer):
    """
    DPO (Direct Preference Optimization) Trainer
    
    实现简化版的DPO算法
    """
    
    def __init__(self, *args, beta=0.1, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.ref_model = ref_model
        
        # 冻结参考模型
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算DPO损失
        
        DPO损失公式:
        L = -log(σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
        
        其中:
        - y_w: chosen response
        - y_l: rejected response
        - π_θ: 当前策略
        - π_ref: 参考策略
        - β: 温度参数
        """
        
        # 获取输入
        chosen_input_ids = inputs['chosen_input_ids']
        chosen_attention_mask = inputs['chosen_attention_mask']
        chosen_labels = inputs['chosen_labels']
        
        rejected_input_ids = inputs['rejected_input_ids']
        rejected_attention_mask = inputs['rejected_attention_mask']
        rejected_labels = inputs['rejected_labels']
        
        # 计算chosen的log概率
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask,
            labels=chosen_labels
        )
        chosen_logps = -chosen_outputs.loss  # 负损失即为log概率
        
        # 计算rejected的log概率
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask,
            labels=rejected_labels
        )
        rejected_logps = -rejected_outputs.loss
        
        # 如果有参考模型，计算参考log概率
        if self.ref_model is not None:
            with torch.no_grad():
                ref_chosen_outputs = self.ref_model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                    labels=chosen_labels
                )
                ref_chosen_logps = -ref_chosen_outputs.loss
                
                ref_rejected_outputs = self.ref_model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                    labels=rejected_labels
                )
                ref_rejected_logps = -ref_rejected_outputs.loss
        else:
            ref_chosen_logps = 0.0
            ref_rejected_logps = 0.0
        
        # 计算DPO损失
        logits = self.beta * (
            chosen_logps - rejected_logps -
            ref_chosen_logps + ref_rejected_logps
        )
        
        loss = -F.logsigmoid(logits).mean()
        
        return (loss, chosen_outputs) if return_outputs else loss


def prepare_dpo_dataset(preference_pairs: List[Dict], tokenizer) -> Dataset:
    """准备DPO训练数据集"""
    
    print(f"🔄 准备DPO数据集...")
    
    def tokenize_pair(example):
        """Tokenize偏好对"""
        
        # Tokenize prompt + chosen
        chosen_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['chosen']}<|im_end|>"
        chosen_tokens = tokenizer(
            chosen_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length'
        )
        
        # Tokenize prompt + rejected
        rejected_text = f"<|im_start|>user\n{example['prompt']}<|im_end|>\n<|im_start|>assistant\n{example['rejected']}<|im_end|>"
        rejected_tokens = tokenizer(
            rejected_text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding='max_length'
        )
        
        return {
            'chosen_input_ids': chosen_tokens['input_ids'],
            'chosen_attention_mask': chosen_tokens['attention_mask'],
            'chosen_labels': chosen_tokens['input_ids'],  # 标签与input_ids相同
            'rejected_input_ids': rejected_tokens['input_ids'],
            'rejected_attention_mask': rejected_tokens['attention_mask'],
            'rejected_labels': rejected_tokens['input_ids']
        }
    
    # 转换为Dataset
    dataset = Dataset.from_dict({
        "prompt": [p["prompt"] for p in preference_pairs],
        "chosen": [p["chosen"] for p in preference_pairs],
        "rejected": [p["rejected"] for p in preference_pairs]
    })
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_pair,
        remove_columns=dataset.column_names
    )
    
    print(f"✅ DPO数据集准备完成，共 {len(tokenized_dataset)} 个偏好对")
    
    return tokenized_dataset


def main():
    """主函数"""
    print("=" * 80)
    print("Step 9: 强化学习优化 (RL) - DPO实现")
    print("=" * 80)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PREFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载SFT-2模型
    print(f"\n🔄 加载SFT-2模型: {SFT2_MODEL_PATH}")
    
    if not SFT2_MODEL_PATH.exists():
        print(f"❌ 错误: {SFT2_MODEL_PATH} 不存在")
        print("请先运行 Step 8 训练 SFT-2 模型")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(SFT2_MODEL_PATH, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载策略模型
    policy_model = AutoModelForCausalLM.from_pretrained(
        SFT2_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # 加载参考模型（用于DPO）
    ref_model = AutoModelForCausalLM.from_pretrained(
        SFT2_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("✅ 模型加载完成")
    
    # 生成偏好对
    print("\n🎲 生成偏好对...")
    
    teacher_chains_path = TRAIN_RL_DIR / "teacher_chains.json"
    if not teacher_chains_path.exists():
        print(f"❌ 错误: {teacher_chains_path} 不存在")
        print("请先运行 Step 5 生成数据集划分")
        return
    
    with open(teacher_chains_path, 'r', encoding='utf-8') as f:
        teacher_chains = json.load(f)
    
    markdown_dir = TRAIN_RL_DIR / "markdown_tables"
    
    all_preference_pairs = []
    
    print(f"📊 共 {len(teacher_chains)} 个表格")
    
    for table_id, data in tqdm(teacher_chains.items(), desc="生成偏好对"):
        markdown_path = markdown_dir / f"{table_id}.md"
        
        if not markdown_path.exists():
            continue
        
        pairs = generate_preference_pairs(
            tokenizer,
            policy_model,
            table_id,
            markdown_path,
            data['chains'],
            num_samples=NUM_SAMPLES_PER_TABLE
        )
        
        all_preference_pairs.extend(pairs)
    
    print(f"✅ 生成了 {len(all_preference_pairs)} 个偏好对")
    
    # 保存偏好对
    with open(PREFERENCE_DATA_DIR / "preference_pairs.json", 'w', encoding='utf-8') as f:
        json.dump(all_preference_pairs, f, ensure_ascii=False, indent=2)
    
    if len(all_preference_pairs) == 0:
        print("❌ 没有生成有效的偏好对")
        return
    
    # 准备数据集
    dpo_dataset = prepare_dpo_dataset(all_preference_pairs, tokenizer)
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=50,
        logging_dir=str(LOGS_DIR),
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        remove_unused_columns=False  # 重要：保留自定义列
    )
    
    # 创建DPO Trainer
    trainer = DPOTrainer(
        model=policy_model,
        args=training_args,
        train_dataset=dpo_dataset,
        beta=BETA,
        ref_model=ref_model
    )
    
    # 开始训练
    print("\n🚀 开始DPO训练...")
    print(f"  - 偏好对数量: {len(all_preference_pairs)}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Beta (DPO temperature): {BETA}")
    
    trainer.train()
    
    # 保存最终模型
    print("\n💾 保存最终模型...")
    trainer.save_model(str(OUTPUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "final"))
    
    # 保存训练配置
    config = {
        "method": "DPO (Direct Preference Optimization)",
        "base_model": str(SFT2_MODEL_PATH),
        "num_preference_pairs": len(all_preference_pairs),
        "num_samples_per_table": NUM_SAMPLES_PER_TABLE,
        "min_r_rule_diff": MIN_R_RULE_DIFF,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "max_seq_length": MAX_SEQ_LENGTH,
        "beta": BETA
    }
    
    with open(OUTPUT_DIR / "training_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("\n✅ Step 9 完成!")
    print(f"模型保存在: {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
