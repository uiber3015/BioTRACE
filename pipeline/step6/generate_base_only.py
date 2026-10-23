#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立运行基座模型生成 results_base.json

可以与 complete_and_generate_base.py 并行运行在不同的 GPU 上。

Usage:
CUDA_VISIBLE_DEVICES=1 python step6/generate_base_only.py \
  --base_model_path /home/new_disk/yjw/Model/Qwen3-8B \
  --data_dir step6/Test \
  --results_dir step6/test_results \
  --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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
4) 覆盖所有数值字段（含单位/百分号/科学计数），ID/编号优先放 condition.

实体识别规则（对齐 01_entity_type）：
1) 实体值必须在 condition 或 result 原文中**完整匹配出现**，不得拆分、不得补全。
2) 不拆分 items.text（逗号/分号等仍视为整体）；若含并列混杂则不标。
3) 仅允许 14 类实体：
   gene_symbol, regulator, target, functional_mechanism, species, disease,
   organ, tissue, cell, protein, compound, drug, experimental_method, phenotype
4) 同类去重，未识别则用空对象 {}，不得幻觉新增或虚构实体。

请直接输出合法 JSON，不要额外文本。"""


def build_messages(table_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{INPUT_PROMPT}\n\n<table_content>\n{table_content}\n</table_content>",
        },
    ]


def extract_json_from_text(text: str) -> Optional[Dict]:
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    return None


def load_model_single_gpu(model_path: str, device: str):
    """单卡加载模型"""
    print(f"Loading tokenizer from {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} to {device} ...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return model, tokenizer


def generate_with_model(model, tokenizer, table_content: str, max_new_tokens: int = 8192) -> str:
    messages = build_messages(table_content)
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = SYSTEM_PROMPT + "\n\n" + messages[1]["content"]

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=30000)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def generate_base_results(model, tokenizer, data_dir: Path, output_file: Path, trained_results_file: Path, batch_size: int = 20):
    """生成基座模型的结果，每 batch_size 个样本保存一次"""
    print(f"\n=== 生成基座模型结果 -> {output_file} ===")
    
    # 读取 trained 结果以获取相同的样本列表
    with open(trained_results_file, "r", encoding="utf-8") as f:
        trained_results = json.load(f)
    
    # 构建 file_id -> data_file 映射
    json_files = {p.stem: p for p in data_dir.glob("*.json")}
    
    # 检查是否有已存在的结果（断点续传）
    results = []
    start_idx = 0
    if output_file.exists():
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            start_idx = len(results)
            print(f"发现已有结果 {start_idx} 个，从第 {start_idx + 1} 个开始继续")
        except:
            pass
    
    total = len(trained_results)
    
    for idx in range(start_idx, total):
        trained_sample = trained_results[idx]
        file_id = trained_sample.get('file_id', 'unknown')
        ground_truth = trained_sample.get('ground_truth', {})
        
        # 从原始数据文件读取 table_content
        data_file = json_files.get(str(file_id))
        if data_file is None:
            results.append({"file_id": file_id, "generated_json": None, "ground_truth": ground_truth})
            print(f"[{idx+1}/{total}] {file_id}: 找不到原始数据文件")
            continue
        
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        table_content = data.get("table_content", "").strip()
        if not table_content:
            results.append({"file_id": file_id, "generated_json": None, "ground_truth": ground_truth})
            continue
        
        try:
            gen_text = generate_with_model(model, tokenizer, table_content)
            gen_json = extract_json_from_text(gen_text)
        except Exception as e:
            gen_text = f"ERROR: {e}"
            gen_json = None
        
        results.append({
            "file_id": file_id,
            "generated_json": gen_json,
            "generated_text": gen_text,
            "ground_truth": ground_truth
        })
        
        status = "OK" if gen_json else ("ERROR" if gen_text.startswith("ERROR:") else "PARSE_FAIL")
        preview = gen_text.replace("\n", " ")[:150]
        print(f"[{datetime.now().isoformat()}] [{idx+1}/{total}] {file_id} {status} preview={preview}")
        
        sys.stdout.flush()
        
        # 每 batch_size 个保存一次
        if (idx + 1) % batch_size == 0:
            with open(output_file, "w", encoding="utf-8") as wf:
                json.dump(results, wf, ensure_ascii=False, indent=2)
            print(f"[SAVE] 已保存进度: {idx+1}/{total}")
    
    # 最终保存
    with open(output_file, "w", encoding="utf-8") as wf:
        json.dump(results, wf, ensure_ascii=False, indent=2)
    
    print(f"\n基座模型生成完成: {output_file}")
    print(f"共 {len(results)} 个样本")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, 
                        default="/home/new_disk/yjw/Model/Qwen3-8B",
                        help="基座模型路径")
    parser.add_argument("--data_dir", type=str, default="step6/Test",
                        help="测试数据目录")
    parser.add_argument("--results_dir", type=str, default="step6/test_results",
                        help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="使用的 GPU 设备，如 cuda:0")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="每多少个样本保存一次")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    trained_results_file = results_dir / "results_trained.json"
    base_results_file = results_dir / "results_base.json"

    if not trained_results_file.exists():
        print(f"ERROR: {trained_results_file} 不存在，请先运行 complete_and_generate_base.py 或确保文件存在")
        sys.exit(1)

    print("="*60)
    print("基座模型独立生成脚本")
    print(f"  模型: {args.base_model_path}")
    print(f"  设备: {args.device}")
    print(f"  输出: {base_results_file}")
    print("="*60)
    
    base_model, base_tokenizer = load_model_single_gpu(args.base_model_path, args.device)
    generate_base_results(base_model, base_tokenizer, data_dir, base_results_file, trained_results_file, args.batch_size)
    
    del base_model
    del base_tokenizer
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("基座模型生成完成！")
    print("="*60)


if __name__ == "__main__":
    main()
