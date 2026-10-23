#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run evaluation for a trained SFT-1 model and a base model on a test dataset,
save per-sample generated results and run the simplified evaluator to produce summaries.

Usage example:
python step6/run_evaluation_all.py \
  --trained_model_path /home/new_disk/yjw/checkpoints/qwen3_sft1_openr1_20260124_135733 \
  --base_model_path /home/new_disk/yjw/Model/Qwen3-8B \
  --data_dir step6/Test \
  --results_dir step6/test_results \
  --num_samples 500 \
  --device cuda

This script reuses the same prompts as evaluate_model.py.
"""

import argparse
import json
import random
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime


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


def generate_with_model(model, tokenizer, table_content: str, max_new_tokens: int = 2048) -> str:
    messages = build_messages(table_content)
    # apply_chat_template is expected on this project's tokenizer
    try:
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # fallback: concatenate system + user content
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


def run_generation_for_model(model_path: str, data_dir: str, out_path: str, num_samples: int, device: str):
    model_path = str(model_path)
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {model_path} ...")
    # Some tokenizers (Mistral variants) require fix_mistral_regex to avoid incorrect tokenization
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        # older transformers may not support fix_mistral_regex
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} ...")
    # try to load with reasonable device_map
    try:
        # use `dtype` instead of deprecated `torch_dtype`
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto" if "cuda" in device and torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Model load warning: {e}\nRetrying with device_map=None...")
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    model.eval()

    json_files = sorted([p for p in data_dir.glob("*.json")], key=lambda x: int(x.stem))
    if len(json_files) == 0:
        raise RuntimeError(f"No JSON files found in {data_dir}")

    samples = random.sample(json_files, min(num_samples, len(json_files)))

    results = []
    for idx, jf in enumerate(samples, 1):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        table_content = data.get("table_content", "").strip()
        ground_truth = {"knowledge_mining": data.get("knowledge_mining", [])}
        file_id = data.get("file_id", jf.stem)

        if not table_content:
            results.append({"file_id": file_id, "generated_json": None, "ground_truth": ground_truth})
            continue

        try:
            gen_text = generate_with_model(model, tokenizer, table_content)
            gen_json = extract_json_from_text(gen_text)
        except Exception as e:
            gen_text = f"ERROR: {e}"
            gen_json = None

        results.append({"file_id": file_id, "generated_json": gen_json, "generated_text": gen_text, "ground_truth": ground_truth})

        # determine status
        if gen_json:
            status = "OK"
        elif gen_text and gen_text.startswith("ERROR:"):
            status = "ERROR"
        elif gen_text and gen_text.strip().startswith("{"):
            status = "JSON_PARSE_FAIL"
        else:
            status = "OK_TEXT"

        preview = gen_text.replace("\n", " ")[:400].replace("\r", " ")
        # Print detailed per-sample log to stdout (visible in nohup)
        print(f"[{datetime.now().isoformat()}] {Path(model_path).name} {idx}/{len(samples)} {file_id} {status} preview={preview}")
        # flush to ensure nohup captures it promptly
        import sys
        sys.stdout.flush()

        if idx % 50 == 0 or idx == len(samples):
            print(f"Generated {idx}/{len(samples)}")

    with open(out_path, "w", encoding="utf-8") as wf:
        json.dump(results, wf, ensure_ascii=False, indent=2)

    print(f"Saved generation results to {out_path}")
    return out_path


def run_simplified_evaluation(results_file: str, output_file: str, num_samples: int):
    cmd = [
        "python",
        "step6/evaluate_simplified.py",
        "--results_file",
        str(results_file),
        "--output_file",
        str(output_file),
        "--num_samples",
        str(num_samples),
    ]
    print("Running simplified evaluator:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Evaluator output saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="step6/Test")
    parser.add_argument("--results_dir", type=str, default="step6/test_results")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run for trained model
    trained_results = results_dir / "results_trained.json"
    print("\n=== Generating with trained model ===")
    run_generation_for_model(args.trained_model_path, args.data_dir, trained_results, args.num_samples, args.device)
    run_simplified_evaluation(trained_results, results_dir / "eval_trained.json", args.num_samples)

    # Run for base model
    base_results = results_dir / "results_base.json"
    print("\n=== Generating with base model ===")
    run_generation_for_model(args.base_model_path, args.data_dir, base_results, args.num_samples, args.device)
    run_simplified_evaluation(base_results, results_dir / "eval_base.json", args.num_samples)

    print("\nAll done. Compare eval_trained.json and eval_base.json in", results_dir)


if __name__ == "__main__":
    main()


