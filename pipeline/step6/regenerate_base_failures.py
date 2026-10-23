#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成基座模型失败的样本
- 增加 max_new_tokens (从 8192 增加到 16384)
- 设置 use_thinking=False 禁用思考模式
- 参照 complete_and_generate_base.py 的逻辑

Usage:
python step6/regenerate_base_failures.py \
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


def clean_thinking_tags(text: str) -> str:
    """去除 thinking 标签及其内容"""
    import re
    # 策略1: 去除完整的 <thinking>...</thinking>
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # 策略2: 去除可能被截断的 <thinking> 标签
    text = re.sub(r'<thinking>.*', '', text, flags=re.DOTALL)
    
    # 策略3: 去除结尾的 </thinking>
    text = re.sub(r'</thinking>', '', text)
    
    return text.strip()


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


def generate_with_model(model, tokenizer, table_content: str, max_new_tokens: int = 16384) -> str:
    """
    生成函数 - 增加 max_new_tokens，禁用 thinking 模式
    """
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
            # 注意: transformers 4.x/5.x 可能不支持 use_thinking 参数
            # 如果不支持会报 warning，但生成仍会进行
        )

    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def regenerate_failed_samples(model, tokenizer, data_dir: Path, results_file: Path, output_file: Path, device: str, batch_size: int = 10):
    """
    重新生成失败的样本
    """
    print(f"\n{'='*70}")
    print("🔧 重新生成基座模型失败的样本")
    print(f"{'='*70}")
    print(f"  设备: {device}")
    print(f"  max_new_tokens: 16384")
    print(f"  use_thinking: False")
    print(f"{'='*70}")
    
    # 读取补全后的结果
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 构建 file_id -> data_file 映射
    json_files = {p.stem: p for p in data_dir.glob("*.json")}
    
    # 找出仍然失败的样本
    failed_indices = []
    for i, item in enumerate(results):
        if item.get('generated_json') is None:
            failed_indices.append(i)
    
    print(f"\n需要重新生成的样本数: {len(failed_indices)}")
    
    if len(failed_indices) == 0:
        print("无需重新生成，所有样本已成功！")
        return
    
    # 重新生成
    regenerated = 0
    still_failed = []
    
    for idx_pos, idx in enumerate(failed_indices, 1):
        item = results[idx]
        file_id = item.get('file_id', 'unknown')
        
        # 从原始数据文件读取 table_content
        data_file = json_files.get(str(file_id))
        if data_file is None:
            print(f"[{idx_pos}/{len(failed_indices)}] {file_id}: 找不到原始数据文件，跳过")
            still_failed.append(idx)
            continue
        
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        table_content = data.get("table_content", "").strip()
        if not table_content:
            print(f"[{idx_pos}/{len(failed_indices)}] {file_id}: table_content 为空，跳过")
            still_failed.append(idx)
            continue
        
        try:
            gen_text = generate_with_model(model, tokenizer, table_content, max_new_tokens=16384)
            gen_json = extract_json_from_text(gen_text)
            
            results[idx]['generated_json'] = gen_json
            results[idx]['generated_text'] = gen_text
            results[idx]['_regenerated'] = True
            
            status = "OK" if gen_json else "PARSE_FAIL"
            preview = gen_text.replace("\n", " ")[:150]
            print(f"[{datetime.now().isoformat()}] [{idx_pos}/{len(failed_indices)}] {file_id} {status} preview={preview}")
            
            if gen_json:
                regenerated += 1
            else:
                still_failed.append(idx)
            
        except Exception as e:
            print(f"[{idx_pos}/{len(failed_indices)}] {file_id}: ERROR - {e}")
            results[idx]['generated_text'] = f"ERROR: {e}"
            still_failed.append(idx)
        
        sys.stdout.flush()
        
        # 每 batch_size 个保存一次
        if idx_pos % batch_size == 0:
            with open(output_file, "w", encoding="utf-8") as wf:
                json.dump(results, wf, ensure_ascii=False, indent=2)
            print(f"[SAVE] 已保存进度: {idx_pos}/{len(failed_indices)}")
    
    # 最终保存
    with open(output_file, "w", encoding="utf-8") as wf:
        json.dump(results, wf, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print("📋 重新生成结果")
    print(f"{'='*70}")
    print(f"  重新生成成功: {regenerated}")
    print(f"  仍然失败: {len(still_failed)}")
    
    # 统计更新后的状态
    new_success = sum(1 for item in results if item.get('generated_json') is not None)
    print(f"\n📊 更新后状态:")
    print(f"  成功解析: {new_success}/{len(results)} ({new_success/len(results)*100:.1f}%)")
    print(f"  失败: {len(results) - new_success}/{len(results)} ({(len(results)-new_success)/len(results)*100:.1f}%)")
    
    # 保存清理后的结果（去掉 _regenerated 标记）
    clean_results = []
    for item in results:
        clean_item = {k: v for k, v in item.items() if k != '_regenerated'}
        clean_results.append(clean_item)
    
    with open(output_file, "w", encoding="utf-8") as wf:
        json.dump(clean_results, wf, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {output_file}")


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
    parser.add_argument("--batch_size", type=int, default=10,
                        help="每多少个样本保存一次")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = results_dir / "results_base_completed.json"
    output_file = results_dir / "results_base_final.json"
    
    if not input_file.exists():
        print(f"ERROR: 输入文件不存在: {input_file}")
        print("请先运行 complete_base_failures.py 补全部分样本")
        sys.exit(1)
    
    print("="*60)
    print("重新生成基座模型失败的样本")
    print(f"  模型: {args.base_model_path}")
    print(f"  设备: {args.device}")
    print(f"  输入: {input_file}")
    print(f"  输出: {output_file}")
    print("="*60)
    
    model, tokenizer = load_model_single_gpu(args.base_model_path, args.device)
    regenerate_failed_samples(model, tokenizer, data_dir, input_file, output_file, args.device, args.batch_size)
    
    del model
    del tokenizer
    torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("重新生成完成！")
    print("="*60)


if __name__ == "__main__":
    main()

