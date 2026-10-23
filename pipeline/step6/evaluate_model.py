"""
评估训练后的模型质量
- 从测试集或训练集中选择样本进行推理
- 检查生成结果的格式、知识链和实体识别质量
"""

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 从训练脚本导入相同的提示词构建函数
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


def build_messages(table_content: str) -> List[Dict[str, str]]:
    """构建推理时的消息格式（不包含assistant回复）"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{INPUT_PROMPT}\n\n<table_content>\n{table_content}\n</table_content>",
        },
    ]


def load_sample(data_dir: Path, sample_id: int = None) -> Dict:
    """加载一个样本"""
    json_files = sorted(data_dir.glob("*.json"), key=lambda x: int(x.stem))
    if sample_id is None:
        sample_id = random.randint(0, len(json_files) - 1)
    
    with open(json_files[sample_id], "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def generate(model, tokenizer, table_content: str, max_new_tokens: int = 2048):
    """生成知识挖掘结果"""
    messages = build_messages(table_content)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=30000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
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


def extract_json_from_text(text: str) -> Dict:
    """从生成文本中提取JSON"""
    # 尝试找到JSON部分
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except:
            pass
    
    # 尝试找到第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    
    return None


def evaluate_sample(generated: Dict, ground_truth: Dict) -> Dict:
    """评估单个样本的质量"""
    metrics = {
        "json_valid": False,
        "has_knowledge_mining": False,
        "num_chains_generated": 0,
        "num_chains_ground_truth": 0,
        "chains_match": False,
        "entities_match": False,
    }
    
    if generated is None:
        return metrics
    
    metrics["json_valid"] = True
    
    gen_km = generated.get("knowledge_mining", [])
    gt_km = ground_truth.get("knowledge_mining", [])
    
    if gen_km:
        metrics["has_knowledge_mining"] = True
        metrics["num_chains_generated"] = len(gen_km)
    
    if gt_km:
        metrics["num_chains_ground_truth"] = len(gt_km)
    
    # 简单比较：检查生成的链数量是否接近
    if metrics["num_chains_generated"] > 0 and metrics["num_chains_ground_truth"] > 0:
        ratio = metrics["num_chains_generated"] / metrics["num_chains_ground_truth"]
        metrics["chains_match"] = 0.8 <= ratio <= 1.2  # 允许20%误差
    
    # 检查是否有实体识别
    gen_has_entities = any(
        item.get("entity_recognition", {}) for item in gen_km
    )
    gt_has_entities = any(
        item.get("entity_recognition", {}) for item in gt_km
    )
    metrics["entities_match"] = gen_has_entities == gt_has_entities
    
    return metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/new_disk/yjw/checkpoints/qwen3_sft1",
        help="训练后的模型路径",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/weijie_yu/yjw/ECLO-RL/step6/Test",
        help="测试数据目录",
    )
    parser.add_argument("--num_samples", type=int, default=10, help="测试样本数量")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    args = parser.parse_args()
    
    print(f"加载模型: {args.model_path}")
    # 尝试从checkpoint加载tokenizer，如果失败则从原始模型加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"从checkpoint加载tokenizer失败: {e}")
        print("尝试从原始模型路径加载tokenizer...")
        base_model_path = "/home/new_disk/yjw/Model/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()
    print("模型加载完成\n")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return
    
    # 随机选择样本进行测试
    json_files = sorted(data_dir.glob("*.json"), key=lambda x: int(x.stem))
    if len(json_files) == 0:
        print(f"数据目录中没有JSON文件: {data_dir}")
        return
    
    test_samples = random.sample(json_files, min(args.num_samples, len(json_files)))
    
    print(f"开始测试 {len(test_samples)} 个样本...\n")
    print("=" * 80)
    
    all_metrics = []
    
    for i, json_file in enumerate(test_samples, 1):
        print(f"\n[样本 {i}/{len(test_samples)}] {json_file.name}")
        print("-" * 80)
        
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        table_content = data.get("table_content", "").strip()
        ground_truth = {"knowledge_mining": data.get("knowledge_mining", [])}
        
        if not table_content:
            print("⚠️  表格内容为空，跳过")
            continue
        
        print(f"表格内容长度: {len(table_content)} 字符")
        print(f"真实知识链数量: {len(ground_truth['knowledge_mining'])}")
        
        # 生成
        print("正在生成...")
        try:
            generated_text = generate(model, tokenizer, table_content)
            print(f"生成文本长度: {len(generated_text)} 字符")
            print(f"生成文本预览: {generated_text[:200]}...")
            
            # 提取JSON
            generated_json = extract_json_from_text(generated_text)
            
            if generated_json:
                print("✅ JSON解析成功")
                metrics = evaluate_sample(generated_json, ground_truth)
                all_metrics.append(metrics)
                
                print(f"  - 知识链数量: {metrics['num_chains_generated']} (真实: {metrics['num_chains_ground_truth']})")
                print(f"  - 链数量匹配: {'✅' if metrics['chains_match'] else '❌'}")
                print(f"  - 实体识别: {'✅' if metrics['entities_match'] else '❌'}")
                
                # 显示第一个知识链示例
                if generated_json.get("knowledge_mining"):
                    first_chain = generated_json["knowledge_mining"][0]
                    print(f"  - 示例知识链:")
                    print(f"    Condition: {first_chain.get('knowledge_chain', {}).get('condition', 'N/A')[:100]}")
                    print(f"    Result: {first_chain.get('knowledge_chain', {}).get('result', 'N/A')[:100]}")
                    entities = first_chain.get('entity_recognition', {})
                    if entities:
                        print(f"    Entities: {list(entities.keys())}")
            else:
                print("❌ JSON解析失败")
                print(f"生成文本: {generated_text[:500]}")
                all_metrics.append({"json_valid": False})
        
        except Exception as e:
            print(f"❌ 生成失败: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    # 统计结果
    print("\n" + "=" * 80)
    print("总体评估结果")
    print("=" * 80)
    
    if all_metrics:
        total = len(all_metrics)
        json_valid = sum(1 for m in all_metrics if m.get("json_valid", False))
        has_km = sum(1 for m in all_metrics if m.get("has_knowledge_mining", False))
        chains_match = sum(1 for m in all_metrics if m.get("chains_match", False))
        entities_match = sum(1 for m in all_metrics if m.get("entities_match", False))
        
        print(f"总样本数: {total}")
        print(f"JSON格式有效: {json_valid}/{total} ({json_valid/total*100:.1f}%)")
        print(f"包含knowledge_mining: {has_km}/{total} ({has_km/total*100:.1f}%)")
        print(f"知识链数量匹配: {chains_match}/{total} ({chains_match/total*100:.1f}%)")
        print(f"实体识别匹配: {entities_match}/{total} ({entities_match/total*100:.1f}%)")
        
        # 平均知识链数量
        avg_gen = sum(m.get("num_chains_generated", 0) for m in all_metrics) / total
        avg_gt = sum(m.get("num_chains_ground_truth", 0) for m in all_metrics) / total
        print(f"平均生成知识链数: {avg_gen:.1f}")
        print(f"平均真实知识链数: {avg_gt:.1f}")


if __name__ == "__main__":
    main()

