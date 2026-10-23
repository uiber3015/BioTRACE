#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7: COT多轮修复数据生成

输入：step5/output/train_cot/ - 集合2的Markdown + Teacher知识链
输出：多轮修复的COT轨迹

核心功能：
1. SFT-1生成初始知识链并选择修复策略（Add/Delete/Modify）
2. GPT-5执行修复操作
3. R_rule评分（与Teacher对比）
4. 迭代修复直到不再提升或达到最大轮次
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import requests
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 配置
SFT1_MODEL_PATH = project_root / "step6" / "output" / "sft1_model" / "final"
TRAIN_COT_DIR = project_root / "step5" / "output" / "train_cot"
OUTPUT_DIR = project_root / "step7" / "output"
COT_TRAJECTORIES_DIR = OUTPUT_DIR / "cot_trajectories"
PROGRESS_PATH = OUTPUT_DIR / "progress.json"

GPT5_API_URL = os.environ.get("LLM_API_URL", "")
GPT5_MODEL = "gpt-5"
GPT5_API_KEY = os.getenv("OPENAI_API_KEY", "")
TIMEOUT = 120
MAX_ROUNDS = 10  # 最大修复轮次


def load_progress() -> Dict:
    """加载进度"""
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, 'r', encoding='utf-8') as f:
            progress = json.load(f)
            return {k: set(v) if isinstance(v, list) else v for k, v in progress.items()}
    return {"processed": set(), "failed": set()}


def save_progress(progress: Dict):
    """保存进度"""
    progress_serializable = {
        k: list(v) if isinstance(v, set) else v
        for k, v in progress.items()
    }
    with open(PROGRESS_PATH, 'w', encoding='utf-8') as f:
        json.dump(progress_serializable, f, ensure_ascii=False, indent=2)


def load_sft1_model():
    """加载SFT-1模型"""
    print("🔄 加载SFT-1模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(SFT1_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SFT1_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print("✅ SFT-1模型加载完成")
    
    return tokenizer, model


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


def generate_initial_chain(tokenizer, model, caption: str, table_content: str) -> Optional[List[dict]]:
    """使用SFT-1生成初始知识链"""
    
    input_text = f"""## Caption
{caption}

## Table Content
{table_content}

请抽取知识链（JSON格式）:"""
    
    # 使用Qwen格式
    prompt = f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # 解析JSON
    try:
        cleaned = clean_json_response(response)
        chains = json.loads(cleaned)
        
        if isinstance(chains, list):
            return chains
    except:
        pass
    
    return None


def select_repair_strategy(
    tokenizer, model,
    current_chain: List[dict],
    teacher_chain: List[dict],
    markdown_table: str
) -> Optional[Dict]:
    """
    使用SFT-1选择修复策略
    
    Returns:
        {
            "strategy": "Add/Delete/Modify",
            "target_chain_index": int,  # 如果是Modify/Delete
            "reason": str,
            "expected_improvement": str
        }
    """
    
    prompt = f"""你是一个知识链修复策略专家。

当前知识链:
{json.dumps(current_chain, ensure_ascii=False, indent=2)}

Teacher参考知识链（仅供参考，不要直接复制）:
{json.dumps(teacher_chain, ensure_ascii=False, indent=2)}

原始表格:
{markdown_table}

请分析当前知识链的问题，并选择一种修复策略：
1. Add - 添加缺失的知识链
2. Delete - 删除错误或冗余的知识链
3. Modify - 修正某条链的内容

输出格式（JSON）:
{{
  "strategy": "Add/Delete/Modify",
  "target_chain_index": 0,
  "reason": "具体原因",
  "expected_improvement": "预期改进效果"
}}"""
    
    qwen_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(qwen_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    # 解析JSON
    try:
        cleaned = clean_json_response(response)
        strategy = json.loads(cleaned)
        return strategy
    except:
        return None


def execute_repair_with_gpt5(
    strategy: Dict,
    current_chain: List[dict],
    markdown_table: str
) -> Optional[Dict]:
    """
    使用GPT-5执行修复
    
    Returns:
        {
            "action_type": "Add/Delete/Modify",
            "action_detail": {
                "before": ...,
                "after": ...,
                "reason": ...,
                "evidence": ...
            },
            "modified_chain": [...]
        }
    """
    
    prompt = f"""你是一个知识链修复执行专家。

修复策略: {json.dumps(strategy, ensure_ascii=False)}
当前知识链: {json.dumps(current_chain, ensure_ascii=False)}
原始表格: {markdown_table}

请严格按照策略执行修复，输出格式:
{{
  "action_type": "Add/Delete/Modify",
  "action_detail": {{
    "before": "修复前的内容（如果是Add则为null）",
    "after": "修复后的内容（如果是Delete则为null）",
    "reason": "修复理由",
    "evidence": "表格中的证据（单元格或文本）"
  }},
  "modified_chain": [...]
}}"""
    
    for attempt in range(3):
        try:
            response = requests.post(
                GPT5_API_URL,
                headers={
                    "Authorization": f"Bearer {GPT5_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": GPT5_MODEL,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                cleaned = clean_json_response(content)
                result = json.loads(cleaned)
                return result
        
        except Exception as e:
            print(f"  ⚠️  GPT-5调用失败: {e}, 重试 {attempt+1}/3")
            time.sleep(2 ** attempt)
    
    return None


def calculate_r_rule(chain: List[dict], teacher_chain: List[dict], markdown_table: str) -> float:
    """
    计算R_rule评分（改进版）
    
    维度：
    1. 结构一致性 (25%): 检查必需字段和格式
    2. 实体覆盖度 (25%): 实体集合的Jaccard相似度
    3. 条件-结果匹配度 (30%): 条件和结果的语义对齐
    4. 证据准确性 (20%): 链数量和实体数量的综合评估
    """
    
    if not chain or not teacher_chain:
        return 0.0
    
    score = 0.0
    
    # 1. 结构一致性 (25%)
    structure_score = 0.0
    valid_chains = 0
    for c in chain:
        if all(k in c for k in ['condition', 'result', 'entity']):
            if c['condition'] and c['result'] and isinstance(c['entity'], dict):
                if ':' in c['result']:  # 必须有指标:数值格式
                    valid_chains += 1
    
    if len(chain) > 0:
        structure_score = valid_chains / len(chain)
    score += 0.25 * structure_score
    
    # 2. 实体覆盖度 (25%): Jaccard相似度
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
            jaccard = intersection / union
            score += 0.25 * jaccard
    
    # 3. 条件-结果匹配度 (30%): 检查条件和结果的合理性
    condition_result_score = 0.0
    
    for c in chain:
        c_score = 0.0
        
        # 检查condition非空且包含实体
        if c.get('condition'):
            c_score += 0.5
        
        # 检查result格式正确
        result = c.get('result', '')
        if ':' in result and '|' not in result[:result.index(':')]:
            c_score += 0.5
        
        condition_result_score += c_score
    
    if len(chain) > 0:
        score += 0.30 * (condition_result_score / len(chain))
    
    # 4. 证据准确性 (20%): 链数量和质量的综合
    evidence_score = 0.0
    
    # 链数量接近度
    if len(teacher_chain) > 0:
        count_ratio = min(len(chain), len(teacher_chain)) / max(len(chain), len(teacher_chain))
        evidence_score += 0.5 * count_ratio
    
    # 实体类型覆盖度
    chain_types = set()
    teacher_types = set()
    
    for c in chain:
        if 'entity' in c and isinstance(c['entity'], dict):
            chain_types.update(c['entity'].keys())
    
    for c in teacher_chain:
        if 'entity' in c and isinstance(c['entity'], dict):
            teacher_types.update(c['entity'].keys())
    
    if teacher_types:
        type_overlap = len(chain_types & teacher_types) / len(teacher_types)
        evidence_score += 0.5 * type_overlap
    
    score += 0.20 * evidence_score
    
    return min(1.0, score)


def clean_json_response(response: str) -> str:
    """清理JSON响应"""
    cleaned = response.strip()
    
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    return cleaned.strip()


def process_single_table(
    table_id: str,
    markdown_path: Path,
    teacher_chain: List[dict],
    tokenizer,
    model
) -> Optional[Dict]:
    """处理单个表格的COT修复轨迹"""
    
    # 读取markdown
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    caption, table_content = extract_caption_and_table(markdown_content)
    markdown_table = f"## Caption\n{caption}\n\n## Table Content\n{table_content}"
    
    # 1. 生成初始知识链
    initial_chain = generate_initial_chain(tokenizer, model, caption, table_content)
    
    if not initial_chain:
        return None
    
    # 2. 计算初始R_rule
    initial_r_rule = calculate_r_rule(initial_chain, teacher_chain, markdown_table)
    
    # 初始化轨迹
    trajectory = {
        "table_id": table_id,
        "markdown": markdown_table,
        "teacher_chain": teacher_chain,
        "initial_chain": initial_chain,
        "initial_r_rule": initial_r_rule,
        "rounds": []
    }
    
    current_chain = initial_chain
    current_r_rule = initial_r_rule
    no_improvement_count = 0
    
    # 3. 迭代修复
    for round_num in range(1, MAX_ROUNDS + 1):
        # SFT-1选择策略
        strategy = select_repair_strategy(
            tokenizer, model,
            current_chain,
            teacher_chain,
            markdown_table
        )
        
        if not strategy:
            break
        
        # GPT-5执行修复
        repair_result = execute_repair_with_gpt5(
            strategy,
            current_chain,
            markdown_table
        )
        
        if not repair_result or 'modified_chain' not in repair_result:
            break
        
        modified_chain = repair_result['modified_chain']
        
        # 计算修复后的R_rule
        new_r_rule = calculate_r_rule(modified_chain, teacher_chain, markdown_table)
        
        improvement = new_r_rule - current_r_rule
        
        # 记录本轮修复
        round_info = {
            "round": round_num,
            "strategy_from_sft1": strategy,
            "action_from_gpt5": repair_result.get('action_detail', {}),
            "modified_chain": modified_chain,
            "r_rule_after": new_r_rule,
            "improvement": improvement
        }
        
        trajectory["rounds"].append(round_info)
        
        # 检查是否有提升
        if improvement > 0:
            current_chain = modified_chain
            current_r_rule = new_r_rule
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
            # 连续3轮无提升则停止
            if no_improvement_count >= 3:
                break
    
    # 4. 设置最终结果
    trajectory["final_chain"] = current_chain
    trajectory["final_r_rule"] = current_r_rule
    trajectory["total_improvement"] = current_r_rule - initial_r_rule
    trajectory["total_rounds"] = len(trajectory["rounds"])
    trajectory["status"] = "improved" if current_r_rule > initial_r_rule else "no_improvement"
    
    return trajectory


def main():
    """主函数"""
    print("=" * 80)
    print("Step 7: COT多轮修复数据生成")
    print("=" * 80)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    COT_TRAJECTORIES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载进度
    progress = load_progress()
    processed_set = progress.get("processed", set())
    failed_set = progress.get("failed", set())
    
    print(f"📊 已处理: {len(processed_set)} 个表格")
    print(f"📊 已失败: {len(failed_set)} 个表格")
    
    # 加载SFT-1模型
    tokenizer, model = load_sft1_model()
    
    # 加载train_cot数据
    print("\n📁 加载train_cot数据...")
    
    teacher_chains_path = TRAIN_COT_DIR / "teacher_chains.json"
    if not teacher_chains_path.exists():
        print(f"❌ 错误: {teacher_chains_path} 不存在")
        print("请先运行 Step 5 生成 Teacher 知识链")
        return
    
    with open(teacher_chains_path, 'r', encoding='utf-8') as f:
        teacher_chains = json.load(f)
    
    markdown_dir = TRAIN_COT_DIR / "markdown_tables"
    
    print(f"✅ 共 {len(teacher_chains)} 个表格")
    print(f"📊 待处理: {len(teacher_chains) - len(processed_set) - len(failed_set)} 个表格")
    
    # 处理每个表格
    trajectories = []
    statistics = {
        "total": len(teacher_chains),
        "processed": len(processed_set),
        "improved": 0,
        "no_improvement": 0,
        "failed": len(failed_set)
    }
    
    print("\n🚀 开始生成COT轨迹...")
    
    for idx, (table_id, data) in enumerate(tqdm(teacher_chains.items(), desc="COT修复")):
        # 跳过已处理的
        if table_id in processed_set or table_id in failed_set:
            # 读取已有轨迹更新统计
            traj_file = COT_TRAJECTORIES_DIR / f"{table_id}.json"
            if traj_file.exists():
                with open(traj_file, 'r', encoding='utf-8') as f:
                    trajectory = json.load(f)
                    if trajectory.get("status") == "improved":
                        statistics["improved"] += 1
                    else:
                        statistics["no_improvement"] += 1
            continue
        
        markdown_path = markdown_dir / f"{table_id}.md"
        
        if not markdown_path.exists():
            statistics["failed"] += 1
            failed_set.add(table_id)
            progress["failed"] = failed_set
            save_progress(progress)
            continue
        
        # 处理单个表格
        try:
        trajectory = process_single_table(
            table_id,
            markdown_path,
            data['chains'],
            tokenizer,
            model
        )
        
        if trajectory:
            # 保存轨迹
            traj_file = COT_TRAJECTORIES_DIR / f"{table_id}.json"
            with open(traj_file, 'w', encoding='utf-8') as f:
                json.dump(trajectory, f, ensure_ascii=False, indent=2)
            
            trajectories.append({
                "table_id": table_id,
                "status": trajectory["status"],
                "initial_r_rule": trajectory["initial_r_rule"],
                "final_r_rule": trajectory["final_r_rule"],
                "total_rounds": trajectory["total_rounds"]
            })
            
            statistics["processed"] += 1
                processed_set.add(table_id)
                progress["processed"] = processed_set
                
            if trajectory["status"] == "improved":
                statistics["improved"] += 1
            else:
                statistics["no_improvement"] += 1
        else:
            statistics["failed"] += 1
                failed_set.add(table_id)
                progress["failed"] = failed_set
        
        except Exception as e:
            print(f"\n⚠️  处理失败 {table_id}: {e}")
            statistics["failed"] += 1
            failed_set.add(table_id)
            progress["failed"] = failed_set
        
        # 每10个表格保存一次进度
        if (idx + 1) % 10 == 0:
            save_progress(progress)
        
        # 延迟避免API限流
        time.sleep(1)
    
    # 保存汇总
    with open(OUTPUT_DIR / "cot_summary.json", 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    
    with open(OUTPUT_DIR / "statistics.json", 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print("\n" + "=" * 80)
    print("📊 统计信息")
    print("=" * 80)
    print(f"总表格数: {statistics['total']}")
    print(f"成功处理: {statistics['processed']}")
    print(f"有提升: {statistics['improved']}")
    print(f"无提升: {statistics['no_improvement']}")
    print(f"失败: {statistics['failed']}")
    
    print("\n✅ Step 7 完成!")


if __name__ == "__main__":
    main()
