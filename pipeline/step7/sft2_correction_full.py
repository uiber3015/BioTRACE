"""
SFT2修正循环数据生成脚本
自动处理测试集，生成完整的修正轨迹

功能：
1. 加载SFT1/SFT2模型
2. 使用DeepSeek API进行策略选择和修正
3. 最大修正轮数：4轮
4. 双卡并行运行
5. 生成完整的修正轨迹数据（SFT2训练集）
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse
import logging

# 设置日志级别，减少冗余输出
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("transformers.models.qwen2").setLevel(logging.WARNING)

# 添加路径
sys.path.insert(0, 'step6')
sys.path.insert(0, 'step7')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import requests
import time
import re

# ============== 配置 ==============
# 可以在命令行参数中覆盖这些配置

# 模型路径（两个 GPU 都使用同一个 SFT1 模型）
SFT1_MODEL_PATH = "/home/new_disk/yjw/checkpoints/qwen3_sft1_openr1_20260124_135733"

# DeepSeek API配置
DEEPSEEK_API_URL = os.environ.get("LLM_API_URL", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# 默认输出目录
DEFAULT_OUTPUT_DIR = "step7/sft2_correction_results"

# 测试集配置
# 优先使用 SFT2 目录中的所有 JSON 文件
SFT2_DIR = "step6/SFT2"
TEST_JSONL = ""  # 留空表示使用 SFT2_DIR

# 修正参数
MAX_ROUNDS = 4  # 最大修正轮数
MAX_WORKERS = 2  # 并行卡数

# ============== 提示词加载 ==============
def load_prompt(filename):
    with open(f"step7/prompts/{filename}", 'r', encoding='utf-8') as f:
        return f.read()

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

PROMPT_SELECT = load_prompt("student_prompt_select.txt")

TEACHER_PROMPTS = {
    "Add/chain": ("teacher_add_chain.txt", "teacher_add_chain.txt"),
    "Add/condition_kv": ("teacher_add_condition_kv.txt", "teacher_add_condition_kv.txt"),
    "Add/entity_type": ("teacher_add_entity_type.txt", "teacher_add_entity_type.txt"),
    "Add/entity_value": ("teacher_add_entity_value.txt", "teacher_add_entity_value.txt"),
    "Modify/condition_kv": ("teacher_modify_condition_kv.txt", "teacher_modify_condition_kv.txt"),
    "Modify/result": ("teacher_modify_result.txt", "teacher_modify_result.txt"),
    "Modify/entity_type": ("teacher_modify_entity_type.txt", "teacher_modify_entity_type.txt"),
    "Modify/entity_value": ("teacher_modify_entity_value.txt", "teacher_modify_entity_value.txt"),
    "Delete/chain": ("teacher_delete_chain.txt", "teacher_delete_chain.txt"),
    "Delete/condition_kv": ("teacher_delete_condition_kv.txt", "teacher_delete_condition_kv.txt"),
    "Delete/entity_type": ("teacher_delete_entity_type.txt", "teacher_delete_entity_type.txt"),
    "Delete/entity_value": ("teacher_delete_entity_value.txt", "teacher_delete_entity_value.txt"),
}

def load_teacher_prompt(strategy):
    if strategy not in TEACHER_PROMPTS:
        return None
    filename = TEACHER_PROMPTS[strategy][0]
    with open(f"step7/prompts/{filename}", 'r', encoding='utf-8') as f:
        return f.read()

# ============== 学生生成 ==============
def student_generate(model, tokenizer, table_content, device):
    """学生生成知识链 - 与 SFT 训练格式一致"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INPUT_PROMPT}\n\n<table_content>\n{table_content}\n</table_content>"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.cuda(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=8192, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    response = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(response, skip_special_tokens=False)
    
    # 返回原始文本，外部用parse_output解析
    cleaned = clean_thinking_tags(generated_text)
    return cleaned

# ============== 学生选择（DeepSeek API）==============
def student_select_deepseek(table_content, current, gt, sample_id, history_context=""):
    """DeepSeek API 选择修正策略（唯一可返回STOP的节点）"""
    
    user_content = f"""<table_content>
{table_content}
</table_content>

<current_knowledge_chains>
{json.dumps(current, ensure_ascii=False, indent=2)}
</current_knowledge_chains>

<ground_truth>
{json.dumps(gt, ensure_ascii=False, indent=2)}
</ground_truth>

请选择最佳修正操作，直接输出 operation JSON 或 STOP："""

    if history_context:
        user_content = f"""{user_content}

{history_context}"""

    payload = {
        "model": "DeepSeek-V3.2",
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.1,
        "max_tokens": 2048  # 与 test_closed_loop_full.py 对齐
    }
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    timeout_seconds = 60
    
    for attempt in range(3):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"].strip()
            
            # 解析输出
            if output.upper() == "STOP":
                return {"type": "STOP"}
            
            # 尝试提取JSON
            if "```json" in output:
                output = output.split("```json")[1].split("```")[0].strip()
            
            result = json.loads(output)
            
            # 检查是否包含STOP
            if result.get("strategy", "").upper() == "STOP":
                return {"type": "STOP"}
            
            return result
            
        except Exception as e:
            print(f"  [DeepSeek] 第{attempt+1}次尝试失败: {e}")
            if attempt < 2:
                time.sleep(5)
    
    return None

# ============== 教师修正（仅执行，不返回STOP）==============
def teacher_correct(operation, table_content, current, sample_id, last_correction_summary=""):
    """教师执行修正
    
    与 test_closed_loop_full.py 中的 teacher_correct 保持完全一致
    
    Args:
        operation: 学生选择的操作
        table_content: 表格内容
        current: 当前知识链
        sample_id: 样本ID
        last_correction_summary: 上一轮教师的修正总结
    """
    
    # 解析操作类型
    if "strategy" in operation:
        strategy_key = operation.get("strategy", "")
    elif "op_type" in operation:
        strategy_key = f"{operation.get('op_type', '')}/{operation.get('op_subtype', '')}"
    else:
        return None
    
    if strategy_key not in TEACHER_PROMPTS:
        print(f"  ⚠️ 未知操作: {strategy_key}")
        return None
    
    prompt_file = TEACHER_PROMPTS[strategy_key][0]
    prompt = load_prompt(prompt_file)
    reason = operation.get("reason", "")
    
    # 与 test_closed_loop_full.py 完全对齐的 input_text 结构
    input_text = f"""{prompt}
====================
【table_content】
{table_content}

【current_generated_json】
{json.dumps(current, ensure_ascii=False, indent=2)}

【student_operation_json】
{json.dumps(operation, ensure_ascii=False, indent=2)}

【student_reason】
{reason}

{last_correction_summary}

请直接输出 JSON 或 STOP："""
    
    payload = {
        "model": "DeepSeek-V3.2",
        "messages": [{"role": "user", "content": input_text}],
        "temperature": 0.1,
        "max_tokens": 8192,
    }
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    timeout_seconds = 240
    
    for attempt in range(3):
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=timeout_seconds)
            response.raise_for_status()
            output = response.json()["choices"][0]["message"]["content"].strip()
            
            if output.upper() == "STOP":
                return {"type": "STOP"}
            
            if "```json" in output:
                output = output.split("```json")[1].split("```")[0].strip()
            elif "```" in output:
                output = output.split("```")[1].split("```")[0].strip()

            # 提取修正总结（如果有）
            correction_summary = ""
            json_output = output  # 默认使用完整输出
            if "【上一轮修正总结】" in output:
                # 从输出中提取修正总结部分
                summary_start = output.find("【上一轮修正总结】")
                correction_summary = output[summary_start:].strip()
                # 只保留【修正总结】之前的JSON部分
                json_output = output[:summary_start].strip()

            # 解析JSON部分（只解析JSON内容，不包含修正总结）
            try:
                corrected_json = json.loads(json_output)
            except json.JSONDecodeError as json_err:
                # 如果直接解析失败，尝试找到并解析第一个JSON对象
                print(f"  ⚠️ JSON解析失败，尝试修复: {json_err}")
                try:
                    # 找到第一个 { 开始的位置
                    json_start = output.find('{')
                    if json_start != -1:
                        # 尝试逐字符找到合理的结束位置
                        brace_count = 0
                        in_string = False
                        escape_next = False
                        json_end = -1
                        
                        for i, char in enumerate(output[json_start:], json_start):
                            if escape_next:
                                escape_next = False
                                continue
                            if char == '\\':
                                escape_next = True
                                continue
                            if char == '"' and not escape_next:
                                in_string = not in_string
                                continue
                            if in_string:
                                continue
                            
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        
                        if json_end > json_start:
                            json_str = output[json_start:json_end]
                            corrected_json = json.loads(json_str)
                            # 修正总结可能在JSON之后
                            if "【上一轮修正总结】" in output[json_end:]:
                                summary_start = output.find("【上一轮修正总结】", json_end)
                                correction_summary = output[summary_start:].strip()
                            print(f"  ✅ 修复成功，提取到JSON对象")
                        else:
                            raise json_err
                    else:
                        raise json_err
                except Exception as fix_err:
                    print(f"  ❌ JSON修复失败: {fix_err}")
                    return None

            # 返回修正结果和总结
            return {
                "result": corrected_json,
                "summary": correction_summary
            }
            
        except Exception as e:
            error_str = str(e)
            if "timed out" in error_str.lower():
                if attempt < 2:
                    print(f"  ⚠️ API超时 (尝试 {attempt + 1}/3)，等待5秒后重试...")
                    time.sleep(5)
                    continue
                else:
                    print(f"  ❌ API超时，已重试 3 次: {e}")
                    return None
            else:
                print(f"  ❌ API错误: {e}")
                return None
    
    return None

# ============== R-rule 评估 ==============
from r_rule_evaluator_final import compute_r_rule as evaluate_r_rule

def evaluate_sample(current, gt, sample_id, stage):
    """使用R-rule最终版评估（对齐r_rule_evaluator_final.py）

    Args:
        current: 生成的 knowledge_mining JSON
        gt: ground truth JSON
        sample_id: 样本ID
        stage: 阶段标识 ("before" 或 "after")

    Returns:
        包含 R_rule, S_structure, S_content 等字段的字典
    """
    sample = {
        "sample_id": sample_id,
        "generated_json": current,
        "ground_truth": gt
    }

    try:
        result = evaluate_r_rule(sample)
        return {
            "stage": stage,
            "R_rule": round(result["R_rule"], 4),
            "S_structure": round(result["S_structure"], 4),
            "S_content": round(result["S_content"], 4),
            "chain_count": len(current.get("knowledge_mining", [])),
            "per_chain_debug": result.get("per_chain_debug", []),
            # 额外信息用于分析
            "n_pred": result.get("n_pred", 0),
            "n_gold": result.get("n_gold", 0),
            "match_strategy": result.get("strategy", ""),
        }
    except Exception as e:
        return {
            "stage": stage,
            "R_rule": 0.0,
            "S_structure": 0.0,
            "S_content": 0.0,
            "chain_count": len(current.get("knowledge_mining", [])),
            "error": str(e)
        }

# ============== 清理思考标签 ==============
def clean_thinking_tags(text: str) -> str:
    """清理Qwen3模型的thinking标签和特殊token，提取JSON内容"""
    # 移除特殊token
    text = re.sub(r'<\|im_end\|>', '', text)
    text = re.sub(r'<\|im_start\|>', '', text)
    text = re.sub(r'<\|endoftext\|>', '', text)
    
    # 找到JSON开始位置
    json_start = text.find('{')
    json_start_list = text.find('[')
    if json_start == -1:
        json_start = json_start_list
    if json_start_list != -1 and (json_start == -1 or json_start_list < json_start):
        json_start = json_start_list
    
    if json_start != -1:
        json_text = text[json_start:].strip()
        # 尝试修复截断的JSON：移除末尾不完整的部分
        # 尝试找到一个合理的结束位置
        json_text = _fix_truncated_json(json_text)
        return json_text
    
    return text.strip()

def _fix_truncated_json(text: str) -> str:
    """修复截断的JSON，尝试找到一个合理的结束位置"""

    # 如果文本看起来完整，直接返回
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 特殊情况1：chain_indices 列表被截断，但reason还在后面
    # 匹配 "chain_indices": [1, 2, 3, ... 后面可能被截断
    chain_match = re.search(r'"chain_indices":\s*\[', text)
    if chain_match:
        # 尝试找到 chain_indices 的闭合括号
        bracket_end = -1
        depth = 0
        in_string = False
        escape_next = False
        for i in range(chain_match.end() - 1, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    bracket_end = i + 1
                    break

        if bracket_end > 0:
            # 提取已有的索引
            indices_str = text[chain_match.end():bracket_end-1]
            indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
            fixed_indices = sorted(indices)[:5] if indices else []

            # 提取 prefix（到 chain_indices 开始之前）
            prefix = text[:chain_match.start()]

            # 提取 suffix（从 chain_indices 闭合括号之后）
            suffix = text[bracket_end:]

            # 在 suffix 中找到 "reason" 字段
            reason_match = re.search(r'"reason"\s*:\s*"', suffix)
            if reason_match:
                # 找到 reason 的闭合引号
                reason_start = reason_match.end()
                reason_end = -1
                escape_next = False
                for i in range(reason_start, len(suffix)):
                    char = suffix[i]

                    if escape_next:
                        escape_next = False
                        continue

                    if char == '\\':
                        escape_next = True
                        continue

                    if char == '"' and not escape_next:
                        reason_end = i
                        break

                if reason_end > reason_start:
                    reason_content = suffix[reason_start:reason_end]
                    # 构建修复后的 JSON
                    fixed_text = prefix + f'"chain_indices": {fixed_indices}, "reason": "{reason_content}"' + suffix[reason_end+1:]
                    try:
                        json.loads(fixed_text)
                        return fixed_text
                    except:
                        pass

    # 特殊情况2：chain_indices 和 reason 都被截断
    chain_match = re.search(r'"chain_indices":\s*\[\s*([\d,\s]+)(,?\s*)$', text)
    if chain_match:
        indices_str = chain_match.group(1)
        indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
        if indices:
            fixed_indices = sorted(indices)[:5]
            prefix = text[:chain_match.start()]
            suffix = text[chain_match.end():]
            next_char_match = re.search(r'\S', suffix)
            if next_char_match:
                suffix = suffix[next_char_match.start():]
                reason_match = re.search(r'"reason"\s*:\s*"', suffix)
                if reason_match:
                    suffix = suffix[reason_match.start():]
                else:
                    suffix = "}"
            else:
                suffix = "}"

            # 🔧 关键修复：使用合理的默认 reason，而不是空字符串
            reason_map = {
                "Modify/condition_kv": "修正 condition 字段名或值",
                "Modify/result": "修正 result 字段",
                "Modify/entity_type": "修正实体类型",
                "Modify/entity_value": "修正实体值",
                "Add/chain": "添加新知识链",
                "Delete/chain": "删除无效知识链",
                "Delete/condition_kv": "删除错误 condition",
                "Delete/result": "删除错误 result",
                "Delete/entity_type": "删除错误实体类型",
                "Delete/entity_value": "删除错误实体值",
            }
            # 从 prefix 中尝试提取 strategy
            strategy_match = re.search(r'"strategy"\s*:\s*"([^"]+)"', prefix)
            strategy = strategy_match.group(1) if strategy_match else ""
            default_reason = reason_map.get(strategy, "修正知识链")

            fixed_text = prefix + f'"chain_indices": {fixed_indices}, "reason": "{default_reason}"' + suffix
            try:
                json.loads(fixed_text)
                return fixed_text
            except:
                pass

    # 尝试移除末尾不完整的部分
    depth = 0
    in_string = False
    escape_next = False

    for i in range(len(text) - 1, -1, -1):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char in '}]':
            depth += 1
        elif char in '[{':
            depth -= 1
            if depth == 0:
                return text[:i + 1].strip()

    return text

# ============== 解析输出 ==============
def parse_output(text: str) -> tuple:
    """解析模型输出，返回 (success, output_type, data)"""
    text_clean = text.strip()
    
    # 如果是 STOP
    if text_clean.upper() == "STOP":
        return True, "STOP", None
    
    # 尝试解析JSON
    try:
        if "```json" in text_clean:
            text_clean = text_clean.split("```json")[1].split("```")[0].strip()
        elif "```" in text_clean:
            text_clean = text_clean.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(text_clean)
        
        # 检查是 operation 还是 strategy 格式
        if "op_type" in parsed:
            return True, "operation", parsed
        elif "strategy" in parsed:
            return True, "strategy", parsed
        elif "knowledge_mining" in parsed:
            return True, "knowledge_mining", parsed
        else:
            return False, "unknown", parsed
            
    except json.JSONDecodeError:
        return False, "parse_error", None
    
    return False, "no_match", None



# ============== Worker 函数 ==============
def run_worker(samples, device, max_rounds, output_dir):
    """Worker函数：每个GPU加载模型一次，处理所有样本"""
    model_path = SFT1_MODEL_PATH
    print(f"[GPU{device}] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_memory={i: "40GB" for i in range(2) if i != device},
    ).cuda(device)
    model.eval()
    # 设置 generation_config，与 test_closed_loop_full.py 一致
    model.generation_config = GenerationConfig(
        do_sample=False, pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, max_new_tokens=8192
    )
    print(f"[GPU{device}] 模型加载完成，共 {len(samples)} 个样本", flush=True)
    sys.stdout.flush()

    results = []
    for sample in samples:
        result = process_sample(sample, model, tokenizer, device, max_rounds, output_dir)
        results.append(result)

    return results

# ============== 处理单个样本 ==============
def process_sample(sample, model, tokenizer, device, max_rounds, output_dir):
    """处理单个样本的完整修正流程 - 模型已加载，传入即可"""
    sample_id = sample.get("sample_id", "unknown")
    table_content = sample.get("table_content", "")
    ground_truth = sample.get("ground_truth", {})

    result = {
        "sample_id": sample_id,
        "table_content": table_content[:1000] + "..." if len(table_content) > 1000 else table_content,
        "ground_truth": ground_truth,
        "status": "pending",
        "error": None,
    }

    try:
        # 学生生成（使用传入的模型）
        print(f"[GPU{device}] 样本 {sample_id}: 学生生成...", flush=True)
        generated_raw = student_generate(model, tokenizer, table_content, device)
        success, output_type, generated = parse_output(generated_raw)

        # 如果解析失败但有原始输出，尝试更彻底的修复
        if not success and generated_raw:
            # 直接从原始输出尝试提取 JSON
            for fix_type in ["{", "```json"]:
                try:
                    if fix_type == "```json":
                        if "```json" in generated_raw:
                            test_text = generated_raw.split("```json")[1].split("```")[0].strip()
                        else:
                            continue
                    else:
                        json_start = generated_raw.find(fix_type)
                        if json_start != -1:
                            test_text = _fix_truncated_json(generated_raw[json_start:])
                        else:
                            continue

                    if test_text:
                        fixed_parsed = json.loads(test_text)
                        if fixed_parsed and "knowledge_mining" in fixed_parsed:
                            success, output_type, generated = True, "json_fixed", fixed_parsed
                            print(f"[GPU{device}] 样本 {sample_id}: JSON修复成功")
                            break
                except (json.JSONDecodeError, IndexError):
                    continue

        if not success or not generated or "knowledge_mining" not in generated:
            result["status"] = "error"
            result["error"] = f"学生生成失败 ({output_type})"
            # 保存更多调试信息
            result["debug_raw"] = generated_raw[:1000] if generated_raw else "None"
            result["debug_cleaned"] = generated_clean[:1000] if generated_clean else "None"
            print(f"[GPU{device}] 样本 {sample_id}: 错误详情 - {result['error']}")
            return result
        
        result["initial_generated"] = generated
        initial_chains = len(generated.get("knowledge_mining", []))
        print(f"[GPU{device}] 样本 {sample_id}: 初始生成 {initial_chains} 条知识链")
        
        current = generated
        correction_history = []
        
        # 评估前
        eval_before = evaluate_sample(current, ground_truth, sample_id, "before")
        result["stages"] = {"before": eval_before}
        
        # 修正循环
        for iteration in range(max_rounds):
            current_chains_count = len(current.get("knowledge_mining", []))
            
            # 构建历史上下文（与 test_closed_loop_full.py 对齐）
            history_context = ""
            if iteration > 0 and correction_history:
                last = correction_history[-1]
                last_strategy = last.get('strategy', 'N/A')
                last_chains_before = last.get('chains_before', 0)
                last_chains_after = last.get('chains_after', 0)
                last_change = last_chains_after - last_chains_before
                last_reason = last.get('student_operation', {}).get('reason', '')[:200] if last.get('student_operation') else 'N/A'
                last_summary = last.get('teacher_summary', '')

                history_context = f"""【上一轮修正结果】
- 策略: {last_strategy}
- 链数变化: {last_chains_before} → {last_chains_after} ({last_change:+d})
- 上一轮学生要求: {last_reason}...
{last_summary}

【重要检查项】
请先确认上一轮修改是否成功执行：
1. 如果链数无变化，说明Teacher可能已正确执行，或者修改内容不符合预期
2. 如果策略与上一轮相同，必须在reason中说明"为什么还需要重复修改"
3. 如果该问题已解决，请转向其他问题或选择STOP"""

            # 学生选择策略
            print(f"[GPU{device}] 样本 {sample_id}: 第{iteration+1}轮 学生选择...")
            operation = student_select_deepseek(table_content, current, ground_truth, sample_id, history_context)
            
            # 处理STOP：只由学生选择STOP
            if not operation or operation.get("type") == "STOP":
                print(f"[GPU{device}] 样本 {sample_id}: 学生选择STOP，结束修正")
                correction_history.append({
                    "iteration": iteration + 1,
                    "action": "STOP",
                    "strategy": "N/A",
                    "chains_before": current_chains_count,
                    "chains_after": current_chains_count,
                    "reason": "学生判断无需继续修正"
                })
                break
            
            strategy = operation.get("strategy", f"{operation.get('op_type', '')}/{operation.get('op_subtype', '')}")
            print(f"[GPU{device}] 样本 {sample_id}: 学生选择策略: {strategy}")
            
            # 获取上一轮的修正总结（如果有）
            # 与 test_closed_loop_full.py 对齐
            last_correction_summary = ""
            if iteration > 0 and correction_history:
                # 找到上一轮有效的teacher_summary
                for hist in reversed(correction_history):
                    if hist.get("teacher_summary"):
                        last_correction_summary = hist["teacher_summary"]
                        break
            
            # 教师执行修正
            teacher_result = teacher_correct(operation, table_content, current, sample_id, last_correction_summary)
            
            if not teacher_result:
                print(f"[GPU{device}] 样本 {sample_id}: 教师修正失败")
                correction_history.append({
                    "iteration": iteration + 1,
                    "action": "FAILED",
                    "strategy": strategy,
                    "chains_before": current_chains_count,
                    "chains_after": current_chains_count,
                    "reason": "教师修正失败"
                })
                break
            
            corrected = teacher_result.get("result", teacher_result)
            correction_summary = teacher_result.get("summary", "")
            corrected_chains = len(corrected.get("knowledge_mining", []))
            
            correction_history.append({
                "iteration": iteration + 1,
                "action": "CORRECT",
                "strategy": strategy,
                "student_operation": operation,
                "teacher_summary": correction_summary,
                "chains_before": current_chains_count,
                "chains_after": corrected_chains,
                "corrected_json": corrected
            })
            
            current = corrected
            print(f"[GPU{device}] 样本 {sample_id}: 修正后 {corrected_chains} 条链")
        
        # 评估后
        eval_after = evaluate_sample(current, ground_truth, sample_id, "after")
        result["stages"]["after"] = eval_after
        
        result["status"] = "completed"
        result["final_generated"] = current
        result["correction_history"] = correction_history
        
        # 计算分数变化
        if eval_before.get("R_rule", 0) and eval_after.get("R_rule", 0):
            result["score_change"] = {
                "R_rule": round(eval_after["R_rule"] - eval_before["R_rule"], 4),
                "S_structure": round(eval_after["S_structure"] - eval_before["S_structure"], 4),
                "S_content": round(eval_after["S_content"] - eval_before["S_content"], 4),
            }
        
        print(f"[GPU{device}] 样本 {sample_id}: 完成")
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"[GPU{device}] 样本 {sample_id}: 错误 - {e}")
    
    return result

# ============== 生成报告 ==============
def generate_report(results, report_file, timestamp):
    """生成修正结果报告"""
    
    completed = [r for r in results if r.get("status") == "completed"]
    errors = [r for r in results if r.get("status") == "error"]
    
    report = []
    report.append("# SFT2 修正循环结果报告\n")
    report.append(f"**时间**: {timestamp}\n")
    report.append(f"**总样本数**: {len(results)}\n")
    report.append(f"**成功**: {len(completed)}\n")
    report.append(f"**失败**: {len(errors)}\n\n")
    
    # 分数统计
    if completed:
        r_rule_before = sum(r.get("stages", {}).get("before", {}).get("R_rule", 0) for r in completed) / len(completed)
        r_rule_after = sum(r.get("stages", {}).get("after", {}).get("R_rule", 0) for r in completed) / len(completed)
        s_struct_before = sum(r.get("stages", {}).get("before", {}).get("S_structure", 0) for r in completed) / len(completed)
        s_struct_after = sum(r.get("stages", {}).get("after", {}).get("S_structure", 0) for r in completed) / len(completed)
        s_content_before = sum(r.get("stages", {}).get("before", {}).get("S_content", 0) for r in completed) / len(completed)
        s_content_after = sum(r.get("stages", {}).get("after", {}).get("S_content", 0) for r in completed) / len(completed)
        
        report.append("## 分数变化统计\n\n")
        report.append("| 指标 | 修正前 | 修正后 | 变化 |\n")
        report.append("|------|--------|--------|------|\n")
        report.append(f"| R-rule | {r_rule_before:.4f} | {r_rule_after:.4f} | {r_rule_after - r_rule_before:+.4f} |\n")
        report.append(f"| S_structure | {s_struct_before:.4f} | {s_struct_after:.4f} | {s_struct_after - s_struct_before:+.4f} |\n")
        report.append(f"| S_content | {s_content_before:.4f} | {s_content_after:.4f} | {s_content_after - s_content_before:+.4f} |\n\n")
    
    # 提升/下降统计
    if completed:
        improved = sum(1 for r in completed if r.get("score_change", {}).get("R_rule", 0) > 0)
        stable = sum(1 for r in completed if r.get("score_change", {}).get("R_rule", 0) == 0)
        declined = sum(1 for r in completed if r.get("score_change", {}).get("R_rule", 0) < 0)
        
        report.append(f"## 样本级别变化\n\n")
        report.append(f"- 提升: {improved} 个 ({improved/len(completed)*100:.1f}%)\n")
        report.append(f"- 稳定: {stable} 个 ({stable/len(completed)*100:.1f}%)\n")
        report.append(f"- 下降: {declined} 个 ({declined/len(completed)*100:.1f}%)\n\n")
    
    # 样本详情
    report.append("## 样本详细结果\n\n")
    for r in completed:
        sample_id = r.get("sample_id", "unknown")
        before = r.get("stages", {}).get("before", {})
        after = r.get("stages", {}).get("after", {})
        change = r.get("score_change", {})
        history = r.get("correction_history", [])
        
        report.append(f"### 样本 {sample_id}\n\n")
        report.append(f"- 初始链数: {before.get('chain_count', 0)} → 最终链数: {after.get('chain_count', 0)}\n")
        report.append(f"- R-rule: {before.get('R_rule', 0):.4f} → {after.get('R_rule', 0):.4f} ({change.get('R_rule', 0):+.4f})\n")
        report.append(f"- 修正轮数: {len(history)}\n")
        
        if history:
            report.append("\n**修正历史**:\n")
            for h in history:
                report.append(f"\n- **轮次 {h.get('iteration', '?')}**: {h.get('strategy', '?')} ({h.get('action', '?')})\n")
                report.append(f"  - 链数变化: {h.get('chains_before', '?')} → {h.get('chains_after', '?')}\n")
                
                # 学生操作详情
                student_op = h.get("student_operation", {})
                if student_op:
                    report.append(f"  - 学生操作:\n")
                    report.append(f"    - 策略: {student_op.get('strategy', 'N/A')}\n")
                    if student_op.get('chain_indices'):
                        report.append(f"    - 目标链索引: {student_op.get('chain_indices')}\n")
                    if student_op.get('reason'):
                        report.append(f"    - 修正理由: {student_op.get('reason')}\n")
                
                # 教师总结
                teacher_summary = h.get("teacher_summary", "")
                if teacher_summary:
                    report.append(f"  - 教师总结: {teacher_summary}\n")
        
        report.append("\n---\n\n")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"报告已生成: {report_file}")

# ============== 增量保存函数 ==============
def incremental_save(results, output_dir, count):
    """增量保存结果到临时文件"""
    temp_file = output_dir / f"sft2_correction_temp_{count}.json"
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return temp_file

# ============== 主程序 ==============
def main():
    parser = argparse.ArgumentParser(description='SFT2 修正循环数据生成')
    parser.add_argument('--sft2_dir', type=str, default=SFT2_DIR, help='SFT2数据目录（包含id.json文件）')
    parser.add_argument('--test_jsonl', type=str, default=TEST_JSONL, help='测试集路径（可选，与sft2_dir二选一）')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, help='输出目录')
    parser.add_argument('--max_rounds', type=int, default=MAX_ROUNDS, help='最大修正轮数')
    parser.add_argument('--sample_limit', type=int, default=0, help='样本数限制（0表示全部）')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='并行数')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SFT2 修正循环数据生成")
    print(f"时间: {timestamp}")
    print("=" * 70)
    print()
    print(f"配置信息:")
    print(f"  - SFT2目录: {args.sft2_dir}")
    print(f"  - 测试集(可选): {args.test_jsonl if args.test_jsonl else '未使用'}")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 最大修正轮数: {args.max_rounds}")
    print(f"  - 并行数: {args.workers}")
    print(f"  - 样本限制: {args.sample_limit if args.sample_limit > 0 else '全部'}")
    print()
    
    # 加载样本（优先使用SFT2目录，其次使用test_jsonl）
    samples = []
    
    if args.sft2_dir and Path(args.sft2_dir).exists():
        # 从SFT2目录读取所有JSON文件
        print(f"从目录加载样本: {args.sft2_dir}")
        sft2_files = list(Path(args.sft2_dir).glob("*.json"))
        for file_path in sft2_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sample_id = file_path.stem
                    samples.append({
                        "sample_id": sample_id,
                        "table_content": data.get("table_content", ""),
                        "ground_truth": {"knowledge_mining": data.get("knowledge_mining", [])}
                    })
            except Exception as e:
                print(f"  ⚠️ 加载 {file_path.name} 失败: {e}")
        print(f"  从目录加载了 {len(samples)} 个样本")
        
    elif args.test_jsonl and Path(args.test_jsonl).exists():
        # 从JSONL文件加载
        print(f"从JSONL文件加载样本: {args.test_jsonl}")
        with open(args.test_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        print(f"  从JSONL加载了 {len(samples)} 个样本")
    else:
        print("错误: 未找到有效的数据源")
        return
    
    if not samples:
        print("错误: 样本为空")
        return
    
    print(f"总共加载了 {len(samples)} 个测试样本")
    
    # 限制样本数
    if args.sample_limit > 0:
        samples = samples[:args.sample_limit]
        print(f"限制处理前 {args.sample_limit} 个样本")
    
    # 分配样本到两张卡
    half = len(samples) // 2
    samples_gpu0 = samples[:half]
    samples_gpu1 = samples[half:]
    
    print(f"GPU0 处理 {len(samples_gpu0)} 个样本")
    print(f"GPU1 处理 {len(samples_gpu1)} 个样本")
    print()

    # 先加载模型（每个GPU加载一次）
    print("加载模型...")
    print(f"  GPU0: {SFT1_MODEL_PATH}")
    print(f"  GPU1: {SFT1_MODEL_PATH}")

    all_results = []
    sample_lock = threading.Lock()
    completed_count = 0  # 已完成的样本计数

    # 并行处理：每个GPU运行一个worker
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_gpu0 = executor.submit(run_worker, samples_gpu0, 0, args.max_rounds, output_dir)
        future_gpu1 = executor.submit(run_worker, samples_gpu1, 1, args.max_rounds, output_dir)

        # 收集GPU0结果
        try:
            results_gpu0 = future_gpu0.result()
            with sample_lock:
                all_results.extend(results_gpu0)
                completed_count += len(results_gpu0)
            print(f"GPU0 完成: {len(results_gpu0)} 个样本")
        except Exception as e:
            print(f"GPU0 处理异常: {e}")

        # 收集GPU1结果
        try:
            results_gpu1 = future_gpu1.result()
            with sample_lock:
                all_results.extend(results_gpu1)
                completed_count += len(results_gpu1)
            print(f"GPU1 完成: {len(results_gpu1)} 个样本")
        except Exception as e:
            print(f"GPU1 处理异常: {e}")

        print(f"\n进度: {completed_count}/{len(samples)} 完成")

    # 增量保存
    if all_results:
        temp_file = incremental_save(all_results, output_dir, completed_count)
        print(f"  💾 增量保存: 已完成 {completed_count} 个样本 -> {temp_file.name}")
    
    # 最终保存（覆盖临时文件）
    raw_output_file = output_dir / f"sft2_correction_raw_{timestamp}.json"
    with open(raw_output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 清理临时文件
    for temp_file in output_dir.glob("sft2_correction_temp_*.json"):
        temp_file.unlink()
    
    print()
    print("=" * 70)
    print("运行完成")
    print("=" * 70)
    print()
    print(f"结果文件:")
    print(f"  - 原始数据: {raw_output_file}")
    
    # 生成报告
    report_file = output_dir / f"sft2_correction_report_{timestamp}.md"
    generate_report(all_results, report_file, timestamp)
    
    print(f"  - 分析报告: {report_file}")
    print()

if __name__ == "__main__":
    main()

