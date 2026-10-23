#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step8 全面对比评测脚本

评测两个模型：
1. LoRA训练模型 (有SFT1混合数据)
2. 直接SFT模型 (无SFT1混合数据)

评测指标：
- 初始生成 R-rule 分数
- 多轮修正后 R-rule 分数  
- 修正轮数
- STOP准确率
- 策略选择准确率
"""

import os
import sys
import json
import time
import argparse
import copy
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

# 添加路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "step7"))

# 导入评估器
from step7.r_rule_evaluator_final import compute_r_rule


# ============== 配置 ==============
BASE_MODEL_PATH = "/home/new_disk/yjw/checkpoints/qwen3_sft1_20260225_183025"

# DeepSeek API 配置
DEEPSEEK_API_URL = os.environ.get("LLM_API_URL", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "DeepSeek-V3.2"

# ============== 评测侧保命机制配置 ==============
# 强退化回滚阈值
DEGRADE_ROLLBACK = 0.3
# 连续无进展提前停止
NO_PROGRESS_EPS = 1e-3
MAX_STAGNANT_ROUNDS = 3


def _normalize_text_for_kc(value: Any) -> str:
    """与 step7 评测一致的温和归一化。"""
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("-", "_")
    return text


def normalize_kc_payload(kc: Dict) -> Dict:
    """将 KC 归一化为稳定格式，便于教师修正与后续评分。"""
    if not isinstance(kc, dict):
        return {"knowledge_mining": []}

    items = kc.get("knowledge_mining", [])
    if not isinstance(items, list):
        items = []

    normalized_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        # 兼容两种链结构：
        # 1) 标准结构：{"knowledge_chain": {"condition": "...", "result": "..."}, ...}
        # 2) 扁平结构：{"condition": "...", "result": "...", ...}
        if isinstance(item.get("knowledge_chain"), dict):
            knowledge_chain = item.get("knowledge_chain", {})
        else:
            knowledge_chain = item

        condition = _normalize_text_for_kc(knowledge_chain.get("condition", ""))
        result = _normalize_text_for_kc(knowledge_chain.get("result", ""))

        entity_recognition = item.get("entity_recognition")
        if entity_recognition is None and isinstance(knowledge_chain, dict):
            entity_recognition = knowledge_chain.get("entity_recognition")
        if not isinstance(entity_recognition, dict):
            entity_recognition = {}

        normalized_entities: Dict[str, List[str]] = {}
        for raw_type, raw_values in entity_recognition.items():
            norm_type = _normalize_text_for_kc(raw_type)
            if not norm_type:
                continue

            values = raw_values if isinstance(raw_values, list) else [raw_values]
            norm_values: List[str] = []
            for value in values:
                norm_value = _normalize_text_for_kc(value)
                if norm_value:
                    norm_values.append(norm_value)

            # 去重并保持顺序
            if norm_values:
                normalized_entities[norm_type] = list(dict.fromkeys(norm_values))

        normalized_items.append(
            {
                "knowledge_chain": {
                    "condition": condition,
                    "result": result,
                },
                "entity_recognition": normalized_entities,
            }
        )

    return {"knowledge_mining": normalized_items}


def extract_json_payload(text: str) -> str:
    """从模型回复中提取合法 JSON（对象或数组）。"""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty response")

    def iter_balanced_candidates(s: str):
        n = len(s)
        for start in range(n):
            first = s[start]
            if first not in "{[":
                continue

            stack = [first]
            in_string = False
            escaped = False
            for end in range(start + 1, n):
                ch = s[end]

                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                    continue
                if ch in "{[":
                    stack.append(ch)
                    continue
                if ch in "}]":
                    if not stack:
                        break
                    top = stack[-1]
                    if (top == "{" and ch == "}") or (top == "[" and ch == "]"):
                        stack.pop()
                        if not stack:
                            yield s[start:end + 1]
                            break
                    else:
                        break

    def pick_first_valid(candidates):
        """返回第一个有效的 JSON 候选（最外层的）"""
        for i, candidate in enumerate(candidates):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, (dict, list)):
                    return candidate
            except Exception:
                continue
        return None

    if "```" in cleaned:
        blocks = cleaned.split("```")
        fence_candidates = []
        for block in blocks:
            candidate = block.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                fence_candidates.append(candidate)
        fenced = pick_first_valid(fence_candidates)
        if fenced is not None:
            return fenced

    balanced = pick_first_valid(iter_balanced_candidates(cleaned))
    if balanced is not None:
        return balanced

    raise ValueError("no valid json found")


# ============== Step6 提示词 ==============
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


# ============== 数据结构 ==============
@dataclass
class EvaluationResult:
    """单次评测结果"""
    sample_id: str
    initial_r_rule: float = 0.0
    final_r_rule: float = 0.0
    improvement: float = 0.0
    num_correction_rounds: int = 0
    stopped_correctly: bool = False  # 应该停止时是否停止
    strategy_correct: bool = False   # 策略选择是否正确
    error: str = ""
    # 新增字段
    initial_kc: Dict = field(default_factory=lambda: {})
    final_kc: Dict = field(default_factory=lambda: {})
    strategy_outputs: List[Dict] = field(default_factory=lambda: [])
    # 新增：保存每轮的KC
    round_kcs: List[Dict] = field(default_factory=lambda: [])
    # 新增：保存ground_truth
    ground_truth: List[Dict] = field(default_factory=lambda: [])
    # 新增：保存DeepSeek每轮的修正总结
    correction_summaries: List[str] = field(default_factory=lambda: [])
    # 区分STOP类型
    auto_stop: bool = False      # 因R>=0.9自动停止
    model_stop: bool = False     # 模型主动输出STOP
    target_reached: bool = False
    strategy_calls: int = 0
    strategy_parse_success: int = 0
    strategy_exec_success: int = 0
    strategy_improve_count: int = 0
    strategy_non_degrade_count: int = 0
    strategy_parse_success_rate: float = 0.0
    strategy_effective_rate: float = 0.0


# ============== Step8 模型类 ==============
class Step8Model:
    """Step8 模型包装类"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self._load_strategy_prompt()
    
    def _load_strategy_prompt(self):
        prompt_path = Path(__file__).parent / "prompts" / "strategy_selection.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.strategy_prompt_template = f.read()
    
    def load(self):
        print(f"📁 Loading model from {self.model_path}...")
        
        # 优先从评测模型目录加载tokenizer，失败则回退到基座tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                BASE_MODEL_PATH,
                trust_remote_code=True
            )
        
        # 加载chat_template
        chat_template_path = Path(self.model_path) / "chat_template.jinja"
        if not chat_template_path.exists():
            chat_template_path = Path(BASE_MODEL_PATH) / "chat_template.jinja"
        if chat_template_path.exists():
            self.tokenizer.chat_template = chat_template_path.read_text(encoding='utf-8')
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
        return self
    
    def generate_initial_kc(self, table_content: str) -> Dict:
        """生成初始知识链"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STEP6},
            {"role": "user", "content": f"{INPUT_PROMPT_STEP6}\n\n<table_content>\n{table_content}\n</table_content>"},
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=8000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        try:
            json_str = self._extract_json(generated_text)
            return json.loads(json_str)
        except:
            return {"knowledge_mining": []}
    
    def select_strategy(self, table_content: str, current_kc: Dict,
                        history_context: str = "", round_idx: int = 0) -> Dict:
        """选择修正策略"""
        
        prompt = self.strategy_prompt_template
        prompt = prompt.replace("{table_content}", table_content)
        prompt = prompt.replace("{current_knowledge_chain}", json.dumps(current_kc, ensure_ascii=False))
        
        if history_context:
            prompt = prompt.replace("{previous_summary_block}", f"\n## 历史修正\n{history_context}")
        else:
            prompt = prompt.replace("{previous_summary_block}", "")
        
        messages = [{"role": "user", "content": prompt}]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=8000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # 打印原始输出用于调试
        print(f"\n   🔍 模型原始输出:")
        print("   " + "-"*50)
        # 不截断，显示完整输出
        print("   " + generated_text.replace("\n", "\n   "))
        print("   " + "-"*50)
        
        try:
            # 使用多种方法提取JSON
            all_json_candidates = []
            
            # 方法1: 提取所有JSON候选
            for candidate in self._extract_all_json(generated_text):
                if isinstance(candidate, list):
                    for item in candidate:
                        if isinstance(item, dict):
                            all_json_candidates.append(item)
                elif isinstance(candidate, dict):
                    all_json_candidates.append(candidate)
            
            # 方法2: 使用原来的extract_json
            try:
                extracted = self._extract_json(generated_text)
                parsed_obj = json.loads(extracted)
                if isinstance(parsed_obj, list):
                    for item in parsed_obj:
                        if isinstance(item, dict):
                            all_json_candidates.append(item)
                elif isinstance(parsed_obj, dict):
                    all_json_candidates.append(parsed_obj)
            except Exception:
                pass
            
            print(f"   [DEBUG] 共找到 {len(all_json_candidates)} 个JSON候选")
            
            # 选择最佳候选
            result = None
            
            # 优先选择有action和strategy字段的
            for candidate in all_json_candidates:
                if "action" in candidate and "strategy" in candidate:
                    result = candidate
                    break
            
            # 然后选择任何包含strategy的
            if not result:
                for candidate in all_json_candidates:
                    if "strategy" in candidate:
                        result = candidate
                        break
            
            # 如果还没有，使用第一个
            if not result and all_json_candidates:
                result = all_json_candidates[0]
            
            if not result:
                raise ValueError("未找到有效的JSON候选")

            action = str(result.get("action", "")).strip().upper()
            strategy = str(result.get("strategy", "")).strip()
            if not action:
                action = "STOP" if strategy.upper() == "STOP" else "CORRECT"

            chain_indices = result.get("chain_indices", [])
            if not isinstance(chain_indices, list):
                chain_indices = []
            norm_indices = []
            seen_indices = set()
            for idx in chain_indices:
                try:
                    idx_int = int(idx)
                    if idx_int >= 0 and idx_int not in seen_indices:
                        norm_indices.append(idx_int)
                        seen_indices.add(idx_int)
                        if len(norm_indices) >= 7:
                            break
                except Exception:
                    continue
            
            reason = str(result.get("reason", "")) if "reason" in result else ""
            reason = reason.strip()
            if len(reason) > 256:
                reason = reason[:256]
            
            strategy_out = strategy if strategy else ("STOP" if action == "STOP" else "")
            if len(strategy_out) > 128:
                strategy_out = strategy_out[:128]

            return {
                "action": action,
                "strategy": strategy_out,
                "chain_indices": norm_indices[:7],
                "reason": reason,
                "_parse_ok": True,
            }
        except Exception as e:
            print(f"   ⚠️ 策略解析失败: {e}")
            return {
                "action": "ERROR",
                "strategy": "PARSE_ERROR",
                "chain_indices": [],
                "reason": "解析失败",
                "_parse_ok": False,
            }
    
    def _extract_all_json(self, text: str) -> List[Dict]:
        """提取文本中所有有效的JSON对象"""
        import re
        results = []
        cleaned = text.strip()

        # 提取markdown代码块中的JSON
        if "```" in cleaned:
            blocks = cleaned.split("```")
            for block in blocks:
                candidate = block.strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    try:
                        results.append(json.loads(candidate))
                    except Exception:
                        pass

        # 使用更宽松的正则表达式：找到任何 { 开头到 } 结尾的片段
        # 从每个 { 开始，尝试找到最外层的 }
        start_indices = [i for i, c in enumerate(cleaned) if c == '{']
        for start in start_indices:
            # 尝试从start位置往后找到平衡的 }
            depth = 0
            in_string = False
            escaped = False
            for end in range(start, len(cleaned)):
                ch = cleaned[end]
                if escaped:
                    escaped = False
                    continue
                if ch == '\\':
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = cleaned[start:end+1]
                            try:
                                parsed = json.loads(candidate)
                                if isinstance(parsed, dict):
                                    results.append(parsed)
                            except Exception:
                                pass
                            break
        
        print(f"   [DEBUG _extract_all_json] 找到 {len(results)} 个JSON对象")
        return results

    def _extract_json(self, text: str) -> str:
        """提取JSON"""
        return extract_json_payload(text)


# ============== DeepSeek 修正器 ==============
class DeepSeekCorrector:
    """DeepSeek修正器"""
    
    def __init__(self):
        self.teacher_prompts_dir = Path(__file__).parent / "prompts" / "teacher"
        self.api_url = DEEPSEEK_API_URL
        self.api_key = DEEPSEEK_API_KEY
        self.model_name = DEEPSEEK_MODEL
        
        # 策略到prompt文件映射
        self.strategy_to_file = {
            "Add/chain": "teacher_add_chain.txt",
            "Add/condition_kv": "teacher_add_condition_kv.txt",
            "Add/entity_type": "teacher_add_entity_type.txt",
            "Add/entity_value": "teacher_add_entity_value.txt",
            "Modify/condition_kv": "teacher_modify_condition_kv.txt",
            "Modify/result": "teacher_modify_result.txt",
            "Modify/entity_type": "teacher_modify_entity_type.txt",
            "Modify/entity_value": "teacher_modify_entity_value.txt",
            "Delete/chain": "teacher_delete_chain.txt",
            "Delete/condition_kv": "teacher_delete_condition_kv.txt",
            "Delete/result": "teacher_delete_result.txt",
            "Delete/entity_type": "teacher_delete_entity_type.txt",
            "Delete/entity_value": "teacher_delete_entity_value.txt",
        }
    
    def _load_teacher_prompt(self, strategy: str) -> str:
        filename = self.strategy_to_file.get(strategy, f"teacher_{strategy.replace('/', '_')}.txt")
        prompt_path = self.teacher_prompts_dir / filename
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Teacher prompt not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def correct(self, table_content: str, current_kc: Dict, 
                strategy_output: Dict, last_summary: str = "") -> Tuple[Dict, str]:
        """执行修正，返回(修正后的KC, 修正总结)"""
        
        prompt_template = self._load_teacher_prompt(strategy_output.get("strategy", ""))
        
        chain_indices = strategy_output.get("chain_indices", [])
        reason = strategy_output.get("reason", "")
        
        student_operation_json = {
            "strategy": strategy_output.get("strategy", ""),
            "chain_indices": chain_indices,
            "reason": reason
        }
        
        # 替换模板占位符
        prompt = prompt_template.replace("{TABLE_CONTENT}", table_content)
        normalized_current_kc = normalize_kc_payload(current_kc)
        prompt = prompt.replace("{CURRENT_GENERATED_JSON}", json.dumps(normalized_current_kc, ensure_ascii=False))
        prompt = prompt.replace("{STUDENT_OPERATION_JSON}", json.dumps(student_operation_json, ensure_ascii=False))
        prompt = prompt.replace("{STUDENT_REASON}", reason if reason else "（模型未提供理由，根据策略类型和表格内容自行判断）")
        
        if last_summary:
            prompt = prompt.replace("{LAST_CORRECTION_SUMMARY}", f"\n## 上一轮修正总结\n{last_summary}")
        else:
            prompt = prompt.replace("{LAST_CORRECTION_SUMMARY}", "")
        
        # 其他占位符
        prompt = prompt.replace("{chain_indices}", str(chain_indices))
        prompt = prompt.replace("{学生新增的具体内容摘要}", f"使用策略 {strategy_output.get('strategy', '')} 对链 {chain_indices} 进行修正")
        prompt = prompt.replace("{学生修改的具体内容摘要}", f"使用策略 {strategy_output.get('strategy', '')} 对链 {chain_indices} 进行修正")
        prompt = prompt.replace("{删除了哪些链}", f"删除链 {chain_indices}")
        prompt = prompt.replace("{新增的实体类型和值}", f"对链 {chain_indices} 添加实体")
        prompt = prompt.replace("{新增的实体值}", f"对链 {chain_indices} 添加实体值")
        prompt = prompt.replace("{删除了哪些类型}", f"删除链 {chain_indices} 的实体类型")
        prompt = prompt.replace("{删除了哪些实体值}", f"删除链 {chain_indices} 的实体值")
        prompt = prompt.replace("{删除了哪些片段}", f"删除链 {chain_indices} 的condition片段")
        prompt = prompt.replace("{旧result→新result}", f"修改链 {chain_indices} 的result")
        prompt = prompt.replace("{result置空}", f"置空链 {chain_indices} 的result")
        prompt = prompt.replace("{原类型→新类型}", f"修改链 {chain_indices} 的实体类型")
        prompt = prompt.replace("{旧值→新值}", f"修改链 {chain_indices} 的实体值")
        
        messages = [{"role": "user", "content": prompt}]
        
        # 调用API
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 8192
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=180
        )
        
        if response.status_code != 200:
            raise Exception(f"API call failed: {response.text}")
        
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]

        stripped_text = response_text.strip()

        # 教师明确要求停止时，不改动当前KC
        if re.match(r"^STOP(\b|[\s:：-])", stripped_text, flags=re.IGNORECASE):
            stop_extra = stripped_text[4:].strip()
            if stop_extra:
                return normalized_current_kc, f"TEACHER_STOP: {stop_extra}"
            return normalized_current_kc, "TEACHER_STOP"
        
        # 解析JSON和总结
        try:
            json_str = extract_json_payload(response_text)
        except Exception:
            short_resp = re.sub(r"\s+", " ", stripped_text)
            if len(short_resp) > 160:
                short_resp = short_resp[:157] + "..."
            return normalized_current_kc, f"INVALID_RESPONSE_NO_JSON: {short_resp}"

        try:
            parsed = json.loads(json_str)
            if not isinstance(parsed, dict):
                return normalized_current_kc, "INVALID_RESPONSE_NOT_OBJECT"
            corrected_kc = normalize_kc_payload(parsed)
        except Exception:
            return normalized_current_kc, "INVALID_RESPONSE_PARSE_ERROR"

        summary = ""
        json_pos = response_text.find(json_str)
        if json_pos >= 0:
            summary = response_text[json_pos + len(json_str):].strip()

        return corrected_kc, summary


# ============== 评测函数 ==============
def load_test_samples(n: int = 0) -> List[Dict]:
    """加载测试样本。n<=0 表示加载全部样本。"""
    test_dir = PROJECT_ROOT / "step6" / "Test"
    all_test_files = sorted(test_dir.glob("*.json"), key=lambda x: int(x.stem))
    if n <= 0:
        test_files = all_test_files
    else:
        test_files = all_test_files[:n]
    
    samples = []
    for f in test_files:
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            data['sample_id'] = f.stem
            samples.append(data)
    
    return samples


def select_shard_samples(
    samples: List[Dict],
    num_shards: int,
    shard_id: int,
) -> List[Dict]:
    """按样本顺序做稳定分片：idx % num_shards == shard_id。"""
    if num_shards <= 1:
        return samples
    return [sample for idx, sample in enumerate(samples) if (idx % num_shards) == shard_id]


def _compact_summary_for_history(summary: str, max_chars: int = 280) -> str:
    """压缩教师总结，避免历史上下文过长。"""
    if not summary:
        return ""

    text = str(summary).replace("\r\n", "\n").strip()
    if not text:
        return ""

    lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {"【修正总结】", "【上一轮修正总结】"}:
            continue
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
        if len(lines) >= 3:
            break

    compact = " | ".join(lines) if lines else re.sub(r"\s+", " ", text)
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return compact


def _kc_payload_equal(kc_a: Dict, kc_b: Dict) -> bool:
    """判断两份KC是否完全一致（用于识别无变化修正）。"""
    try:
        return json.dumps(kc_a, ensure_ascii=False, sort_keys=True) == json.dumps(
            kc_b, ensure_ascii=False, sort_keys=True
        )
    except Exception:
        return kc_a == kc_b


def evaluate_sample(model: Step8Model, corrector: DeepSeekCorrector,
                   sample: Dict, max_rounds: int = 5,
                   stop_threshold: float = 0.9) -> EvaluationResult:
    """评测单个样本。

    说明：模型在修正循环中看不到任何 R-rule 分数，R-rule 仅用于评估侧打分与自动停机。
    """

    sample_id = sample.get('sample_id', 'unknown')
    table_content = sample.get('table_content', '')
    ground_truth = sample.get('knowledge_mining', [])

    result = EvaluationResult(sample_id=sample_id)
    result.ground_truth = ground_truth

    if not table_content or not ground_truth:
        result.error = "Empty table or ground truth"
        return result

    gt_sample = {
        'ground_truth': {'knowledge_mining': ground_truth}
    }

    def score_kc(kc: Dict) -> float:
        gt_sample['generated_json'] = normalize_kc_payload(kc)
        try:
            score = compute_r_rule(gt_sample, 'generated_json').get('R_rule', 0.0)
            return float(score)
        except Exception:
            return 0.0

    # 1) 初始生成能力
    print(f"   📥 生成初始KC...")
    initial_kc = normalize_kc_payload(model.generate_initial_kc(table_content))
    result.initial_kc = initial_kc
    result.initial_r_rule = score_kc(initial_kc)

    print(f"\n   📄 初始KC完整JSON (共 {len(initial_kc.get('knowledge_mining', []))} 条):")
    print("   " + "-" * 50)
    initial_kc_json = json.dumps(initial_kc, ensure_ascii=False, indent=2)
    if len(initial_kc_json) > 2000:
        print("   " + initial_kc_json[:2000].replace("\n", "\n   "))
        print("   ... [truncated]")
    else:
        print("   " + initial_kc_json.replace("\n", "\n   "))
    print("   " + "-" * 50)

    # 2) 策略选择 + 多轮修正能力
    print(f"   🔄 开始多轮修正 (最多{max_rounds}轮)... 初始R-rule: {result.initial_r_rule:.4f}")
    current_kc = initial_kc
    current_r_rule = result.initial_r_rule
    history_context = ""
    last_summary = ""
    all_strategy_outputs = []
    round_r_rules = [current_r_rule]
    # 新增：保存每轮的KC
    round_kcs = [{"round": 0, "kc": current_kc, "r_rule": current_r_rule}]
    correction_summaries = []

    best_kc = initial_kc
    best_r_rule = current_r_rule
    best_round = 0
    stagnant_rounds = 0

    def append_history_entry(round_idx: int, strategy_name: str, indices: List[int], kc_changed: bool, summary: str):
        nonlocal history_context
        history_context += (
            f"\nRound {round_idx + 1}: strategy={strategy_name}, "
            f"indices={indices}, action=CORRECT, kc_changed={'yes' if kc_changed else 'no'}"
        )
        compact_summary = _compact_summary_for_history(summary)
        if compact_summary:
            history_context += f"\nRound {round_idx + 1} summary: {compact_summary}"

    for round_num in range(max_rounds):
        print(f"   Round {round_num + 1}...")

        strategy_output = model.select_strategy(
            table_content, current_kc, history_context, round_idx=round_num
        )
        all_strategy_outputs.append(strategy_output)
        result.strategy_calls += 1

        print(f"\n   📄 Round {round_num + 1} 策略输出完整JSON:")
        print("   " + "-" * 50)
        strategy_json = json.dumps(strategy_output, ensure_ascii=False, indent=2)
        print("   " + strategy_json.replace("\n", "\n   "))
        print("   " + "-" * 50)

        if not strategy_output.get("_parse_ok", True):
            result.error = "Strategy parse failed"
            print("   ❌ 策略解析失败，终止该样本评测")
            break

        result.strategy_parse_success += 1
        action = str(strategy_output.get('action', 'CORRECT')).upper()
        strategy = str(strategy_output.get('strategy', '')).strip()
        chain_indices = strategy_output.get('chain_indices', [])

        # 模型主动STOP（测试策略选择的停止能力）
        if action == "STOP" or strategy.upper() == "STOP":
            result.model_stop = True
            result.auto_stop = False
            result.stopped_correctly = current_r_rule >= stop_threshold
            print(f"   🛑 模型主动输出STOP (当前R-rule={current_r_rule:.4f})")
            break

        if action != "CORRECT":
            result.error = f"Unexpected action: {action}"
            print(f"   ❌ 非法action: {action}")
            break

        if strategy not in corrector.strategy_to_file:
            result.error = f"Unknown strategy: {strategy}"
            print(f"   ❌ 未知策略: {strategy}")
            break

        kc_before = copy.deepcopy(current_kc)
        before_r_rule = current_r_rule

        # 执行教师修正
        try:
            corrected_kc, last_summary = corrector.correct(
                table_content, current_kc, strategy_output, last_summary
            )
        except Exception as e:
            result.error = f"Correction failed: {e}"
            print(f"   ❌ 修正失败: {e}")
            break

        summary_text = (last_summary or "").strip()
        corrected_kc = normalize_kc_payload(corrected_kc)

        # 教师返回原KC或无需修改时，继续循环而不是提前结束
        if summary_text.upper().startswith("TEACHER_STOP"):
            result.strategy_exec_success += 1
            result.strategy_non_degrade_count += 1
            result.num_correction_rounds += 1
            # 教师返回STOP意味着无需修改，继续下一轮而不是提前结束
            round_r_rules.append(current_r_rule)
            round_kcs.append(
                {
                    "round": round_num + 1,
                    "kc": current_kc,  # 保持当前KC不变
                    "r_rule": current_r_rule,
                    "teacher_stop": True,
                }
            )
            correction_summaries.append(summary_text)
            append_history_entry(
                round_num,
                strategy,
                chain_indices,
                False,
                summary_text,
            )
            stagnant_rounds += 1
            print(f"   ℹ️ 教师认为无需修改，继续修正 (当前R-rule={current_r_rule:.4f})")
            if stagnant_rounds >= MAX_STAGNANT_ROUNDS:
                print("   ⚠️ 教师连续认为无需修改，提前停止修正")
                break
            continue  # 继续下一轮

        if summary_text.upper().startswith("INVALID_RESPONSE"):
            result.num_correction_rounds += 1
            current_kc = corrected_kc
            round_r_rules.append(current_r_rule)
            round_kcs.append(
                {
                    "round": round_num + 1,
                    "kc": current_kc,
                    "r_rule": current_r_rule,
                    "teacher_invalid_response": True,
                }
            )
            correction_summaries.append(summary_text)
            append_history_entry(
                round_num,
                strategy,
                chain_indices,
                False,
                summary_text,
            )
            stagnant_rounds += 1
            print(f"   ⚠️ 教师响应无有效JSON: {summary_text}")
            if stagnant_rounds >= MAX_STAGNANT_ROUNDS:
                print("   ⚠️ 教师连续异常响应，提前停止修正")
                break
            continue

        result.strategy_exec_success += 1
        result.num_correction_rounds += 1
        kc_changed = not _kc_payload_equal(kc_before, corrected_kc)
        current_kc = corrected_kc
        current_r_rule = score_kc(current_kc)
        round_r_rules.append(current_r_rule)
        # 保存每轮的KC
        round_kcs.append({"round": round_num + 1, "kc": current_kc, "r_rule": current_r_rule})
        # 保存DeepSeek的修正总结
        correction_summaries.append(summary_text)

        delta = current_r_rule - before_r_rule
        if delta > 0:
            result.strategy_improve_count += 1
        if delta >= 0:
            result.strategy_non_degrade_count += 1

        print(f"   📊 修正后R-rule: {current_r_rule:.4f} (Δ={delta:+.4f})")

        # --- 保命机制：强退化回滚 + 多轮无进展停机 ---
        if current_r_rule < before_r_rule - DEGRADE_ROLLBACK:
            print(
                f"   ⚠️ 退化过大 {before_r_rule:.4f} -> {current_r_rule:.4f}，已回滚并继续修正"
            )
            current_kc = kc_before
            current_r_rule = before_r_rule
            round_r_rules[-1] = current_r_rule
            round_kcs[-1] = {
                "round": round_num + 1,
                "kc": current_kc,
                "r_rule": current_r_rule,
                "rolled_back": True,
            }
            rollback_note = (
                f"[ROLLBACK] 上轮策略 {strategy} 退化 {before_r_rule:.4f} -> {current_r_rule:.4f}，"
                "已回滚为之前的 KC，当前轮仍可继续修正。"
            )
            correction_summaries[-1] = rollback_note
            last_summary = rollback_note
            append_history_entry(
                round_num,
                strategy,
                chain_indices,
                False,
                rollback_note,
            )
            stagnant_rounds = 0
            continue

        if abs(delta) < NO_PROGRESS_EPS:
            stagnant_rounds += 1
        else:
            stagnant_rounds = 0
        if stagnant_rounds >= MAX_STAGNANT_ROUNDS:
            print("   ⚠️ 连续多轮无提升，提前停止修正")
            break

        if current_r_rule > best_r_rule:
            best_r_rule = current_r_rule
            best_kc = current_kc
            best_round = round_num + 1
            print(f"   ⭐ 更新最佳结果 (Round {best_round}, R-rule={best_r_rule:.4f})")

        # 历史仅传动作与修正摘要，不传分数，避免把评估信号泄露给模型
        append_history_entry(
            round_num,
            strategy,
            chain_indices,
            kc_changed,
            last_summary,
        )

        # 评估侧自动停机（模型不可见）
        if current_r_rule >= stop_threshold:
            result.auto_stop = True
            result.model_stop = False
            result.stopped_correctly = True
            print(f"   ✅ 修正后R-rule={current_r_rule:.4f} >= {stop_threshold:.2f}，评估侧自动停止")
            break

    # 3) 聚合结果
    result.final_r_rule = best_r_rule
    result.final_kc = best_kc
    result.improvement = result.final_r_rule - result.initial_r_rule
    result.target_reached = result.final_r_rule >= stop_threshold
    result.strategy_outputs = all_strategy_outputs
    result.round_kcs = round_kcs
    result.correction_summaries = correction_summaries

    if result.strategy_calls > 0:
        result.strategy_parse_success_rate = result.strategy_parse_success / result.strategy_calls
    if result.strategy_exec_success > 0:
        result.strategy_effective_rate = result.strategy_non_degrade_count / result.strategy_exec_success
        result.strategy_correct = result.strategy_effective_rate >= 0.5
    elif result.model_stop:
        result.strategy_correct = result.stopped_correctly
        result.strategy_effective_rate = 1.0 if result.stopped_correctly else 0.0

    print(f"\n   📈 各轮R-rule: {[f'{r:.4f}' for r in round_r_rules]}")
    print(f"   🏆 最佳结果: Round {best_round}, R-rule={best_r_rule:.4f}")

    return result


def evaluate_model(model_name: str, model_path: str, corrector: DeepSeekCorrector,
                  test_samples: List[Dict], max_rounds: int = 5,
                  stop_threshold: float = 0.9,
                  on_sample_done: Optional[Callable[[List[EvaluationResult], int], None]] = None) -> List[EvaluationResult]:
    """评测一个模型"""
    
    print(f"\n{'='*60}")
    print(f"🔬 评测模型: {model_name}")
    print(f"   路径: {model_path}")
    print(f"{'='*60}")
    
    # 加载模型
    model = Step8Model(model_path)
    model.load()
    
    results = []
    
    for i, sample in enumerate(test_samples):
        print(f"\n[{i+1}/{len(test_samples)}] 评测样本 {sample.get('sample_id')}...")
        
        result = evaluate_sample(
            model, corrector, sample,
            max_rounds=max_rounds,
            stop_threshold=stop_threshold,
        )
        results.append(result)
        if on_sample_done is not None:
            on_sample_done(results, i + 1)
        
        print(f"   初始R-rule: {result.initial_r_rule:.4f}")
        print(f"   最终R-rule: {result.final_r_rule:.4f}")
        print(f"   提升: {result.improvement:.4f}")
        print(f"   修正轮数: {result.num_correction_rounds}")
        print(f"   策略解析成功率: {result.strategy_parse_success_rate:.2%}")
        print(f"   策略有效率: {result.strategy_effective_rate:.2%}")
        print(f"   停止正确: {result.stopped_correctly}")
    
    return results


def print_summary(model_name: str, results: List[EvaluationResult]):
    """打印评测结果摘要"""
    
    valid_results = [r for r in results if not r.error]
    
    if not valid_results:
        print(f"\n❌ 模型 {model_name}: 所有样本评测失败")
        return
    
    avg_initial = sum(r.initial_r_rule for r in valid_results) / len(valid_results)
    avg_final = sum(r.final_r_rule for r in valid_results) / len(valid_results)
    avg_improvement = sum(r.improvement for r in valid_results) / len(valid_results)
    avg_rounds = sum(r.num_correction_rounds for r in valid_results) / len(valid_results)
    stop_accuracy = sum(1 for r in valid_results if r.stopped_correctly) / len(valid_results)
    strategy_parse_rate = sum(r.strategy_parse_success_rate for r in valid_results) / len(valid_results)
    strategy_effective_rate = sum(r.strategy_effective_rate for r in valid_results) / len(valid_results)
    target_reach_rate = sum(1 for r in valid_results if r.target_reached) / len(valid_results)
    
    print(f"\n📊 模型: {model_name}")
    print(f"   样本数: {len(valid_results)}")
    print(f"   平均初始R-rule: {avg_initial:.4f}")
    print(f"   平均最终R-rule: {avg_final:.4f}")
    print(f"   平均提升: {avg_improvement:.4f}")
    print(f"   平均修正轮数: {avg_rounds:.1f}")
    print(f"   平均策略解析成功率: {strategy_parse_rate:.2%}")
    print(f"   平均策略有效率: {strategy_effective_rate:.2%}")
    print(f"   目标达成率(R≥阈值): {target_reach_rate:.2%}")
    print(f"   STOP准确率: {stop_accuracy:.2%}")


def build_output_data(
    args: argparse.Namespace,
    results: List[EvaluationResult],
    total_samples: int,
    completed_samples: int,
    output_file: Path,
    complete: bool,
) -> Dict[str, Any]:
    valid_results = [r for r in results if not r.error]
    if valid_results:
        avg_initial = sum(r.initial_r_rule for r in valid_results) / len(valid_results)
        avg_final = sum(r.final_r_rule for r in valid_results) / len(valid_results)
        avg_improvement = sum(r.improvement for r in valid_results) / len(valid_results)
        avg_rounds = sum(r.num_correction_rounds for r in valid_results) / len(valid_results)
        stop_accuracy = sum(1 for r in valid_results if r.stopped_correctly) / len(valid_results)
    else:
        avg_initial = avg_final = avg_improvement = avg_rounds = stop_accuracy = 0.0

    progress = float(completed_samples) / float(max(1, total_samples))
    return {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "run_state": {
            "complete": bool(complete),
            "total_assigned_samples": int(total_samples),
            "completed_samples": int(completed_samples),
            "progress": progress,
            "output_file": str(output_file),
        },
        "model": {
            "name": args.model_name,
            "path": args.model_path,
            "run_config": {
                "test_samples": args.test_samples,
                "max_rounds": args.max_rounds,
                "num_shards": args.num_shards,
                "shard_id": args.shard_id,
                "output_tag": args.output_tag,
            },
            "stats": {
                "avg_initial_r_rule": avg_initial,
                "avg_final_r_rule": avg_final,
                "avg_improvement": avg_improvement,
                "avg_correction_rounds": avg_rounds,
                "stop_accuracy": stop_accuracy,
            },
            "samples": [
                {
                    "sample_id": r.sample_id,
                    "initial_r_rule": r.initial_r_rule,
                    "final_r_rule": r.final_r_rule,
                    "improvement": r.improvement,
                    "num_correction_rounds": r.num_correction_rounds,
                    "stopped_correctly": r.stopped_correctly,
                    "auto_stop": r.auto_stop,
                    "model_stop": r.model_stop,
                    "error": r.error,
                    "initial_kc": r.initial_kc,
                    "final_kc": r.final_kc,
                    "strategy_outputs": r.strategy_outputs,
                    "round_kcs": r.round_kcs,
                    "ground_truth": r.ground_truth,
                    "correction_summaries": r.correction_summaries,
                }
                for r in results
            ],
        },
    }


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Step8 LoRA模型评测")
    parser.add_argument("--model_path", type=str, 
                       default="/home/new_disk/yjw/checkpoints/step8_merged",
                       help="模型路径")
    parser.add_argument("--model_name", type=str, default="LoRA模型(有SFT1混合)",
                       help="模型名称")
    parser.add_argument("--test_samples", type=int, default=0,
                       help="测试样本数量，<=0表示全部样本")
    parser.add_argument("--max_rounds", type=int, default=5,
                       help="最大修正轮数")
    parser.add_argument("--num_shards", type=int, default=1,
                       help="并行分片数（1表示不分片）")
    parser.add_argument("--shard_id", type=int, default=0,
                       help="当前分片ID（0-based）")
    parser.add_argument("--output_dir", type=str, default="step8/logs",
                       help="评测结果输出目录")
    parser.add_argument("--output_tag", type=str, default="",
                       help="输出文件附加标签（用于并行时避免重名）")
    
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(
            f"shard_id must be in [0, {args.num_shards - 1}], got {args.shard_id}"
        )
    
    print("="*60)
    print("🔬 Step8 LoRA模型评测")
    print("="*60)
    print(f"🔀 分片配置: shard {args.shard_id}/{args.num_shards}")
    
    # 加载测试样本
    load_target_desc = "全部" if args.test_samples <= 0 else str(args.test_samples)
    print(f"\n📂 加载 {load_target_desc} 个测试样本...")
    all_samples = load_test_samples(args.test_samples)
    test_samples = select_shard_samples(all_samples, args.num_shards, args.shard_id)
    print(f"✅ 总样本数: {len(all_samples)}")
    print(f"✅ 当前分片样本数: {len(test_samples)}")
    if not test_samples:
        raise RuntimeError(
            f"No samples for shard {args.shard_id}/{args.num_shards}. "
            "Please check test_samples/num_shards/shard_id."
        )
    
    # 先确定输出文件，再做增量写入（每完成1个样本写一次同一个JSON）。
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix_parts: List[str] = []
    if args.num_shards > 1:
        suffix_parts.append(f"shard{args.shard_id}of{args.num_shards}")
    if args.output_tag:
        safe_tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(args.output_tag).strip())
        if safe_tag:
            suffix_parts.append(safe_tag)
    filename = f"evaluation_{timestamp}"
    if suffix_parts:
        filename += "_" + "_".join(suffix_parts)
    output_file = output_dir / f"{filename}.json"

    total_assigned_samples = len(test_samples)

    def persist_snapshot(
        current_results: List[EvaluationResult],
        completed_samples: int,
        complete: bool,
    ) -> None:
        payload = build_output_data(
            args=args,
            results=current_results,
            total_samples=total_assigned_samples,
            completed_samples=completed_samples,
            output_file=output_file,
            complete=complete,
        )
        atomic_write_json(output_file, payload)
        print(
            f"💾 增量保存: {completed_samples}/{total_assigned_samples} -> {output_file}",
            flush=True,
        )

    # 创建初始空快照，方便外部实时监控。
    persist_snapshot([], 0, complete=False)

    # 初始化修正器
    corrector = DeepSeekCorrector()

    # 评测模型（每个样本完成后做一次增量保存）
    results = evaluate_model(
        args.model_name,
        args.model_path,
        corrector,
        test_samples,
        max_rounds=args.max_rounds,
        on_sample_done=lambda rs, done_count: persist_snapshot(rs, done_count, complete=False),
    )
    print_summary(args.model_name, results)

    # 最终快照
    persist_snapshot(results, len(results), complete=True)

    print(f"\n💾 详细结果已保存到: {output_file}")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
