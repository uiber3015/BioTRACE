#!/usr/bin/env python3
import os
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "step7"))

from step7.r_rule_evaluator_final import compute_r_rule

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

MAX_ROUNDS = 5
STOP_THRESHOLD = 0.9
DEGRADE_ROLLBACK = 0.3
NO_PROGRESS_EPS = 1e-3
MAX_STAGNANT_ROUNDS = 3

DEEPSEEK_API_URL = os.environ.get("LLM_API_URL", "")
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY_HERE"
DEEPSEEK_MODEL = "DeepSeek-V3.2"

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
4) 覆盖所有数值字段（含单位/百分号/科学计数），ID/编号优先放 condition.

实体识别规则（对齐 01_entity_type）：
1) 实体值必须在 condition 或 result 原文中**完整匹配出现**，不得拆分、不得补全。
2) 不拆分 items.text（逗号/分号等仍视为整体）；若含并列混杂则不标。
3) 仅允许 14 类实体：
   gene_symbol, regulator, target, functional_mechanism, species, disease,
   organ, tissue, cell, protein, compound, drug, experimental_method, phenotype
4) 同类去重，未识别则用空对象 {}，不得幻觉新增或虚构实体。

请直接输出合法 JSON，不要额外文本。"""


def _normalize_text_for_kc(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("-", "_")
    return text


def normalize_kc_payload(kc: Dict) -> Dict:
    if not isinstance(kc, dict):
        return {"knowledge_mining": []}

    items = kc.get("knowledge_mining", [])
    if not isinstance(items, list):
        items = []

    normalized_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue

        knowledge_chain = item.get("knowledge_chain", {}) if isinstance(item.get("knowledge_chain"), dict) else item
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
            norm_values = []
            for value in values:
                norm_value = _normalize_text_for_kc(value)
                if norm_value:
                    norm_values.append(norm_value)
            if norm_values:
                normalized_entities[norm_type] = list(dict.fromkeys(norm_values))

        normalized_items.append(
            {
                "knowledge_chain": {"condition": condition, "result": result},
                "entity_recognition": normalized_entities,
            }
        )

    return {"knowledge_mining": normalized_items}


def extract_json_payload(text: str) -> str:
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
                            yield s[start : end + 1]
                            break
                    else:
                        break

    def pick_first_valid(candidates):
        for candidate in candidates:
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


def _compact_summary_for_history(text: str, max_chars: int = 500) -> str:
    if not text:
        return ""
    lines = []
    for line in text.strip().split("\n"):
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
    try:
        return json.dumps(kc_a, ensure_ascii=False, sort_keys=True) == json.dumps(
            kc_b, ensure_ascii=False, sort_keys=True
        )
    except Exception:
        return kc_a == kc_b


def score_kc(generated_kc: Dict, ground_truth: List[Dict]) -> float:
    if not generated_kc or not generated_kc.get("knowledge_mining"):
        return 0.0
    gt_sample = {
        "ground_truth": {"knowledge_mining": ground_truth},
        "generated_json": normalize_kc_payload(generated_kc),
    }
    try:
        return float(compute_r_rule(gt_sample, "generated_json").get("R_rule", 0.0))
    except Exception:
        return 0.0


def load_model(model_path: str, device: str):
    print(f"Loading tokenizer from {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading model from {model_path} to {device} ...")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, tokenizer


class Step8Model:
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        initial_prompt_max_length: int = 30000,
        initial_max_new_tokens: int = 16384,
        strategy_prompt_max_length: int = 25000,
        strategy_max_new_tokens: int = 4096,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.initial_prompt_max_length = initial_prompt_max_length
        self.initial_max_new_tokens = initial_max_new_tokens
        self.strategy_prompt_max_length = strategy_prompt_max_length
        self.strategy_max_new_tokens = strategy_max_new_tokens
        prompt_path = Path(__file__).parent / "prompts" / "strategy_selection.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.strategy_prompt_template = f.read()

    def generate_initial_kc(self, table_content: str) -> Dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STEP6},
            {"role": "user", "content": f"{INPUT_PROMPT_STEP6}\n\n<table_content>\n{table_content}\n</table_content>"},
        ]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.initial_prompt_max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.initial_max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                return json.loads(self._extract_json(generated_text))
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    print(f"    初始生成失败，{RETRY_DELAY_SECONDS}秒后重试 ({attempt + 1}/{MAX_RETRIES})...")
                    time.sleep(RETRY_DELAY_SECONDS)
        print(f"    初始生成最终失败: {last_error}")
        return {"knowledge_mining": []}

    def select_strategy(self, table_content: str, current_kc: Dict, history_context: str = "", round_idx: int = 0) -> Dict:
        prompt = self.strategy_prompt_template
        prompt = prompt.replace("{table_content}", table_content)
        prompt = prompt.replace("{current_knowledge_chain}", json.dumps(current_kc, ensure_ascii=False))
        if history_context:
            prompt = prompt.replace("{previous_summary_block}", f"\n## 历史修正\n{history_context}")
        else:
            prompt = prompt.replace("{previous_summary_block}", "")

        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.strategy_prompt_max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_text = ""
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.strategy_max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    print(f"    策略选择失败，{RETRY_DELAY_SECONDS}秒后重试...")
                    time.sleep(RETRY_DELAY_SECONDS)
        else:
            return {"action": "ERROR", "strategy": "PARSE_ERROR", "chain_indices": [], "reason": f"生成失败: {last_error}", "_parse_ok": False}

        print("\n   🔍 模型原始输出:")
        print("   " + "-" * 50)
        print("   " + generated_text.replace("\n", "\n   "))
        print("   " + "-" * 50)

        try:
            all_json_candidates = []
            for candidate in self._extract_all_json(generated_text):
                if isinstance(candidate, list):
                    for item in candidate:
                        if isinstance(item, dict):
                            all_json_candidates.append(item)
                elif isinstance(candidate, dict):
                    all_json_candidates.append(candidate)

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
            result = None
            for candidate in all_json_candidates:
                if "action" in candidate and "strategy" in candidate:
                    result = candidate
                    break
            if not result:
                for candidate in all_json_candidates:
                    if "strategy" in candidate:
                        result = candidate
                        break
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

            reason = str(result.get("reason", "")).strip() if "reason" in result else ""
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
            return {"action": "ERROR", "strategy": "PARSE_ERROR", "chain_indices": [], "reason": "解析失败", "_parse_ok": False}

    def _extract_all_json(self, text: str) -> List[Dict]:
        results = []
        cleaned = text.strip()
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

        start_indices = [i for i, c in enumerate(cleaned) if c == "{"]
        for start in start_indices:
            depth = 0
            in_string = False
            escaped = False
            for end in range(start, len(cleaned)):
                ch = cleaned[end]
                if escaped:
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            candidate = cleaned[start : end + 1]
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
        return extract_json_payload(text)


class DeepSeekCorrector:
    def __init__(self):
        self.teacher_prompts_dir = Path(__file__).parent / "prompts" / "teacher"
        self.api_url = DEEPSEEK_API_URL
        self.api_key = DEEPSEEK_API_KEY
        self.model_name = DEEPSEEK_MODEL
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
        with open(prompt_path, "r", encoding="utf-8-sig") as f:
            return f.read()

    def correct(self, table_content: str, current_kc: Dict, strategy_output: Dict, last_summary: str = "") -> Tuple[Dict, str]:
        prompt_template = self._load_teacher_prompt(strategy_output.get("strategy", ""))
        chain_indices = strategy_output.get("chain_indices", [])
        reason = strategy_output.get("reason", "")
        student_operation_json = {
            "strategy": strategy_output.get("strategy", ""),
            "chain_indices": chain_indices,
            "reason": reason,
        }

        normalized_current_kc = normalize_kc_payload(current_kc)
        prompt = prompt_template.replace("{TABLE_CONTENT}", table_content)
        prompt = prompt.replace("{CURRENT_GENERATED_JSON}", json.dumps(normalized_current_kc, ensure_ascii=False))
        prompt = prompt.replace("{STUDENT_OPERATION_JSON}", json.dumps(student_operation_json, ensure_ascii=False))
        prompt = prompt.replace("{STUDENT_REASON}", reason if reason else "（模型未提供理由，根据策略类型和表格内容自行判断）")
        prompt = prompt.replace("{LAST_CORRECTION_SUMMARY}", f"\n## 上一轮修正总结\n{last_summary}" if last_summary else "")
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

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": self.model_name, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 8192}

        response_text = None
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(self.api_url, headers=headers, json=data, timeout=180)
                if response.status_code != 200:
                    raise Exception(f"API call failed: {response.text}")
                result = response.json()
                response_text = result["choices"][0]["message"]["content"]
                break
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    print(f"    API调用失败，{RETRY_DELAY_SECONDS}秒后重试...")
                    time.sleep(RETRY_DELAY_SECONDS)
        if response_text is None:
            raise last_error if last_error else RuntimeError("API call failed")

        stripped_text = response_text.strip()
        if re.match(r"^STOP(\b|[\s:：-])", stripped_text, flags=re.IGNORECASE):
            stop_extra = stripped_text[4:].strip()
            return (normalized_current_kc, f"TEACHER_STOP: {stop_extra}") if stop_extra else (normalized_current_kc, "TEACHER_STOP")

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
            summary = response_text[json_pos + len(json_str) :].strip()
        return corrected_kc, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="data/checkpoints/step8_merged")
    parser.add_argument("--test_dir", type=str, default="step6/Test")
    parser.add_argument("--output_dir", type=str, default="logs")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--failed_sample_ids", type=str, default=None)
    parser.add_argument("--initial_prompt_max_length", type=int, default=30000)
    parser.add_argument("--initial_max_new_tokens", type=int, default=16384)
    parser.add_argument("--strategy_prompt_max_length", type=int, default=25000)
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.failed_sample_ids:
        with open(args.failed_sample_ids, "r", encoding="utf-8") as f:
            failed_sample_ids = json.load(f)
    else:
        log_files = list(output_dir.glob("evaluation_*.json"))
        if not log_files:
            print("未找到评测结果文件，请指定 --failed_sample_ids")
            return
        failed_sample_ids = set()
        for lf in log_files:
            with open(lf, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sample in data.get("model", {}).get("samples", []):
                sid = sample["sample_id"]
                error = sample.get("error", "")
                initial_kc = sample.get("initial_kc", {})
                is_initial_empty = (
                    not initial_kc
                    or initial_kc == {}
                    or initial_kc.get("knowledge_mining") is None
                    or initial_kc.get("knowledge_mining") == []
                )
                is_api_error = (
                    "HTTPSConnectionPool" in error
                    or "API call failed" in error
                    or "dial tcp" in error
                    or "No available channel" in error
                    or "model_not_found" in error
                    or "Connection" in error
                    or "timeout" in error.lower()
                )
                if is_initial_empty or is_api_error:
                    failed_sample_ids.add(sid)
        failed_sample_ids = list(failed_sample_ids)

    print(f"需要重新生成的样本数: {len(failed_sample_ids)}")
    print(f"  - 分片: {args.num_shards}个, 当前第{args.shard_id}个")

    failed_sample_ids = sorted(failed_sample_ids)
    total_samples = len(failed_sample_ids)
    samples_per_shard = (total_samples + args.num_shards - 1) // args.num_shards
    start_idx = args.shard_id * samples_per_shard
    end_idx = min(start_idx + samples_per_shard, total_samples)
    my_samples = failed_sample_ids[start_idx:end_idx]

    json_files = {p.stem: p for p in test_dir.glob("*.json")}
    model, tokenizer = load_model(args.model_path, args.device)
    step8_model = Step8Model(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        initial_prompt_max_length=args.initial_prompt_max_length,
        initial_max_new_tokens=args.initial_max_new_tokens,
        strategy_prompt_max_length=args.strategy_prompt_max_length,
    )
    corrector = DeepSeekCorrector()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"regeneration_{timestamp}_shard{args.shard_id}of{args.num_shards}.json"
    results = {
        "timestamp": timestamp,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "model": {
            "name": "regeneration_with_correction",
            "path": args.model_path,
            "initial_prompt_max_length": args.initial_prompt_max_length,
            "initial_max_new_tokens": args.initial_max_new_tokens,
            "strategy_prompt_max_length": args.strategy_prompt_max_length,
            "max_rounds": MAX_ROUNDS,
        },
        "samples": [],
        "stats": {"total": len(my_samples), "success": 0, "parse_fail": 0, "error": 0, "rounds_total": 0},
    }

    for idx, sample_id in enumerate(my_samples, 1):
        file_path = json_files.get(str(sample_id))
        if not file_path:
            print(f"[{idx}/{len(my_samples)}] 样本 {sample_id}: 找不到测试数据文件")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        table_content = test_data.get("table_content", "").strip()
        ground_truth = test_data.get("knowledge_mining", [])
        if not table_content:
            print(f"[{idx}/{len(my_samples)}] 样本 {sample_id}: table_content为空")
            continue

        sample_result = {
            "sample_id": sample_id,
            "ground_truth": ground_truth,
            "initial_kc": None,
            "initial_r_rule": 0.0,
            "round_kcs": [],
            "round_r_rules": [],
            "strategy_outputs": [],
            "correction_summaries": [],
            "final_kc": None,
            "final_r_rule": 0.0,
            "best_round": 0,
            "num_correction_rounds": 0,
            "model_stop": False,
            "auto_stop": False,
            "stopped_correctly": False,
            "target_reached": False,
            "improvement": 0.0,
            "status": "error",
            "error": None,
        }

        try:
            print(f"[{idx}/{len(my_samples)}] 样本 {sample_id}: 初始生成...")
            initial_kc = normalize_kc_payload(step8_model.generate_initial_kc(table_content))
            sample_result["initial_kc"] = initial_kc
            sample_result["initial_r_rule"] = score_kc(initial_kc, ground_truth)

            if not initial_kc.get("knowledge_mining"):
                sample_result["status"] = "parse_fail"
                sample_result["error"] = "初始生成JSON解析失败"
                results["stats"]["parse_fail"] += 1
                results["samples"].append(sample_result)
                print("  初始生成解析失败")
                continue

            current_kc = initial_kc
            current_r_rule = sample_result["initial_r_rule"]
            history_context = ""
            last_summary = ""
            round_r_rules = [current_r_rule]
            round_kcs = [{"round": 0, "kc": current_kc, "r_rule": current_r_rule}]
            correction_summaries = []
            all_strategy_outputs = []
            best_kc = initial_kc
            best_r_rule = current_r_rule
            best_round = 0
            stagnant_rounds = 0
            correction_error = None

            print(f"   🔄 开始多轮修正 (最多{MAX_ROUNDS}轮)... 初始R-rule: {current_r_rule:.4f}")

            def append_history_entry(round_idx: int, strategy_name: str, indices: List[int], kc_changed: bool, summary: str):
                nonlocal history_context
                history_context += (
                    f"\nRound {round_idx + 1}: strategy={strategy_name}, "
                    f"indices={indices}, action=CORRECT, kc_changed={'yes' if kc_changed else 'no'}"
                )
                compact_summary = _compact_summary_for_history(summary)
                if compact_summary:
                    history_context += f"\nRound {round_idx + 1} summary: {compact_summary}"

            for round_num in range(MAX_ROUNDS):
                print(f"   Round {round_num + 1}...")
                strategy_output = step8_model.select_strategy(table_content, current_kc, history_context, round_idx=round_num)
                all_strategy_outputs.append(strategy_output)

                # 输出完整的策略 JSON
                print(f"\n   📄 Round {round_num + 1} 策略输出完整JSON:")
                print("   " + "-" * 50)
                strategy_json = json.dumps(strategy_output, ensure_ascii=False, indent=2)
                print("   " + strategy_json.replace("\n", "\n   "))
                print("   " + "-" * 50)

                if not strategy_output.get("_parse_ok", True):
                    correction_error = "Strategy parse failed"
                    print("   ❌ 策略解析失败，终止该样本评测")
                    break

                action = str(strategy_output.get("action", "CORRECT")).upper()
                strategy = str(strategy_output.get("strategy", "")).strip()
                chain_indices = strategy_output.get("chain_indices", [])

                if action == "STOP" or strategy.upper() == "STOP":
                    sample_result["model_stop"] = True
                    sample_result["stopped_correctly"] = current_r_rule >= STOP_THRESHOLD
                    break
                if action != "CORRECT":
                    correction_error = f"Unexpected action: {action}"
                    break
                if strategy not in corrector.strategy_to_file:
                    correction_error = f"Unknown strategy: {strategy}"
                    break

                kc_before = copy.deepcopy(current_kc)
                before_r_rule = current_r_rule
                corrected_kc, last_summary = corrector.correct(table_content, current_kc, strategy_output, last_summary)
                summary_text = (last_summary or "").strip()
                corrected_kc = normalize_kc_payload(corrected_kc)

                if summary_text.upper().startswith("TEACHER_STOP"):
                    sample_result["num_correction_rounds"] += 1
                    round_r_rules.append(current_r_rule)
                    round_kcs.append({"round": round_num + 1, "kc": current_kc, "r_rule": current_r_rule, "teacher_stop": True})
                    correction_summaries.append(summary_text)
                    append_history_entry(round_num, strategy, chain_indices, False, summary_text)
                    stagnant_rounds += 1
                    if stagnant_rounds >= MAX_STAGNANT_ROUNDS:
                        break
                    continue

                if summary_text.upper().startswith("INVALID_RESPONSE"):
                    sample_result["num_correction_rounds"] += 1
                    current_kc = corrected_kc
                    round_r_rules.append(current_r_rule)
                    round_kcs.append({"round": round_num + 1, "kc": current_kc, "r_rule": current_r_rule, "teacher_invalid_response": True})
                    correction_summaries.append(summary_text)
                    append_history_entry(round_num, strategy, chain_indices, False, summary_text)
                    stagnant_rounds += 1
                    if stagnant_rounds >= MAX_STAGNANT_ROUNDS:
                        break
                    continue

                sample_result["num_correction_rounds"] += 1
                kc_changed = not _kc_payload_equal(kc_before, corrected_kc)
                current_kc = corrected_kc
                current_r_rule = score_kc(current_kc, ground_truth)
                round_r_rules.append(current_r_rule)
                round_kcs.append({"round": round_num + 1, "kc": current_kc, "r_rule": current_r_rule})
                correction_summaries.append(summary_text)

                # 输出修正后的 R-rule 变化
                delta = current_r_rule - before_r_rule
                print(f"   📊 修正后R-rule: {current_r_rule:.4f} (Δ={delta:+.4f})")

                if current_r_rule < before_r_rule - DEGRADE_ROLLBACK:
                    print(
                        f"   ⚠️ 退化过大 {before_r_rule:.4f} -> {current_r_rule:.4f}，已回滚并继续修正"
                    )
                    current_kc = kc_before
                    current_r_rule = before_r_rule
                    round_r_rules[-1] = current_r_rule
                    round_kcs[-1] = {"round": round_num + 1, "kc": current_kc, "r_rule": current_r_rule, "rolled_back": True}
                    rollback_note = (
                        f"[ROLLBACK] 上轮策略 {strategy} 退化 {before_r_rule:.4f} -> {current_r_rule:.4f}，"
                        "已回滚为之前的 KC，当前轮仍可继续修正。"
                    )
                    correction_summaries[-1] = rollback_note
                    last_summary = rollback_note
                    append_history_entry(round_num, strategy, chain_indices, False, rollback_note)
                    stagnant_rounds = 0
                    continue

                stagnant_rounds = stagnant_rounds + 1 if abs(delta) < NO_PROGRESS_EPS else 0
                if stagnant_rounds >= MAX_STAGNANT_ROUNDS:
                    break
                if current_r_rule > best_r_rule:
                    best_r_rule = current_r_rule
                    best_kc = current_kc
                    best_round = round_num + 1

                append_history_entry(round_num, strategy, chain_indices, kc_changed, last_summary)
                if current_r_rule >= STOP_THRESHOLD:
                    print(f"   ✅ 修正后R-rule={current_r_rule:.4f} >= {STOP_THRESHOLD:.2f}，评估侧自动停止")
                    sample_result["auto_stop"] = True
                    sample_result["stopped_correctly"] = True
                    break

            sample_result["round_kcs"] = round_kcs
            sample_result["round_r_rules"] = round_r_rules
            sample_result["strategy_outputs"] = all_strategy_outputs
            sample_result["correction_summaries"] = correction_summaries
            sample_result["best_round"] = best_round
            sample_result["final_kc"] = best_kc
            sample_result["final_r_rule"] = best_r_rule
            sample_result["improvement"] = best_r_rule - sample_result["initial_r_rule"]
            sample_result["target_reached"] = best_r_rule >= STOP_THRESHOLD

            # 输出最终汇总信息
            print(f"\n   📈 各轮R-rule: {[f'{r:.4f}' for r in round_r_rules]}")
            print(f"   🏆 最佳结果: Round {best_round}, R-rule={best_r_rule:.4f}")
            print(f"   初始R-rule: {sample_result['initial_r_rule']:.4f}")
            print(f"   最终R-rule: {best_r_rule:.4f}")
            print(f"   提升: {sample_result['improvement']:.4f}")
            print(f"   修正轮数: {sample_result['num_correction_rounds']}")

            # 计算策略解析成功率
            total_rounds = len(all_strategy_outputs)
            parse_success = sum(1 for s in all_strategy_outputs if s.get("_parse_ok", True))
            parse_rate = (parse_success / total_rounds * 100) if total_rounds > 0 else 0
            print(f"   策略解析成功率: {parse_rate:.2f}%")

            # 计算策略有效率（KC有变化且R-rule提升或不下降）
            if total_rounds > 0:
                effective = 0
                for i, sr in enumerate(round_r_rules[1:], 1):
                    prev = round_r_rules[i-1]
                    if abs(sr - prev) < NO_PROGRESS_EPS or sr >= prev:
                        effective += 1
                effective_rate = (effective / total_rounds * 100) if total_rounds > 0 else 0
                print(f"   策略有效率: {effective_rate:.2f}%")

            print(f"   停止正确: {sample_result['stopped_correctly']}")

            if correction_error:
                sample_result["error"] = correction_error
                results["stats"]["error"] += 1
            else:
                sample_result["status"] = "success"
                results["stats"]["success"] += 1

            results["stats"]["rounds_total"] += sample_result["num_correction_rounds"]
        except Exception as e:
            sample_result["error"] = str(e)
            results["stats"]["error"] += 1

        results["samples"].append(sample_result)
        if idx % 10 == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n重新生成+修正完成!\n  输出文件: {output_file}")


if __name__ == "__main__":
    main()
