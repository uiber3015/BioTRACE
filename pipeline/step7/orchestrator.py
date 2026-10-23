"""
Step7 COT 修复 Orchestrator

完整流程：
1. 加载 SFT1 checkpoint（双卡）
2. 使用 SFT prompt 生成初始 knowledge_mining
3. 计算 r_rule
4. 循环（最多6轮）：
   - Student 选择策略
   - Teacher (deepseek-v3) 执行修正
   - 计算 r_rule，记录轨迹
   - 如果输出 STOP 则退出
5. 保存完整轨迹
"""
import os
import json
import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

# 导入 R_rule 评估器
from step7.r_rule_evaluator import compute_r_rule

# ============== 配置 ==============
@dataclass
class Config:
    # SFT1 模型配置
    sft1_model_path: str = "/home/new_disk/yjw/checkpoints/qwen3_sft1_openr1_20260124_135733"
    sft1_checkpoint_path: str = ""  # 如果有具体 checkpoint，填这里
    
    # DeepSeek Teacher 配置
    deepseek_api_url: str = os.environ.get("LLM_API_URL", "")
    deepseek_api_key: str = ""  # set via env DEEPSEEK_API_KEY or CLI
    deepseek_model: str = "DeepSeek-V3.2"
    
    # 数据配置
    sft2_data_dir: str = "step6/SFT2"
    output_dir: str = "output"
    trajectory_dir: str = "trajectories"
    
    # 流程配置
    max_rounds: int = 6
    delta: float = 0.01  # 接受阈值
    
    # 提示词路径
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    student_generate_prompt_path: str = "student_prompt_generate.txt"
    student_select_prompt_path: str = "student_prompt_select.txt"
    teacher_prompts_dir: str = "teacher_prompts"

    def __post_init__(self) -> None:
        base = self.base_dir
        if not Path(self.sft2_data_dir).is_absolute():
            self.sft2_data_dir = str((base.parent / self.sft2_data_dir).resolve())
        if not Path(self.output_dir).is_absolute():
            self.output_dir = str((base / self.output_dir).resolve())
        if not Path(self.trajectory_dir).is_absolute():
            self.trajectory_dir = str((base / self.trajectory_dir).resolve())
        if not Path(self.student_generate_prompt_path).is_absolute():
            self.student_generate_prompt_path = str((base / self.student_generate_prompt_path).resolve())
        if not Path(self.student_select_prompt_path).is_absolute():
            self.student_select_prompt_path = str((base / self.student_select_prompt_path).resolve())
        if not Path(self.teacher_prompts_dir).is_absolute():
            self.teacher_prompts_dir = str((base / self.teacher_prompts_dir).resolve())


# ============== 提示词加载 ==============
def load_prompt(path: str) -> str:
    """加载提示词文件"""
    prompt_path = Path(path)
    if not prompt_path.is_absolute():
        prompt_path = Path(__file__).resolve().parent / prompt_path
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_teacher_prompt_path(op_type: str, op_subtype: str) -> str:
    """获取对应策略的 Teacher Prompt 路径"""
    prompt_files = {
        ("Add", "condition_kv"): "teacher_add_condition_kv.txt",
        ("Add", "entity_value"): "teacher_add_entity_value.txt",
        ("Add", "new_chain"): "teacher_add_new_chain.txt",
        ("Modify", "condition_kv"): "teacher_modify_condition_kv.txt",
        ("Modify", "result"): "teacher_modify_result.txt",
        ("Modify", "entity_value"): "teacher_modify_entity_value.txt",
        ("Delete", "chain"): "teacher_delete_chain.txt",
        ("Delete", "condition_kv"): "teacher_delete_condition_kv.txt",
        ("Delete", "entity_value"): "teacher_delete_entity_value.txt",
        ("Delete", "result"): "teacher_delete_result.txt",
    }
    filename = prompt_files.get((op_type, op_subtype))
    if filename is None:
        raise ValueError(f"Unknown strategy: {op_type}/{op_subtype}")
    base_dir = Path(__file__).resolve().parent
    return str(base_dir / Config.teacher_prompts_dir / filename)


# ============== 与 SFT 训练一致的提示词 ==============
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


def build_messages_for_generate(table_content: str):
    """构建与 SFT 训练一致的消息格式（用于初始生成）"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{INPUT_PROMPT}\n\n<table_content>\n{table_content}\n</table_content>",
        },
    ]


# ============== SFT1 模型调用 ==============
class SFT1Model:
    """SFT1 模型（用于初始生成和策略选择）"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        # 加载策略选择提示词（用于 select_strategy）
        self.select_prompt_template = load_prompt(config.student_select_prompt_path)
    
    def load(self):
        """加载模型（单卡）"""
        print(f"Loading SFT1 model from {self.config.sft1_model_path}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.sft1_model_path,
                trust_remote_code=True,
                fix_mistral_regex=True
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.sft1_model_path, trust_remote_code=True)
        
        if self.config.sft1_checkpoint_path and os.path.exists(self.config.sft1_checkpoint_path):
            model_path = self.config.sft1_checkpoint_path
        else:
            model_path = self.config.sft1_model_path
        
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        return self
    
    def generate_initial(self, table_content: str) -> str:
        """使用与 SFT 训练一致的 chat template 格式生成初始 knowledge_mining"""
        # 构建与 SFT 训练一致的消息格式
        messages = build_messages_for_generate(table_content)
        
        # 使用 apply_chat_template（与 SFT 训练一致）
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=30000)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8192,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 只解码新生成的部分
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        # 提取 JSON 部分
        return self._extract_json(generated_text.strip(), expected_keys=["knowledge_mining"])
    
    def select_strategy(self, table_content: str, current_generated_json: str) -> Dict[str, Any]:
        """Student 策略选择（使用 chat template 格式）"""
        user_content = self._build_select_prompt(table_content, current_generated_json)
        
        # 使用 chat template 格式
        messages = [
            {"role": "system", "content": "你是知识链修正策略选择器（Student-Selector），负责识别当前输出的问题并选择修正策略。"},
            {"role": "user", "content": user_content},
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=30000)
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 只解码新生成的部分
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        json_str = self._extract_json(generated_text.strip(), expected_keys=["operation"])
        
        try:
            payload = json.loads(json_str)
            op = payload.get("operation")
            if isinstance(op, dict):
                spec = op.get("spec")
                if isinstance(spec, dict):
                    if "reason" not in op and "reason" in spec:
                        op["reason"] = spec.pop("reason")
                    if "confidence" not in op and "confidence" in spec:
                        op["confidence"] = spec.pop("confidence")
            return self._clip_strategy_output(payload)
        except json.JSONDecodeError as e:
            print(f"Failed to parse strategy selection: {e}")
            print(f"Raw response: {generated_text}")
            return {"error": str(e)}

    def select_strategy_forced(self, table_content: str, current_generated_json: str, summary: str) -> Dict[str, Any]:
        """强制换策略：在多轮无提升后触发"""
        user_content = self._build_select_prompt(
            table_content,
            current_generated_json,
            extra_hint=(
                "你已经连续多轮选择相同策略且 R_rule 没有提升。"
                "请务必更换 op_type/op_subtype（不要再用相同策略），并严格控制输出规模。"
                "仅允许最多 3 条链、每条最多 3 个 kv。\n"
                f"当前简要状态：{summary}"
            ),
        )

        messages = [
            {"role": "system", "content": "你是知识链修正策略选择器（Student-Selector），负责识别当前输出的问题并选择修正策略。"},
            {"role": "user", "content": user_content},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        json_str = self._extract_json(generated_text.strip(), expected_keys=["operation"])
        try:
            payload = json.loads(json_str)
            op = payload.get("operation")
            if isinstance(op, dict):
                spec = op.get("spec")
                if isinstance(spec, dict):
                    if "reason" not in op and "reason" in spec:
                        op["reason"] = spec.pop("reason")
                    if "confidence" not in op and "confidence" in spec:
                        op["confidence"] = spec.pop("confidence")
            return self._clip_strategy_output(payload)
        except json.JSONDecodeError as e:
            print(f"Failed to parse forced strategy selection: {e}")
            print(f"Raw response: {generated_text}")
            return {"error": str(e)}

    def _build_select_prompt(self, table_content: str, current_generated_json: str, extra_hint: str = "") -> str:
        user_content = self.select_prompt_template.replace("{TABLE_CONTENT}", table_content)
        user_content = user_content.replace("{CURRENT_GENERATED_JSON}", current_generated_json)
        if extra_hint:
            user_content = user_content + "\n\n[额外提示]\n" + extra_hint
        return user_content
    
    def _extract_json(self, text: str, expected_keys: Optional[List[str]] = None) -> str:
        """从模型输出中提取 JSON"""
        expected_keys = expected_keys or []
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        candidates = []
        stack = []
        start_idx = None
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if not stack:
                    start_idx = i
                stack.append(ch)
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack and start_idx is not None:
                        candidates.append(cleaned[start_idx:i + 1])
                        start_idx = None

        for cand in candidates:
            repaired = None
            try:
                payload = json.loads(cand)
            except Exception:
                repaired = self._repair_json_fragment(cand)
                try:
                    payload = json.loads(repaired)
                except Exception:
                    continue
            if all(k in payload for k in expected_keys):
                return repaired or cand

        for cand in candidates:
            try:
                json.loads(cand)
                return cand
            except Exception:
                repaired = self._repair_json_fragment(cand)
                try:
                    json.loads(repaired)
                    return repaired
                except Exception:
                    continue

        start = cleaned.find("{")
        if start != -1:
            fragment = cleaned[start:]
            fragment = self._repair_json_fragment(fragment)
            try:
                json.loads(fragment)
                return fragment
            except Exception:
                pass

        return cleaned

    def _repair_json_fragment(self, fragment: str) -> str:
        stack = []
        for ch in fragment:
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()
        if stack:
            fragment = fragment + "".join(reversed(stack))
        return fragment

    def _clip_strategy_output(self, payload: Dict[str, Any], max_chains: int = 3, max_kvs: int = 3) -> Dict[str, Any]:
        op = payload.get("operation")
        if not isinstance(op, dict):
            return payload
        spec = op.get("spec")
        if not isinstance(spec, dict):
            return payload

        op_type = op.get("op_type")
        op_subtype = op.get("op_subtype")

        def clip_list(obj: Dict[str, Any], key: str, limit: int) -> None:
            if key in obj and isinstance(obj[key], list):
                obj[key] = obj[key][:limit]

        if op_type == "Add" and op_subtype == "condition_kv":
            clip_list(spec, "adds", max_chains)
            for item in spec.get("adds", []):
                if isinstance(item, dict):
                    clip_list(item, "kvs", max_kvs)
            chain_indices = [item.get("chain_idx") for item in spec.get("adds", []) if isinstance(item, dict)]
            chain_indices = [idx for idx in chain_indices if idx is not None]
            if chain_indices:
                op["target_chain_indices"] = chain_indices[:max_chains]
            elif isinstance(op.get("target_chain_indices"), list):
                op["target_chain_indices"] = op["target_chain_indices"][:max_chains]
        elif op_type == "Add" and op_subtype == "entity_value":
            clip_list(spec, "adds", max_chains)
            for item in spec.get("adds", []):
                if isinstance(item, dict):
                    clip_list(item, "values", max_kvs)
        elif op_type == "Add" and op_subtype == "new_chain":
            clip_list(spec, "new_chains", max_chains)
        elif op_type == "Modify" and op_subtype in {"condition_kv", "result", "entity_value"}:
            clip_list(spec, "edits", max_chains)
        elif op_type == "Delete" and op_subtype == "chain":
            clip_list(spec, "chain_indices", max_chains)
        elif op_type == "Delete" and op_subtype == "condition_kv":
            clip_list(spec, "deletes", max_chains)
            for item in spec.get("deletes", []):
                if isinstance(item, dict):
                    clip_list(item, "kvs", max_kvs)
        elif op_type == "Delete" and op_subtype == "entity_value":
            clip_list(spec, "deletes", max_chains)
            for item in spec.get("deletes", []):
                if isinstance(item, dict):
                    clip_list(item, "values", max_kvs)
        elif op_type == "Delete" and op_subtype == "result":
            clip_list(spec, "chain_indices", max_chains)
        return payload


# ============== DeepSeek Teacher 调用 ==============
class DeepSeekTeacher:
    """DeepSeek Teacher（执行修正）"""
    
    def __init__(self, config: Config):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.deepseek_api_key}",
            "Content-Type": "application/json"
        }
    
    def execute(self, 
                table_content: str,
                current_generated_json: str,
                student_operation_json: Dict[str, Any],
                teacher_prompt_path: str) -> Tuple[str, bool]:
        """
        执行 Teacher 修正
        
        Returns:
            (修正后的 knowledge_mining JSON, 是否为 STOP)
        """
        # 加载 Teacher Prompt
        with open(teacher_prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        
        prompt = prompt_template.replace("{TABLE_CONTENT}", table_content)
        prompt = prompt.replace("{CURRENT_GENERATED_JSON}", current_generated_json)
        prompt = prompt.replace("{STUDENT_OPERATION_JSON}", json.dumps(student_operation_json, ensure_ascii=False))
        student_reason = student_operation_json.get("operation", {}).get("reason", "")
        if "{STUDENT_REASON}" in prompt:
            prompt = prompt.replace("{STUDENT_REASON}", student_reason)
        elif student_reason:
            prompt = prompt + f"\n\n[Student Reason]\n{student_reason}"
        
        payload = {
            "model": self.config.deepseek_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.1
        }
        
        for retry in range(3):
            try:
                response = requests.post(
                    self.config.deepseek_api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 检查是否 STOP
                if content.strip() == "STOP":
                    return content, True
                
                # 提取 JSON
                json_str = self._extract_json(content)
                return json_str, False
                
            except Exception as e:
                print(f"API call failed (retry {retry+1}/3): {e}")
                time.sleep(2 ** retry)  # 指数退避
        
        raise Exception("DeepSeek API call failed after 3 retries")
    
    def _extract_json(self, text: str) -> str:
        """从模型输出中提取 JSON"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            candidate = cleaned[start:end]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                return candidate
        return cleaned


# ============== 轨迹记录 ==============
@dataclass
class RoundTrajectory:
    """单轮轨迹记录"""
    round_id: int
    stage: str  # "pre" / "post" / "student" / "teacher"
    
    # Student 输出
    student_operation: Optional[Dict[str, Any]] = None
    
    # Teacher 输出
    teacher_output: Optional[Dict[str, Any]] = None
    is_stop: bool = False
    
    # 分数
    scores: Optional[Dict[str, float]] = None
    scores_before: Optional[Dict[str, float]] = None
    scores_after: Optional[Dict[str, float]] = None

    # 参考答案（SFT-2 ground truth）
    gold_chains: Optional[List[Dict[str, Any]]] = None
    
    # 状态
    accepted: bool = False
    reason: str = ""
    failure_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrajectoryRecorder:
    """轨迹记录器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectories: List[Dict[str, Any]] = []
    
    def add_round(self, trajectory: RoundTrajectory):
        self.trajectories.append(trajectory.to_dict())
    
    def save(self, sample_id: str):
        """保存单个样本的轨迹"""
        output_path = self.output_dir / f"{sample_id}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for traj in self.trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        print(f"Trajectory saved to {output_path}")
    
    def save_all(self, output_path: str):
        """保存所有轨迹（汇总）"""
        with open(output_path, "w", encoding="utf-8") as f:
            for traj in self.trajectories:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")


# ============== 主 Orchestrator ==============
class Step7Orchestrator:
    """Step7 COT 修复 Orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sft_model = SFT1Model(config)
        self.teacher = DeepSeekTeacher(config)
        self.trajectory_recorder = TrajectoryRecorder(config.trajectory_dir)
    
    def process_sample(self, 
                       sample_id: str,
                       table_content: str,
                       ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本
        
        Returns:
            完整结果包含轨迹
        """
        print(f"\n{'='*60}")
        print(f"Processing sample: {sample_id}")
        print(f"{'='*60}")
        
        # 初始化
        current_generated_json = None
        best_generated_json = None
        best_r_rule = 0.0
        rounds_data = []
        
        # Round 0: 初始生成
        print(f"\n[Round 0] Initial generation...")
        try:
            initial_json_str = self.sft_model.generate_initial(table_content)
            current_generated_json = json.loads(initial_json_str)
            print(f"Initial generated: {len(current_generated_json.get('knowledge_mining', []))} chains")
        except Exception as e:
            print(f"Initial generation failed: {e}")
            current_generated_json = {"knowledge_mining": []}
        
        # 评估初始分数
        sample_for_eval = {
            "generated_json": current_generated_json,
            "ground_truth": ground_truth
        }
        eval_result = compute_r_rule(sample_for_eval)
        R_before = eval_result["R_rule"]
        S_structure = eval_result["S_structure"]
        S_content = eval_result["S_content"]
        
        print(f"Initial R_rule: {R_before:.4f} (S_structure: {S_structure:.4f}, S_content: {S_content:.4f})")
        
        # 记录 Round 0
        gold_chains = ground_truth.get("knowledge_mining", []) if isinstance(ground_truth, dict) else []

        pre_trajectory = RoundTrajectory(
            round_id=0,
            stage="pre",
            scores={
                "R_rule": R_before,
                "S_structure": S_structure,
                "S_content": S_content
            },
            gold_chains=gold_chains
        )
        self.trajectory_recorder.add_round(pre_trajectory)
        
        best_generated_json = json.loads(json.dumps(current_generated_json))
        best_r_rule = R_before

        last_op_type = None
        last_op_subtype = None
        stagnant_repeats = 0
        
        # COT 修正循环（最多6轮）
        for round_id in range(1, self.config.max_rounds + 1):
            print(f"\n[Round {round_id}] Strategy selection and correction...")
            pre_best_r_rule = best_r_rule
            
            # Step 1: Student 选择策略
            current_json_str = json.dumps(current_generated_json, ensure_ascii=False)
            
            try:
                if stagnant_repeats >= 3:
                    summary = (
                        f"best_r_rule={best_r_rule:.4f}, current_chains={len(current_generated_json.get('knowledge_mining', []))}, "
                        f"last_strategy={last_op_type}/{last_op_subtype}"
                    )
                    strategy_result = self.sft_model.select_strategy_forced(
                        table_content, current_json_str, summary
                    )
                    stagnant_repeats = 0
                else:
                    strategy_result = self.sft_model.select_strategy(table_content, current_json_str)
                
                if "error" in strategy_result:
                    raise Exception(f"Strategy selection error: {strategy_result['error']}")
                
                op = strategy_result.get("operation", {})
                op_type = op.get("op_type")
                op_subtype = op.get("op_subtype")
                
                print(f"  Student selected: {op_type}/{op_subtype}")
                print(f"  Reason: {op.get('reason', 'N/A')}")
                print(f"  Confidence: {op.get('confidence', 0)}")
                
            except Exception as e:
                print(f"Strategy selection failed: {e}")
                strategy_result = {"error": str(e)}
                op = {}
                op_type = op_subtype = None
            
            # 记录 Student 轨迹
            student_trajectory = RoundTrajectory(
                round_id=round_id,
                stage="student",
                student_operation=strategy_result,
                gold_chains=gold_chains
            )
            self.trajectory_recorder.add_round(student_trajectory)
            
            # Step 2: Teacher 执行修正
            is_stop = False
            teacher_output = None
            failure_reason = ""
            
            if op_type and op_subtype:
                try:
                    teacher_prompt_path = get_teacher_prompt_path(op_type, op_subtype)
                    print(f"  Loading teacher prompt: {teacher_prompt_path}")
                    
                    teacher_json_str, is_stop = self.teacher.execute(
                        table_content=table_content,
                        current_generated_json=current_json_str,
                        student_operation_json=strategy_result,
                        teacher_prompt_path=teacher_prompt_path
                    )
                    
                    if is_stop:
                        print(f"  Teacher returned STOP - exiting correction loop")
                        failure_reason = "STOP"
                    else:
                        teacher_output = json.loads(teacher_json_str)
                        print(f"  Teacher output: {len(teacher_output.get('knowledge_mining', []))} chains")
                        
                except Exception as e:
                    print(f"Teacher execution failed: {e}")
                    failure_reason = str(e)
                    teacher_output = None
            else:
                failure_reason = "No valid strategy selected"
            
            # Step 3: 评估修正后的分数
            scores_after = None
            accepted = False
            reason = ""
            
            before_scores = {
                "R_rule": best_r_rule,
                "S_structure": S_structure,
                "S_content": S_content
            }

            if teacher_output and not is_stop:
                # 计算修正后的 R_rule
                sample_after = {
                    "generated_json": teacher_output,
                    "ground_truth": ground_truth
                }
                eval_after = compute_r_rule(sample_after)
                R_after = eval_after["R_rule"]
                S_structure_after = eval_after["S_structure"]
                S_content_after = eval_after["S_content"]
                
                scores_after = {
                    "R_rule": R_after,
                    "S_structure": S_structure_after,
                    "S_content": S_content_after
                }
                
                print(f"  After R_rule: {R_after:.4f} (S_structure: {S_structure_after:.4f}, S_content: {S_content_after:.4f})")
                
                # 接受判断
                if R_after >= best_r_rule + self.config.delta:
                    accepted = True
                    reason = f"R improved from {best_r_rule:.4f} to {R_after:.4f}"
                    best_r_rule = R_after
                    best_generated_json = json.loads(json.dumps(teacher_output))
                else:
                    accepted = False
                    reason = f"R did not improve enough ({R_after:.4f} < {best_r_rule:.4f} + {self.config.delta})"
            else:
                R_after = 0.0
                scores_after = {"R_rule": 0.0, "S_structure": 0.0, "S_content": 0.0}
                accepted = False
                reason = f"Teacher failed or STOP: {failure_reason}"
            
            # 记录 Teacher 轨迹
            teacher_trajectory = RoundTrajectory(
                round_id=round_id,
                stage="teacher",
                teacher_output=teacher_output,
                is_stop=is_stop,
                scores_before=before_scores,
                scores_after=scores_after,
                accepted=accepted,
                reason=reason,
                failure_reason=failure_reason,
                gold_chains=gold_chains
            )
            self.trajectory_recorder.add_round(teacher_trajectory)

            # 更新重复策略计数（仅在本轮未提升时累积）
            if op_type and op_subtype:
                no_improve = best_r_rule <= pre_best_r_rule
                if no_improve and op_type == last_op_type and op_subtype == last_op_subtype:
                    stagnant_repeats += 1
                elif no_improve:
                    stagnant_repeats = 0
                else:
                    stagnant_repeats = 0
                last_op_type = op_type
                last_op_subtype = op_subtype
            
            # 更新状态
            if accepted:
                current_generated_json = teacher_output
                S_structure = scores_after["S_structure"]
                S_content = scores_after["S_content"]
            
            # 如果 STOP，提前退出
            if is_stop:
                print(f"  Exiting loop due to STOP")
                break
        
        # 保存轨迹
        self.trajectory_recorder.save(sample_id)
        
        # 记录最终结果
        final_result = {
            "sample_id": sample_id,
            "table_content": table_content,
            "ground_truth": ground_truth,
            "best_generated_json": best_generated_json,
            "best_r_rule": best_r_rule,
            "trajectory": self.trajectory_recorder.trajectories
        }
        
        print(f"\n[Result] Best R_rule: {best_r_rule:.4f}")
        print(f"Trajectory saved with {len(self.trajectory_recorder.trajectories)} entries")
        
        # 清空当前样本的轨迹，准备下一个
        self.trajectory_recorder.trajectories = []
        
        return final_result
    
    def process_all_samples(self, file_ids: List[str] = None):
        """处理所有样本"""
        data_dir = Path(self.config.sft2_data_dir)
        
        if file_ids is None:
            # 处理所有 JSON 文件
            file_ids = [f.stem for f in data_dir.glob("*.json") if f.stem.isdigit()]
        
        print(f"Will process {len(file_ids)} samples")
        
        results = []
        for file_id in file_ids:
            sample_path = data_dir / f"{file_id}.json"
            if not sample_path.exists():
                print(f"Sample {file_id} not found, skipping")
                continue
            
            try:
                with open(sample_path, "r", encoding="utf-8") as f:
                    sample_data = json.load(f)
                
                table_content = sample_data.get("table_content", "")
                ground_truth = {"knowledge_mining": sample_data.get("knowledge_mining", [])}
                
                result = self.process_sample(
                    sample_id=file_id,
                    table_content=table_content,
                    ground_truth=ground_truth
                )
                results.append(result)
                
            except Exception as e:
                print(f"Failed to process sample {file_id}: {e}")
                continue
        
        # 保存汇总结果
        summary_path = Path(self.config.output_dir) / "all_results.jsonl"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"\n{'='*60}")
        print(f"Processing complete! Results saved to {summary_path}")
        print(f"Total samples processed: {len(results)}")
        print(f"{'='*60}")
        
        return results


# ============== 主入口 ==============
def main():
    config = Config()
    
    # 确保输出目录存在
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.trajectory_dir).mkdir(parents=True, exist_ok=True)
    
    # 初始化 Orchestrator
    orchestrator = Step7Orchestrator(config)
    
    # 加载 SFT1 模型
    orchestrator.sft_model.load()
    
    # 处理所有样本（可以指定部分样本进行测试）
    # 先测试几个样本
    test_samples = ["14", "100", "1000"]  # 先测试这3个
    
    orchestrator.process_all_samples(file_ids=test_samples)


if __name__ == "__main__":
    main()

