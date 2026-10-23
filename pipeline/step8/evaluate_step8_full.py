#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step8 完整评测脚本

评测流程（与训练一致）：
1. 初始KC生成：使用step6提示词
2. 策略选择：使用step8训练时的提示词（strategy_selection.txt）
3. DeepSeek Teacher执行修正：使用step8自己的teacher prompts（prompts/teacher/）
4. R_rule评估：用外部评估器计算

用法：
python evaluate_step8_full.py --model_path <step8模型路径> --test_samples <数量>
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "step7"))

# 导入 Step7 的评估器
from step7.r_rule_evaluator_final import compute_r_rule


# ============== Step6 提示词（初始KC生成用，与训练一致） ==============
SYSTEM_PROMPT_STEP6 = (
    "你是一名基因组学知识抽取助手，任务是从表格内容(table_content)中生成知识挖掘结果。"
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


# ============== Step8 策略选择提示词（与训练一致） ==============
def load_step8_strategy_prompt() -> str:
    """加载step8训练时使用的策略选择提示词"""
    prompt_path = Path(__file__).parent / "prompts" / "strategy_selection.txt"
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


# ============== Step8 模型 ==============
class Step8Model:
    """Step8 模型"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        # 加载策略选择提示词（与训练一致）
        self.strategy_prompt_template = load_step8_strategy_prompt()
    
    def load(self):
        """加载模型"""
        print(f"📁 Loading Step8 model from {self.model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded on {self.device}")
        return self
    
    def generate_initial_kc(self, table_content: str) -> Dict:
        """使用step6提示词生成初始知识链"""
        # 使用与step6训练一致的格式
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_STEP6},
            {"role": "user", "content": f"{INPUT_PROMPT_STEP6}\n\n<table_content>\n{table_content}\n</table_content>"},
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=30000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # 提取JSON
        try:
            json_str = self._extract_json(generated_text)
            return json.loads(json_str)
        except:
            return {"knowledge_mining": []}
    
    def select_strategy(
        self, 
        table_content: str, 
        current_kc: Dict,
        history_context: str = "",
        previous_summary: str = ""
    ) -> Dict:
        """使用step8训练时的提示词选择修正策略"""
        
        # 构建提示词（与训练一致）
        prompt = self.strategy_prompt_template
        prompt = prompt.replace("{table_content}", table_content)
        prompt = prompt.replace("{current_knowledge_chain}", json.dumps(current_kc, ensure_ascii=False))
        
        # 处理历史上下文
        if history_context:
            prompt = prompt.replace("{previous_summary_block}", f"\n## 历史修正\n{history_context}")
        else:
            prompt = prompt.replace("{previous_summary_block}", "")
        
        if previous_summary:
            prompt = prompt.replace("{previous_summary_block}", f"\n## 上一轮修正总结\n{previous_summary}\n")
        
        # 使用Qwen的chat format
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=30000)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # 解析JSON（与训练输出格式一致）
        try:
            json_str = self._extract_json(generated_text)
            result = json.loads(json_str)
            
            # 确保字段完整
            if "action" not in result:
                result["action"] = "CORRECT"
            if "chain_indices" not in result:
                result["chain_indices"] = []
            if "reason" not in result:
                result["reason"] = ""
            
            return result
        except:
            return {
                "action": "ERROR",
                "strategy": "PARSE_ERROR",
                "chain_indices": [],
                "reason": f"解析失败: {generated_text[:200]}"
            }
    
    def _extract_json(self, text: str) -> str:
        """从模型输出中提取 JSON"""
        cleaned = text.strip()
        
        # 尝试找到 JSON 块
        if "```" in cleaned:
            blocks = cleaned.split("```")
            for block in blocks:
                if block.strip().startswith("{") or block.strip().startswith("["):
                    candidate = block.strip()
                    if candidate.startswith("json"):
                        candidate = candidate[4:].strip()
                    try:
                        json.loads(candidate)
                        return candidate
                    except:
                        pass
        
        # 尝试直接提取
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        
        if start != -1 and end > start:
            candidate = cleaned[start:end + 1]
            try:
                json.loads(candidate)
                return candidate
            except:
                pass
        
        return cleaned


# ============== DeepSeek Teacher 包装 ==============
class DeepSeekCorrector:
    """DeepSeek修正器（使用step8自己的teacher prompts）"""
    
    def __init__(self):
        # 使用step8自己的teacher prompts路径
        self.teacher_prompts_dir = Path(__file__).parent / "prompts" / "teacher"
        
        # DeepSeek API配置
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.model_name = "deepseek-chat"
        
        if not self.api_key:
            # 尝试从step7的config获取
            try:
                from step7.orchestrator import Config
                config = Config()
                self.api_url = config.deepseek_api_url
                self.api_key = config.deepseek_api_key
                self.model_name = config.deepseek_model
            except:
                pass
    
    def _load_teacher_prompt(self, strategy: str) -> str:
        """加载teacher prompt"""
        # 映射策略到prompt文件
        # Add/condition_kv -> teacher_add_condition_kv.txt
        # Modify/condition_kv -> teacher_modify_condition_kv.txt
        # Delete/chain -> teacher_delete_chain.txt
        # 等等
        
        strategy_to_file = {
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
        
        filename = strategy_to_file.get(strategy, f"teacher_{strategy.replace('/', '_')}.txt")
        prompt_path = self.teacher_prompts_dir / filename
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Teacher prompt not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _build_messages(self, prompt_template: str, table_content: str, 
                       current_kc: Dict, strategy_output: Dict, 
                       last_summary: str = "") -> List[Dict]:
        """构建messages"""
        
        # 准备student_operation_json
        # step8模型输出: {"action": "CORRECT", "strategy": "Modify/condition_kv", "chain_indices": [0, 1]}
        # 如果没有reason字段，使用策略类型作为默认值
        
        chain_indices = strategy_output.get("chain_indices", [])
        reason = strategy_output.get("reason", "")  # 可能为空
        
        student_operation_json = {
            "strategy": strategy_output.get("strategy", ""),
            "chain_indices": chain_indices,
            "reason": reason
        }
        
        # 替换模板中的占位符
        prompt = prompt_template.replace("{TABLE_CONTENT}", table_content)
        prompt = prompt.replace("{CURRENT_GENERATED_JSON}", json.dumps(current_kc, ensure_ascii=False))
        prompt = prompt.replace("{STUDENT_OPERATION_JSON}", json.dumps(student_operation_json, ensure_ascii=False))
        prompt = prompt.replace("{STUDENT_REASON}", reason if reason else "（模型未提供理由，根据策略类型和表格内容自行判断）")
        
        if last_summary:
            prompt = prompt.replace("{LAST_CORRECTION_SUMMARY}", f"\n## 上一轮修正总结\n{last_summary}")
        else:
            prompt = prompt.replace("{LAST_CORRECTION_SUMMARY}", "")
        
        # 替换chain_indices占位符
        prompt = prompt.replace("{chain_indices}", str(chain_indices))
        prompt = prompt.replace("{学生新增的具体内容摘要}", f"使用策略 {strategy_output.get('strategy', '')} 对链 {chain_indices} 进行修正")
        prompt = prompt.replace("{学生修改的具体内容摘要}", f"使用策略 {strategy_output.get('strategy', '')} 对链 {chain_indices} 进行修正")
        prompt = prompt.replace("{删除了哪些链}", f"删除链 {chain_indices}")
        prompt = prompt.replace("{新增的实体类型和值}", f"对链 {chain_indices} 添加实体")
        prompt = prompt.replace("{删除了哪些类型}", f"删除链 {chain_indices} 的实体类型")
        prompt = prompt.replace("{删除了哪些实体值}", f"删除链 {chain_indices} 的实体值")
        prompt = prompt.replace("{旧result→新result}", f"修改链 {chain_indices} 的result")
        prompt = prompt.replace("{result置空}", f"置空链 {chain_indices} 的result")
        prompt = prompt.replace("{原类型→新类型}", f"修改链 {chain_indices} 的实体类型")
        prompt = prompt.replace("{旧值→新值}", f"修改链 {chain_indices} 的实体值")
        
        return [{"role": "user", "content": prompt}]
    
    def _call_deepseek_api(self, messages: List[Dict], max_retries: int = 3, backoff_factor: float = 2.0) -> str:
        """调用DeepSeek API，带重试机制"""
        import time
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4096
        }
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=data,
                    timeout=180
                )
                
                if response.status_code != 200:
                    raise Exception(f"API call failed: {response.text}")
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    print(f"[DeepSeek API] Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    print(f"[DeepSeek API] All {max_retries} attempts failed: {e}")
        
        raise last_exception
    
    def _extract_json_and_summary(self, response: str) -> Tuple[Optional[Dict], str]:
        """从响应中提取JSON和修正总结"""
        
        # 查找JSON部分
        json_start = response.find("{")
        json_end = response.rfind("}")
        
        corrected_kc = None
        summary = ""
        
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end + 1]
            try:
                corrected_kc = json.loads(json_str)
            except:
                pass
        
        # 查找修正总结
        if "【修正总结】" in response:
            summary_start = response.find("【修正总结】")
            summary = response[summary_start:]
            # 清理summary
            if "```" in summary:
                summary = summary.split("```")[0]
        
        return corrected_kc, summary
    
    def correct(
        self,
        table_content: str,
        current_kc: Dict,
        strategy_output: Dict,
        last_summary: str = ""
    ) -> Tuple[Optional[Dict], bool, str, str]:
        """
        执行修正
        
        Returns:
            corrected_kc: 修正后的KC，如果失败返回None
            is_stop: 是否为STOP
            error_msg: 错误信息
            summary: 修正总结
        """
        
        strategy = strategy_output.get("strategy", "")
        
        if not strategy:
            return None, False, "策略为空", ""
        
        # 如果是STOP
        if strategy_output.get("action") == "STOP":
            return None, True, "模型选择STOP", ""
        
        # 加载对应的teacher prompt
        try:
            prompt_template = self._load_teacher_prompt(strategy)
        except FileNotFoundError as e:
            return None, False, str(e), ""
        
        # 构建messages
        messages = self._build_messages(
            prompt_template, table_content, current_kc, 
            strategy_output, last_summary
        )
        
        # 调用API
        try:
            response = self._call_deepseek_api(messages)
        except Exception as e:
            return None, False, f"API调用失败: {str(e)}", ""
        
        # 检查是否是STOP
        if response.strip().upper() == "STOP":
            return None, True, "DeepSeek返回STOP", ""
        
        # 提取JSON和总结
        corrected_kc, summary = self._extract_json_and_summary(response)
        
        if corrected_kc is None:
            return None, False, "无法解析修正结果", summary
        
        return corrected_kc, False, "成功", summary


# ============== Step8 评测 ==============
class Step8Evaluator:
    """Step8 完整评测"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = Step8Model(model_path)
        self.corrector = DeepSeekCorrector()
        self.results = []  # 存储所有样本结果
        self.output_path = None
    
    def load_model(self):
        """加载模型"""
        self.model.load()
    
    def save_results_incremental(self):
        """增量保存结果"""
        if self.output_path and self.results:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            print(f"  💾 增量保存结果到: {self.output_path}")
    
    def evaluate_sample(
        self,
        sample_id: str,
        table_content: str,
        ground_truth: Dict[str, Any],
        max_rounds: int = 6
    ) -> Dict[str, Any]:
        """评测单个样本"""
        
        print(f"\n{'='*60}")
        print(f"📋 评测样本: {sample_id}")
        print(f"{'='*60}")
        
        # Round 0: 生成初始KC（使用step6提示词）
        print(f"\n[Round 0] 生成初始知识链...")
        try:
            current_kc = self.model.generate_initial_kc(table_content)
            print(f"  初始生成: {len(current_kc.get('knowledge_mining', []))} 条链")
        except Exception as e:
            print(f"  ⚠️ 初始生成失败: {e}")
            current_kc = {"knowledge_mining": []}
        
        # 评估初始分数
        sample_for_eval = {
            "generated_json": current_kc,
            "ground_truth": ground_truth
        }
        eval_result = compute_r_rule(sample_for_eval)
        R_before = eval_result["R_rule"]
        initial_kc_snapshot = json.loads(json.dumps(current_kc))
        initial_r_rule = R_before
        
        print(f"  初始 R_rule: {R_before:.4f}")
        
        best_kc = json.loads(json.dumps(current_kc))
        best_r_rule = R_before
        
        # 多轮修正
        history = []
        last_summary = ""
        
        for round_id in range(1, max_rounds + 1):
            print(f"\n[Round {round_id}] 策略选择...")
            
            # Step8模型选择策略（使用step8提示词）
            history_str = "\n".join([
                f"- Round {h['round']}: {h.get('strategy', 'N/A')}, indices: {h.get('chain_indices', [])}, accepted: {h.get('accepted', False)}"
                for h in history[-3:]
            ])
            
            strategy_output = self.model.select_strategy(
                table_content=table_content,
                current_kc=current_kc,
                history_context=history_str,
                previous_summary=last_summary
            )
            
            action = strategy_output.get("action", "CORRECT")
            strategy = strategy_output.get("strategy", "")
            chain_indices = strategy_output.get("chain_indices", [])
            
            print(f"  策略: {action}/{strategy}")
            print(f"  chain_indices: {chain_indices}")
            
            # 如果是STOP，退出
            if action == "STOP":
                print(f"  → 模型选择STOP，退出修正")
                break
            
            # DeepSeek修正
            print(f"  调用DeepSeek修正...")
            corrected_kc, is_stop, error_msg, summary = self.corrector.correct(
                table_content=table_content,
                current_kc=current_kc,
                strategy_output=strategy_output,
                last_summary=last_summary
            )
            
            if is_stop:
                print(f"  → 修正完成/STOP: {error_msg}")
                history.append({
                    "round": round_id,
                    "strategy": strategy,
                    "chain_indices": chain_indices,
                    "accepted": True,
                    "summary": summary if summary else error_msg
                })
                break
            
            if corrected_kc is None:
                print(f"  → 修正失败: {error_msg}")
                history.append({
                    "round": round_id,
                    "strategy": strategy,
                    "chain_indices": chain_indices,
                    "accepted": False,
                    "summary": f"修正失败: {error_msg}"
                })
                last_summary = summary if summary else error_msg
                continue
            
            # 评估修正后分数
            r_before_step = R_before
            eval_after = compute_r_rule({
                "generated_json": corrected_kc,
                "ground_truth": ground_truth
            })
            R_after = eval_after["R_rule"]
            
            print(f"  修正后 R_rule: {R_after:.4f}")
            
            # 统一 accept/reject 机制：与 Step9 一致，使用 rollback_threshold
            # 只有严重退化（delta < -0.3）才拒绝/回滚
            rollback_threshold = 0.3
            raw_delta = R_after - r_before_step

            if raw_delta < -rollback_threshold:
                print(f"  -> reject correction (severe degrade {raw_delta:+.4f})")
                accepted = False
                summary_text = f"{strategy} rejected (severe degrade)"
            else:
                print(f"  -> accept correction (delta: {raw_delta:+.4f})")
                accepted = True
                current_kc = corrected_kc
                R_before = R_after
                if R_after > best_r_rule:
                    best_kc = json.loads(json.dumps(corrected_kc))
                    best_r_rule = R_after
                summary_text = f"{strategy} correction accepted"
            
            # 记录历史（包含修正总结）
            history.append({
                "round": round_id,
                "strategy": strategy,
                "chain_indices": chain_indices,
                "r_rule_before": r_before_step,
                "r_rule_after": R_after,
                "accepted": accepted,
                "summary": summary if summary else summary_text
            })
            
            last_summary = summary if summary else summary_text
            
            # 如果连续3次相同策略无提升，退出
            if len(history) >= 3:
                recent = history[-3:]
                if all(h["strategy"] == strategy and not h["accepted"] for h in recent):
                    print(f"  → 连续3次相同策略无提升，退出")
                    break
        
        print(f"\n[结果] 最终 R_rule: {best_r_rule:.4f}")
        
        # 构建返回结果
        result = {
            "sample_id": sample_id,
            "initial_kc": initial_kc_snapshot,
            "initial_r_rule": initial_r_rule,
            "final_kc": best_kc,
            "final_r_rule": best_r_rule,
            "r_rule_improvement": best_r_rule - initial_r_rule,
            "num_rounds": len(history),
            "history": history
        }
        
        # 添加到结果列表
        self.results.append(result)
        
        # 增量保存
        self.save_results_incremental()
        
        return result


# ============== 主函数 ==============
def main():
    parser = argparse.ArgumentParser(description="Step8 完整评测")
    parser.add_argument("--model_path", type=str, 
                       default="/home/new_disk/yjw/checkpoints/step8_sft_multi_turn/checkpoint-10000",
                       help="Step8模型路径")
    parser.add_argument("--test_samples", type=int, default=10, help="测试样本数量")
    parser.add_argument("--max_rounds", type=int, default=6, help="最大修正轮数")
    parser.add_argument("--output", type=str, default=None, help="输出路径")
    parser.add_argument("--resume", action="store_true", help="从已有结果恢复")
    args = parser.parse_args()
    
    # 输出目录
    output_dir = Path("/home/weijie_yu/yjw/ECLO-RL/step8/evaluation_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化评测器
    print("🚀 初始化 Step8 评测器...")
    evaluator = Step8Evaluator(args.model_path)
    evaluator.load_model()
    
    # 设置输出路径
    if args.output:
        evaluator.output_path = Path(args.output)
    else:
        evaluator.output_path = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # 加载测试数据
    test_dir = Path("/home/weijie_yu/yjw/ECLO-RL/step6/Test")
    test_files = sorted(test_dir.glob("*.json"), key=lambda x: int(x.stem))[:args.test_samples]
    
    print(f"\n📁 找到 {len(test_files)} 个测试样本")
    
    # 如果是恢复模式，加载已有结果
    if args.resume and evaluator.output_path.exists():
        with open(evaluator.output_path, 'r', encoding='utf-8') as f:
            evaluator.results = json.load(f)
        print(f"  📂 已加载 {len(evaluator.results)} 个已有结果")
    
    # 处理每个样本
    processed_ids = set(r["sample_id"] for r in evaluator.results)
    
    for test_file in test_files:
        sample_id = test_file.stem
        
        # 跳过已处理的样本
        if sample_id in processed_ids:
            print(f"\n⏭️ 跳过已处理样本: {sample_id}")
            continue
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            table_content = data.get("table_content", "")
            ground_truth = {"knowledge_mining": data.get("knowledge_mining", [])}
            
            evaluator.evaluate_sample(
                sample_id, table_content, ground_truth,
                max_rounds=args.max_rounds
            )
            
        except Exception as e:
            print(f"  ⚠️ 处理样本失败: {test_file.name}, {e}")
            continue
    
    # 保存最终结果
    with open(evaluator.output_path, 'w', encoding='utf-8') as f:
        json.dump(evaluator.results, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("📊 评测汇总")
    print(f"{'='*60}")
    
    results = evaluator.results
    total = len(results)
    if total > 0:
        avg_initial = sum(r["initial_r_rule"] for r in results) / total
        avg_final = sum(r["final_r_rule"] for r in results) / total
        avg_improvement = sum(r["r_rule_improvement"] for r in results) / total
        
        print(f"样本数: {total}")
        print(f"平均初始 R_rule: {avg_initial:.4f}")
        print(f"平均最终 R_rule: {avg_final:.4f}")
        print(f"平均提升: {avg_improvement:.4f}")
        
        improved = sum(1 for r in results if r["r_rule_improvement"] > 0)
        print(f"提升样本数: {improved}/{total} ({improved/total*100:.1f}%)")
    
    print(f"\n✅ 结果已保存到: {evaluator.output_path}")


if __name__ == "__main__":
    main()

