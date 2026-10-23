#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step9 PPO trainer.

This trainer keeps the same action format and correction loop as Step8:
- initial KC generation (Step6 prompt format)
- strategy JSON generation (step8 strategy_selection prompt)
- teacher correction in environment
- reward from delta-R and safety constraints
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
except Exception:
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from step9.step8_reference import get_step6_prompts, normalize_kc_payload


def _resolve_prompt_root(prompt_root: Optional[Path] = None) -> Path:
    if prompt_root is not None:
        return Path(prompt_root)
    preferred = PROJECT_ROOT / "step9" / "prompt_packs" / "train_v1"
    if not preferred.exists():
        raise FileNotFoundError(f"train_v1 prompt pack not found at {preferred}")
    return preferred


@dataclass
class PPOConfig:
    learning_rate: float = 5e-6
    batch_size: int = 2
    mini_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    clip_epsilon: float = 0.1
    value_loss_coef: float = 0.2
    entropy_coef: float = 0.01
    kl_coef: float = 0.1
    ref_kl_coef: float = 0.05
    target_kl: float = 0.05
    warmup_steps: int = 0
    lr_scheduler: str = "linear"  # "linear", "cosine", "constant"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    max_prompt_tokens: int = 12000  # 与 step8.1 对齐
    max_new_tokens: int = 1024
    # Strategy-action generation cap.
    action_max_new_tokens: int = 512
    # Separate limits for initial-KC generation (与 step8.1 对齐为 12000)
    initial_max_prompt_tokens: int = 12000
    initial_max_new_tokens: int = 4096
    # Reward scale for initial-KC generation quality.
    # Kept lower than strategy-step shaping because initial generation uses
    # length-normalized log-probs for long-form stability.
    initial_r_reward_scale: float = 0.40
    # Give the initial-KC decision a modest optimization boost so PPO does not
    # over-focus on later repair actions only.
    initial_generation_loss_weight: float = 1.35
    # 【修复】初始 KC 的 PPO 评估 token 上限应该与生成上限一致
    # 否则长 KC 的后半段无法被 PPO 优化
    initial_action_max_eval_tokens: int = 4096
    # Initial-KC rollout should stay low-temperature but stochastic during PPO,
    # otherwise the policy cannot explore initial-chain variants.
    initial_generation_temperature: float = 0.10
    initial_generation_top_p: float = 0.95
    temperature: float = 0.15
    top_p: float = 0.95
    early_stop_kl: float = 0.2  # Stop current PPO update once KL is too large.
    ref_kl_max_tokens: int = 1024  # Max tokens for ref_kl computation (to avoid OOM).
    # Downsample ref_kl calls to reduce dual-GPU overhead.
    ref_kl_sample_rate: float = 0.25
    # Compute ref_kl every N mini-batches to reduce cross-device traffic.
    ref_kl_update_interval: int = 4
    # Keep failed-teacher steps in PPO, but down-weight their optimization impact.
    teacher_failed_adv_weight: float = 0.5
    # 【修复】高分阈值应该可配置，而不是硬编码
    high_score_threshold: float = 0.9
    # Prompt overflow handling: "truncate" keeps training robust, "error" fails fast.
    prompt_overflow_mode: str = "truncate"
    # Keep initial long-form KC generation frozen under PPO by default.
    optimize_initial_kc: bool = True


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.dtype = dtype
        # 灏嗘墍鏈夊弬鏁拌浆鎹负鎸囧畾鐨刣type
        self.to(dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Keep all value-head layers on one device and align with input tensor device.
        target_device = hidden_states.device
        param_devices = {p.device for p in self.value_head.parameters()}
        if len(param_devices) != 1 or (param_devices and target_device not in param_devices):
            self.value_head.to(device=target_device, dtype=self.dtype)
        try:
            head_device = next(self.value_head.parameters()).device
        except StopIteration:
            head_device = target_device
        if hidden_states.device != head_device:
            hidden_states = hidden_states.to(device=head_device, non_blocking=True)
        # 纭繚杈撳叆涓巚alue_head鐨刣type鍖归厤
        if hidden_states.dtype != self.dtype:
            hidden_states = hidden_states.to(dtype=self.dtype)
        return self.value_head(hidden_states).squeeze(-1)


class PPOTrainer:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        env: Any,
        config: PPOConfig,
        device: str = "cuda",
        ref_model: Any = None,  # Step8鍐荤粨鍩虹嚎妯″瀷锛岀敤浜巖ef_kl鎯╃綒
        prompt_root: Optional[Path] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.config = config
        self.device = device

        # Frozen Step8 baseline model for ref_kl regularization.
        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        hidden_size = int(self.model.config.hidden_size)
        # Keep value head dtype aligned with policy model.
        model_dtype = torch.bfloat16
        self.value_head = ValueHead(hidden_size, dtype=model_dtype).to(self._model_output_device())

        # Optimizer: separate lr for policy and value head.
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.parameters(), "lr": config.learning_rate},
                {"params": self.value_head.parameters(), "lr": config.learning_rate * 2},
            ],
            weight_decay=0.01,
        )

        # Learning-rate scheduler.
        self.lr_scheduler = self._create_scheduler()

        # Gradient accumulation counter.
        self.gradient_accumulation_counter = 0

        self.system_prompt_step6, self.input_prompt_step6 = get_step6_prompts()
        self.prompt_root = _resolve_prompt_root(prompt_root)
        strategy_prompt_path = self.prompt_root / "prompts" / "strategy_selection.txt"
        self.strategy_prompt_template = strategy_prompt_path.read_text(encoding="utf-8")
        self.strategy_selector_system_message = (
            "你是知识链修正策略选择器。"
            "必须只输出一个JSON对象，且只能包含strategy、chain_indices、reason三个字段。"
            "禁止输出任何额外文本、列表、代码块或解释。"
        )
        self.valid_strategies = {
            "Add/chain",
            "Add/condition_kv",
            "Add/entity_type",
            "Add/entity_value",
            "Modify/condition_kv",
            "Modify/result",
            "Modify/entity_type",
            "Modify/entity_value",
            "Delete/chain",
            "Delete/condition_kv",
            "Delete/result",
            "Delete/entity_type",
            "Delete/entity_value",
            # Backward compatibility alias.
            "Modify/entity_recognition",
        }
        self.initial_baseline_r_cache: Dict[str, float] = {}
        
        # 璁粌缁熻
        self.total_steps = 0
        self.current_iteration = 0
        self.total_training_steps = 1000

    def _create_scheduler(self):
        """鍒涘缓瀛︿範鐜囪皟搴﹀櫒"""
        # 浣跨敤constant璋冨害鍣紝鍚庣画鍙互鏍规嵁total_steps璋冩暣
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda
        )

    def set_total_training_steps(self, total_training_steps: int) -> None:
        self.total_training_steps = max(1, int(total_training_steps))
        print(f"[scheduler] total_training_steps set to {self.total_training_steps}", flush=True)
    
    def _lr_lambda(self, step: int) -> float:
        """Learning-rate multiplier."""
        warmup_steps = max(0, int(self.config.warmup_steps))
        total_steps = max(warmup_steps + 1, int(getattr(self, "total_training_steps", 1000)))
        if self.config.lr_scheduler == "linear":
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return max(0.0, 1.0 - progress)
        if self.config.lr_scheduler == "cosine":
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(1.0, max(0.0, progress))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return 1.0

    def _estimate_total_training_steps(
        self,
        num_iterations: int,
        episodes_per_iteration: int,
        update_every_episodes: int,
    ) -> int:
        chunk_size = int(update_every_episodes) if int(update_every_episodes) > 0 else int(episodes_per_iteration)
        chunk_size = max(1, min(chunk_size, int(episodes_per_iteration)))
        chunks_per_iteration = int(math.ceil(float(max(1, episodes_per_iteration)) / float(chunk_size)))

        approx_episode_steps = int(getattr(self.env, "max_rounds", 1)) + 1
        approx_episode_steps = max(1, approx_episode_steps)
        approx_flat_steps_per_chunk = int(chunk_size) * approx_episode_steps

        mini_batches_per_epoch = int(
            math.ceil(float(max(1, approx_flat_steps_per_chunk)) / float(max(1, int(self.config.mini_batch_size))))
        )
        total_mini_batches = mini_batches_per_epoch * max(1, int(self.config.ppo_epochs))
        opt_steps_per_chunk = int(
            math.ceil(float(max(1, total_mini_batches)) / float(max(1, int(self.config.gradient_accumulation_steps))))
        )
        return max(1, int(num_iterations) * chunks_per_iteration * opt_steps_per_chunk)

    # ---------------------------
    # Public rollout helpers
    # ---------------------------
    def generate_initial_kc(self, table_content: str) -> Dict[str, Any]:
        kc, _prompt_text, _generated_text, _generated_token_ids = self._generate_initial_kc_with_trace(
            table_content,
            deterministic=True,
        )
        if isinstance(kc, dict):
            try:
                return normalize_kc_payload(kc)
            except Exception:
                return kc
        return {"knowledge_mining": []}

    def _generate_initial_kc_with_trace(
        self,
        table_content: str,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], str, str, List[int]]:
        return self._generate_initial_kc_with_trace_for_model(
            self.model,
            table_content,
            context_prefix="generate_initial_kc",
            deterministic=deterministic,
        )

    def _generate_initial_kc_with_trace_for_model(
        self,
        model: Any,
        table_content: str,
        context_prefix: str,
        deterministic: bool,
    ) -> Tuple[Dict[str, Any], str, str, List[int]]:
        messages = [
            {"role": "system", "content": self.system_prompt_step6},
            {
                "role": "user",
                "content": (
                    f"{self.input_prompt_step6}\n\n"
                    f"<table_content>\n{table_content}\n</table_content>"
                ),
            },
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self._prepare_prompt_ids(
            prompt_text,
            max_tokens=self.config.initial_max_prompt_tokens,
            context=f"{context_prefix}_prompt",
        )
        prompt_tensor = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=self._input_device_for_model(model),
        )
        inputs = {
            "input_ids": prompt_tensor,
            "attention_mask": torch.ones_like(prompt_tensor),
        }

        model.eval()
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": self.config.initial_max_new_tokens,
                "do_sample": not deterministic,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if not deterministic:
                generation_kwargs["temperature"] = max(
                    1e-5,
                    float(self.config.initial_generation_temperature),
                )
                generation_kwargs["top_p"] = float(self.config.initial_generation_top_p)
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )
        generated_token_ids = outputs[0][len(prompt_ids) :].tolist()
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # 璋冭瘯锛氭墦鍗版ā鍨嬬敓鎴愮殑鍘熷鏂囨湰
        print(f"[{context_prefix}] Generated text (first 1000 chars):\n{generated_text[:1000]}", flush=True)
        
        try:
            json_str = self._extract_json(generated_text)
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                print(
                    f"[{context_prefix}] Successfully parsed KC with "
                    f"{len(parsed.get('knowledge_mining', []))} entries",
                    flush=True,
                )
                return parsed, prompt_text, generated_text, generated_token_ids
        except Exception as e:
            print(f"[{context_prefix}] Failed to parse: {e}", flush=True)

        recovered = self._recover_truncated_initial_kc(generated_text)
        if recovered is not None:
            recovered_len = len(recovered.get("knowledge_mining", []))
            print(
                f"[{context_prefix}] Recovered truncated KC with {recovered_len} complete entries",
                flush=True,
            )
            return recovered, prompt_text, generated_text, generated_token_ids

        print(f"[{context_prefix}] Returning empty KC due to parse failure", flush=True)
        return {"knowledge_mining": []}, prompt_text, generated_text, generated_token_ids

    def _baseline_initial_r_rule(
        self,
        sample_id: str,
        table_content: str,
    ) -> Optional[float]:
        if self.ref_model is None:
            return None
        cache_key = str(sample_id or "")
        if cache_key in self.initial_baseline_r_cache:
            return float(self.initial_baseline_r_cache[cache_key])

        baseline_kc, _prompt, _text, _token_ids = self._generate_initial_kc_with_trace_for_model(
            self.ref_model,
            table_content,
            context_prefix="ref_initial_kc",
            deterministic=True,
        )
        if isinstance(baseline_kc, dict):
            try:
                baseline_kc = normalize_kc_payload(baseline_kc)
            except Exception:
                pass
        baseline_r = float(self.env._compute_r_rule(baseline_kc))
        self.initial_baseline_r_cache[cache_key] = baseline_r
        print(
            f"[ref_initial_kc] cached baseline R-rule for {sample_id}: {baseline_r:.4f}",
            flush=True,
        )
        return baseline_r

    def select_action(
        self, state: Dict[str, Any], deterministic: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raw_prompt_text = self._build_strategy_prompt(state)
        prompt_text = self._render_strategy_chat_prompt(raw_prompt_text)
        round_idx = int(state.get("round_num", 0))
        retry_count = 0
        max_retries = 2  # 最多尝试 3 次（1 次原始 + 2 次重试）
        last_valid_action = None
        last_valid_token_ids = None
        last_valid_prompt_text = None

        while retry_count <= max_retries:
            print(f"[select_action] prompt len={len(prompt_text)}, round={round_idx}, attempt={retry_count+1}", flush=True)
            print(f"[select_action] Calling model.generate...", flush=True)
            raw_action_text, token_ids, rollout_log_prob, rollout_entropy = self._generate_action_text(
                prompt_text, deterministic=deterministic
            )
            print(f"[select_action] Generation complete, action_text len={len(raw_action_text)}", flush=True)
            action = self._parse_action(raw_action_text, round_idx=round_idx)

            # 保存有效的 action（第一次生成或任何非 INVALID 的 action）
            if action.get("action", "").upper() != "INVALID":
                last_valid_action = action
                last_valid_token_ids = token_ids
                last_valid_prompt_text = prompt_text

            # 检查是否需要重试
            should_retry = self._should_retry_invalid_action(action, retry_count)
            if not should_retry:
                break

            retry_count += 1
            retry_reason = str(action.get("reason", "") or "invalid_action")
            print(
                f"[select_action] invalid action detected, retrying ({retry_count}/{max_retries}): reason={retry_reason}",
                flush=True,
            )
            raw_prompt_text = self._build_strategy_retry_prompt(
                state=state,
                invalid_reason=retry_reason,
                previous_output=raw_action_text,
            )
            prompt_text = self._render_strategy_chat_prompt(raw_prompt_text)

        # 如果所有尝试都失败，使用最后一次有效的 action
        if last_valid_action is not None:
            action = last_valid_action
            token_ids = last_valid_token_ids
            prompt_text = last_valid_prompt_text
            print(f"[select_action] Using last valid action after {retry_count} retries", flush=True)
        elif action.get("action", "").upper() == "INVALID":
            # 兜底：返回 INVALID
            print(f"[select_action] All attempts invalid, returning INVALID", flush=True)

        action_text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip() if token_ids else raw_action_text
        action_token_cap = len(token_ids) if token_ids else int(self.config.action_max_new_tokens)

        # Score the exact rollout token sequence instead of a parsed/canonicalized JSON.
        with torch.no_grad():
            old_lp_t, old_ent_t, _ = self._evaluate_action_ids(
                prompt_text,
                token_ids,
                logprob_mode="token_mean",
            )
        old_log_prob = float(old_lp_t.detach().cpu().item())
        entropy = float(old_ent_t.detach().cpu().item())
        if not np.isfinite(old_log_prob):
            old_log_prob = float(rollout_log_prob)
        if not np.isfinite(entropy):
            entropy = float(rollout_entropy)

        value = self._estimate_value(prompt_text).item()
        meta = {
            "prompt_text": prompt_text,
            "raw_prompt_text": raw_prompt_text[:4000],
            "action_text": action_text,
            "raw_action_text": raw_action_text[:2000],
            "action_token_ids": token_ids,
            "old_log_prob": old_log_prob,
            "old_entropy": entropy,
            "value": float(value),
            "action_token_cap": int(max(1, action_token_cap)),
            "canonical_action_text": self._canonical_action_text(action),
        }
        return action, meta

    # ---------------------------
    # Rollout collection
    # ---------------------------
    def collect_trajectories(
        self,
        samples: List[Dict[str, Any]],
        num_episodes: int,
        deterministic: bool = False,
        start_episode_index: int = 0,
        run_timestamp: str = "",
        shuffle_samples: bool = True,
        record_episode_records: bool = True,
    ) -> List[Dict[str, Any]]:
        episodes: List[Dict[str, Any]] = []
        if not samples:
            return episodes

        episode_records_dir: Optional[Path] = None
        if record_episode_records:
            # Create directory for episode records with timestamp.
            if run_timestamp:
                episode_records_dir = PROJECT_ROOT / "step9" / "output" / f"episode_records_{run_timestamp}"
            else:
                episode_records_dir = PROJECT_ROOT / "step9" / "output" / "episode_records"
            episode_records_dir.mkdir(parents=True, exist_ok=True)

        print(f"[collect_trajectories] Starting with {num_episodes} episodes", flush=True)
        if episode_records_dir is not None:
            print(f"[collect_trajectories] Episode records will be saved to: {episode_records_dir}", flush=True)
        else:
            print("[collect_trajectories] Episode record persistence disabled for this run", flush=True)

        sample_order = list(range(len(samples)))
        if shuffle_samples:
            random.shuffle(sample_order)

        for ep_idx in range(num_episodes):
            if shuffle_samples and ep_idx > 0 and (ep_idx % len(samples) == 0):
                random.shuffle(sample_order)
            print(f"[collect_trajectories] Episode {ep_idx+1}/{num_episodes}", flush=True)
            order_idx = sample_order[ep_idx % len(samples)] if shuffle_samples else (ep_idx % len(samples))
            sample = samples[order_idx]
            table_content = sample["table_content"]
            ground_truth = {"knowledge_mining": sample["knowledge_mining"]}
            sample_id = sample.get("file", f"sample_{ep_idx}")

            print(f"\n{'='*60}", flush=True)
            print(f"[Episode {ep_idx+1}/{num_episodes}] Sample: {sample_id}", flush=True)
            print(f"{'='*60}", flush=True)
            print("Generating initial KC...", flush=True)

            initial_kc, initial_prompt_text, initial_generated_text, initial_generated_token_ids = self._generate_initial_kc_with_trace(table_content)
            # 与 step8.1 对齐：生成初始 KC 后立即归一化
            if isinstance(initial_kc, dict):
                try:
                    initial_kc = normalize_kc_payload(initial_kc)
                    print(f"[normalize] Initial KC normalized successfully", flush=True)
                except Exception as e:
                    print(f"[normalize] Warning: failed to normalize initial KC: {e}", flush=True)
            print(f"Initial KC length: {len(initial_kc.get('knowledge_mining', []))}", flush=True)

            kc_list = initial_kc.get("knowledge_mining", [])
            if kc_list:
                print("Initial KC preview:", flush=True)
                for i, kc in enumerate(kc_list[:2]):
                    print(f"  [{i}]: {str(kc)[:200]}...", flush=True)

            state = self.env.reset(table_content, initial_kc, ground_truth)
            print(f"Initial R-rule: {self.env.initial_r_rule:.4f}", flush=True)

            episode_steps: List[Dict[str, Any]] = []
            threshold_reached = bool(self.env.current_r_rule >= self.env.stop_threshold)
            if self.config.optimize_initial_kc:
                baseline_initial_r = self._baseline_initial_r_rule(sample_id, table_content)
                init_action_ids = list(initial_generated_token_ids or [])
                initial_action_text = self.tokenizer.decode(init_action_ids, skip_special_tokens=True).strip()
                if not init_action_ids:
                    initial_action_text = '{"knowledge_mining":[]}'
                    init_action_ids = self.tokenizer(initial_action_text, add_special_tokens=False)["input_ids"]
                init_action_cap = int(max(1, self.config.initial_action_max_eval_tokens))
                if len(init_action_ids) > init_action_cap:
                    init_action_ids = init_action_ids[:init_action_cap]
                    initial_action_text = self.tokenizer.decode(init_action_ids, skip_special_tokens=True)
                with torch.no_grad():
                    init_lp_t, _init_ent_t, init_val_t = self._evaluate_action_ids(
                        initial_prompt_text,
                        init_action_ids,
                        logprob_mode="token_mean",
                    )
                init_stop_reason = "initial_auto_threshold" if threshold_reached else None
                if baseline_initial_r is not None:
                    init_reward = float(
                        self.config.initial_r_reward_scale
                        * float(self.env.initial_r_rule - baseline_initial_r)
                    )
                else:
                    init_reward = float(self.config.initial_r_reward_scale * float(self.env.initial_r_rule))
                episode_steps.append(
                    {
                        "action": {
                            "action": "INITIAL_KC",
                            "strategy": "INITIAL_KC",
                            "chain_indices": [],
                            "reason": "initial_generation",
                        },
                        "prompt_text": initial_prompt_text,
                        "action_text": initial_action_text,
                        "action_token_ids": init_action_ids,
                        "old_log_prob": float(init_lp_t.detach().cpu().item()),
                        "value": float(init_val_t.detach().cpu().item()),
                        "action_token_cap": init_action_cap,
                        "logprob_mode": "token_mean",
                        "reward": init_reward,
                        "done": False,
                        "info": {
                            "stage": "initial_generation",
                            "baseline_initial_r_rule": baseline_initial_r,
                            "r_before": self.env.initial_r_rule,
                            "r_after": self.env.initial_r_rule,
                            "raw_delta": 0.0,
                            "effective_delta": 0.0,
                            "stop_reason": init_stop_reason,
                            "reward": init_reward,
                        },
                    }
                )
                print(f"Initial generation reward: {init_reward:.4f}", flush=True)
            else:
                print("Initial generation PPO optimization disabled; using Step8 output as rollout start.", flush=True)
            done = False
            if threshold_reached:
                print(
                    f"Initial R-rule already >= threshold ({self.env.stop_threshold:.2f}); "
                    "collecting an on-policy stop/continue decision.",
                    flush=True,
                )
            else:
                print(f"Start correction loop (max_rounds={self.env.max_rounds})...", flush=True)
            round_num = 0
            
            while not done:
                round_num += 1
                print(f"\n--- Round {round_num} ---", flush=True)
                
                action, meta = self.select_action(state, deterministic=deterministic)
                
                print("Model action JSON:", flush=True)
                print(json.dumps(action, ensure_ascii=False, indent=2)[:1000], flush=True)
                
                next_state, reward, done, info = self.env.step(action)
                
                r_before = info.get("r_before", 0)
                r_after = info.get("r_after", 0)
                raw_delta = info.get("raw_delta", 0)
                effective_delta = info.get("effective_delta", 0)
                stop_reason = info.get("stop_reason", "")
                
                print(f"R-before: {r_before:.4f}, R-after: {r_after:.4f}", flush=True)
                print(f"delta={effective_delta:+.4f}, reward={reward:.4f}", flush=True)
                if stop_reason:
                    print(f"Stop reason: {stop_reason}", flush=True)
                
                episode_steps.append(
                    {
                        "action": action,
                        "prompt_text": meta["prompt_text"],
                        "action_text": meta["action_text"],
                        "action_token_ids": meta.get("action_token_ids", []),
                        "old_log_prob": float(meta["old_log_prob"]),
                        "value": float(meta["value"]),
                        "action_token_cap": int(meta.get("action_token_cap", self.config.action_max_new_tokens)),
                        "logprob_mode": "token_mean",
                        "reward": float(reward),
                        "done": bool(done),
                        "info": info,
                    }
                )
                state = next_state

            episode_record = {
                "sample_id": sample_id,
                "initial_r_rule": self.env.initial_r_rule,
                "final_r_rule": self.env.current_r_rule,
                "best_r_rule": self.env.best_r_rule,
                "num_rounds": self.env.round_num,
                "total_reward": float(sum(s["reward"] for s in episode_steps)),
                "steps": [
                    {
                        "round": i + 1,
                        "action": s["action"],
                        "reward": s["reward"],
                        "r_before": s["info"].get("r_before", 0),
                        "r_after": s["info"].get("r_after", 0),
                        "raw_delta": s["info"].get("raw_delta", 0),
                        "effective_delta": s["info"].get("effective_delta", 0),
                        "rolled_back": bool(s["info"].get("rolled_back", False)),
                        "rollback_to_best": bool(s["info"].get("rollback_to_best", False)),
                        "teacher_failed": bool(s["info"].get("teacher_failed", False)),
                        "teacher_error": s["info"].get("teacher_error", ""),
                        "stop_reason": s["info"].get("stop_reason", ""),
                    }
                    for i, s in enumerate(episode_steps)
                ]
            }

            if episode_records_dir is not None:
                episode_index = int(start_episode_index) + ep_idx + 1
                episode_record_path = episode_records_dir / f"{episode_index}.json"
                episode_record_path.parent.mkdir(parents=True, exist_ok=True)
                episode_record_path.write_text(json.dumps(episode_record, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"Episode record saved: {episode_record_path}", flush=True)
            
            print(f"\n{'='*60}", flush=True)
            print(f"Episode complete: initial_R={self.env.initial_r_rule:.4f}, final_R={self.env.current_r_rule:.4f}, best_R={self.env.best_r_rule:.4f}", flush=True)
            print(f"Total rounds: {self.env.round_num}, total_reward: {sum(s['reward'] for s in episode_steps):.4f}", flush=True)
            print(f"{'='*60}\n", flush=True)

            episodes.append(
                {
                    "sample_id": sample_id,
                    "steps": episode_steps,
                    "initial_r_rule": self.env.initial_r_rule,
                    "final_r_rule": self.env.current_r_rule,
                    "best_r_rule": self.env.best_r_rule,
                    "num_rounds": self.env.round_num,
                    "total_reward": float(sum(s["reward"] for s in episode_steps)),
                }
            )
            
            # Clear model caches after each episode to prevent memory accumulation
            torch.cuda.empty_cache()
            
        return episodes

    # ---------------------------
    # PPO update
    # ---------------------------
    def compute_advantages(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        flat_steps: List[Dict[str, Any]] = []
        for episode in episodes:
            steps = episode["steps"]
            if not steps:
                continue

            rewards = [float(s["reward"]) for s in steps]
            values = [float(s["value"]) for s in steps]
            dones = [bool(s["done"]) for s in steps]

            advantages = [0.0] * len(steps)
            returns = [0.0] * len(steps)
            gae = 0.0
            next_value = 0.0
            for t in reversed(range(len(steps))):
                mask = 0.0 if dones[t] else 1.0
                delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
                advantages[t] = gae
                returns[t] = gae + values[t]
                next_value = values[t]

            for idx, step in enumerate(steps):
                info = step.get("info", {}) if isinstance(step.get("info", {}), dict) else {}
                teacher_failed = bool(info.get("teacher_failed", False))
                done_flag = bool(step.get("done", False))
                effective_delta = float(info.get("effective_delta", step.get("effective_delta", 0.0)))
                stage_name = str(info.get("stage", "")).strip().lower()
                is_initial_generation = stage_name == "initial_generation"

                loss_weight = 1.0
                if is_initial_generation:
                    loss_weight = loss_weight * float(self.config.initial_generation_loss_weight)
                if teacher_failed and not done_flag:
                    loss_weight = float(self.config.teacher_failed_adv_weight)
                    loss_weight = max(0.0, min(1.0, loss_weight))

                # Positive sample weighting: boost steps with effective improvement
                POSITIVE_WEIGHT = 1.2
                NEGATIVE_WEIGHT = 0.6
                if is_initial_generation:
                    if float(step.get("reward", 0.0)) > 0:
                        loss_weight = loss_weight * 1.1
                elif effective_delta > 0:
                    loss_weight = loss_weight * POSITIVE_WEIGHT
                elif effective_delta <= 0 and not teacher_failed:
                    # 【修复】使用 config 中的 high_score_threshold 而不是硬编码
                    # 只有高分 teacher_stop (R >= high_score_threshold) 才不降权
                    is_teacher_stop = bool(info.get("teacher_stop", False))
                    r_score = float(info.get("r_score", 0.0))
                    high_score_threshold = self.config.high_score_threshold

                    # 只有同时满足：1) 是 teacher_stop 2) R >= high_score_threshold  才不降权
                    is_high_score_teacher_stop = is_teacher_stop and r_score >= high_score_threshold

                    if not is_high_score_teacher_stop:
                        loss_weight = loss_weight * NEGATIVE_WEIGHT

                row = dict(step)
                row["advantage"] = float(advantages[idx])
                row["return"] = float(returns[idx])
                row["loss_weight"] = float(loss_weight)
                # Keep optimizer payload compact to reduce host memory pressure.
                row.pop("state", None)
                row.pop("info", None)
                flat_steps.append(row)

        if not flat_steps:
            return flat_steps

        adv = np.array([s["advantage"] for s in flat_steps], dtype=np.float32)
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for i, val in enumerate(adv.tolist()):
            flat_steps[i]["advantage"] = float(val)
        return flat_steps

    def update_policy(self, steps: List[Dict[str, Any]]) -> Dict[str, float]:
        if not steps:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "kl": 0.0,
            }

        self.model.train()
        self.value_head.train()

        metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": [], "ref_kl": []}
        indices = list(range(len(steps)))

        accum_steps = max(1, int(self.config.gradient_accumulation_steps))
        batch_size = max(1, int(self.config.batch_size))
        mini_batch_size = max(1, int(self.config.mini_batch_size))
        ref_kl_interval = max(1, int(getattr(self.config, "ref_kl_update_interval", 1)))
        self.optimizer.zero_grad(set_to_none=True)
        self.gradient_accumulation_counter = 0

        def _weighted_mean(values_t: torch.Tensor, weights_t: torch.Tensor) -> torch.Tensor:
            safe_weights = torch.clamp(weights_t, min=0.0)
            denom = safe_weights.sum().clamp(min=1e-6)
            return (values_t * safe_weights).sum() / denom

        print(
            f"[update_policy] Starting PPO update with {len(steps)} steps, "
            f"{self.config.ppo_epochs} epochs, batch_size={batch_size}, mini_batch_size={mini_batch_size}, "
            f"ref_kl_interval={ref_kl_interval}",
            flush=True,
        )

        early_stopped = False
        mini_batch_counter = 0
        for _epoch in range(self.config.ppo_epochs):
            print(f"[update_policy] Epoch {_epoch+1}/{self.config.ppo_epochs}", flush=True)
            random.shuffle(indices)
            for macro_start in range(0, len(indices), batch_size):
                macro_ids = indices[macro_start : macro_start + batch_size]
                if not macro_ids:
                    continue
                for start in range(0, len(macro_ids), mini_batch_size):
                    batch_ids = macro_ids[start : start + mini_batch_size]
                    if not batch_ids:
                        continue
                    mini_batch_counter += 1

                    new_log_probs: List[torch.Tensor] = []
                    entropies: List[torch.Tensor] = []
                    values: List[torch.Tensor] = []
                    old_log_probs: List[float] = []
                    advantages: List[float] = []
                    returns: List[float] = []
                    loss_weights: List[float] = []

                    for idx in batch_ids:
                        step = steps[idx]
                        step_action_ids = step.get("action_token_ids", [])
                        if isinstance(step_action_ids, list) and step_action_ids:
                            new_lp, ent, val = self._evaluate_action_ids(
                                step["prompt_text"],
                                step_action_ids,
                                logprob_mode=str(step.get("logprob_mode", "token_mean")),
                            )
                        else:
                            new_lp, ent, val = self._evaluate_action(
                                step["prompt_text"],
                                step["action_text"],
                                max_action_tokens=int(step.get("action_token_cap", self.config.action_max_new_tokens)),
                                logprob_mode=str(step.get("logprob_mode", "token_mean")),
                            )
                        new_log_probs.append(new_lp)
                        entropies.append(ent)
                        values.append(val)
                        old_log_probs.append(float(step["old_log_prob"]))
                        advantages.append(float(step["advantage"]))
                        returns.append(float(step["return"]))
                        loss_weights.append(float(step.get("loss_weight", 1.0)))

                    new_log_probs_t = torch.stack(new_log_probs).float()
                    entropies_t = torch.stack(entropies).float()
                    values_t = torch.stack(values).float()
                    old_log_probs_t = torch.tensor(
                        old_log_probs, dtype=torch.float32, device=self._model_device()
                    )
                    advantages_t = torch.tensor(
                        advantages, dtype=torch.float32, device=self._model_device()
                    )
                    returns_t = torch.tensor(
                        returns, dtype=torch.float32, device=self._model_device()
                    )
                    loss_weights_t = torch.tensor(
                        loss_weights, dtype=torch.float32, device=self._model_device()
                    )

                    log_ratio = torch.clamp(new_log_probs_t - old_log_probs_t, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * advantages_t
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * advantages_t
                    policy_loss = -_weighted_mean(torch.min(surr1, surr2), loss_weights_t)
                    value_sq_err = (values_t - returns_t) ** 2
                    value_loss = _weighted_mean(value_sq_err, loss_weights_t)
                    entropy_bonus = _weighted_mean(entropies_t, loss_weights_t)
                    # Non-negative PPO KL proxy: E[(r-1)-log(r)].
                    approx_kl = _weighted_mean(((ratio - 1.0) - log_ratio), loss_weights_t)
                    approx_kl = torch.nan_to_num(approx_kl, nan=0.0, posinf=50.0, neginf=0.0)
                    approx_kl_item = float(approx_kl.detach().cpu().item())

                    loss = (
                        policy_loss
                        + self.config.value_loss_coef * value_loss
                        - self.config.entropy_coef * entropy_bonus
                        + self.config.kl_coef * approx_kl
                    )

                    ref_kl_loss = 0.0
                    should_compute_ref_kl = ((mini_batch_counter - 1) % ref_kl_interval) == 0
                    if (
                        self.ref_model is not None
                        and self.config.ref_kl_coef > 0
                        and should_compute_ref_kl
                    ):
                        sample_rate = float(getattr(self.config, "ref_kl_sample_rate", 1.0))
                        sample_rate = min(1.0, max(0.0, sample_rate))
                        selected_ids: List[int] = []
                        for idx in batch_ids:
                            if sample_rate >= 1.0 or random.random() < sample_rate:
                                selected_ids.append(idx)
                        if not selected_ids and batch_ids and sample_rate > 0.0:
                            selected_ids.append(batch_ids[0])
                        if selected_ids:
                            selected_steps = [steps[idx] for idx in selected_ids]
                            selected_weights = [float(steps[idx].get("loss_weight", 1.0)) for idx in selected_ids]
                            ref_kl_t = self._compute_ref_kl_batch(selected_steps)
                            ref_weight = float(sum(selected_weights) / max(1, len(selected_weights)))
                            ref_kl_loss = torch.nan_to_num(ref_kl_t.float() * ref_weight, nan=0.0, posinf=50.0, neginf=0.0)
                            loss = loss + self.config.ref_kl_coef * ref_kl_loss
                            metrics["ref_kl"].append(float(ref_kl_loss.detach().cpu().item()))

                    loss = loss / accum_steps
                    loss.backward()

                    # Clear cache after backward to free activation memory
                    torch.cuda.empty_cache()

                    self.gradient_accumulation_counter += 1

                    if self.gradient_accumulation_counter >= accum_steps:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.parameters()) + list(self.value_head.parameters()),
                            self.config.max_grad_norm,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        self.total_steps += 1
                        self.lr_scheduler.step()
                        self.gradient_accumulation_counter = 0

                    metrics["policy_loss"].append(float(policy_loss.detach().cpu().item()))
                    metrics["value_loss"].append(float(value_loss.detach().cpu().item()))
                    metrics["entropy"].append(float(entropy_bonus.detach().cpu().item()))
                    metrics["kl"].append(approx_kl_item)

                    # Early stop per epoch when KL is too large to avoid policy collapse.
                    if approx_kl_item > self.config.early_stop_kl:
                        early_stopped = True
                        print(
                            f"[update_policy] Early stop triggered: approx_kl={approx_kl_item:.4f} > early_stop_kl={self.config.early_stop_kl:.4f}",
                            flush=True,
                        )
                        break
                if early_stopped:
                    break
            if early_stopped:
                break

        if self.gradient_accumulation_counter > 0:
            if early_stopped:
                # Drop residual partial gradients when KL early-stop is triggered.
                self.optimizer.zero_grad(set_to_none=True)
                self.gradient_accumulation_counter = 0
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.total_steps += 1
                self.lr_scheduler.step()
                self.gradient_accumulation_counter = 0

        out: Dict[str, float] = {}
        for k, vals in metrics.items():
            out[k] = float(np.mean(vals)) if vals else 0.0

        out["early_stop"] = 1.0 if early_stopped else 0.0

        # Adaptive KL coefficient around target_kl.
        mean_kl = max(0.0, float(out.get("kl", 0.0)))
        target_kl = max(1e-6, float(self.config.target_kl))
        if mean_kl > target_kl * 1.5:
            self.config.kl_coef = float(self.config.kl_coef) * 1.2
        elif mean_kl < target_kl * 0.7:
            self.config.kl_coef = float(self.config.kl_coef) * 0.9
        self.config.kl_coef = min(max(float(self.config.kl_coef), 1e-3), 1.0)
        out["kl_coef"] = float(self.config.kl_coef)

        return out

    def save_checkpoint(
        self,
        checkpoint_path: Path,
        iteration: int,
        history: List[Dict[str, float]],
    ) -> None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: Dict[str, Any] = {
            "iteration": int(iteration),
            "history": history,
            "total_steps": int(self.total_steps),
            "gradient_accumulation_counter": int(self.gradient_accumulation_counter),
            "value_head_state_dict": self.value_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
        }
        if get_peft_model_state_dict is not None:
            checkpoint["peft_model_state_dict"] = get_peft_model_state_dict(self.model)
        else:
            checkpoint["model_state_dict"] = self.model.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print(f"[checkpoint] Saved: {checkpoint_path}", flush=True)

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "peft_model_state_dict" in checkpoint and set_peft_model_state_dict is not None:
            set_peft_model_state_dict(self.model, checkpoint["peft_model_state_dict"])
        elif "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.value_head.load_state_dict(checkpoint["value_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.value_head.to(self._model_output_device())
        for param, state in self.optimizer.state.items():
            param_device = param.device if torch.is_tensor(param) else self._model_device()
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(param_device)
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        self.total_steps = int(checkpoint.get("total_steps", 0))
        self.gradient_accumulation_counter = int(
            checkpoint.get("gradient_accumulation_counter", 0)
        )
        print(f"[checkpoint] Loaded: {checkpoint_path}", flush=True)
        return {
            "iteration": int(checkpoint.get("iteration", 0)),
            "history": checkpoint.get("history", []),
        }

    def train(
        self,
        train_samples: List[Dict[str, Any]],
        num_iterations: int,
        episodes_per_iteration: int,
        update_every_episodes: int = 0,
        eval_samples: Optional[List[Dict[str, Any]]] = None,
        eval_every_iterations: int = 1,
        start_iteration: int = 0,
        initial_history: Optional[List[Dict[str, float]]] = None,
        checkpoint_dir: Optional[Path] = None,
        save_every_iterations: int = 1,
        run_timestamp: str = "",
        record_episode_records: bool = True,
        write_iteration_json: bool = True,
    ) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = list(initial_history or [])
        estimated_total_steps = self._estimate_total_training_steps(
            num_iterations=num_iterations,
            episodes_per_iteration=episodes_per_iteration,
            update_every_episodes=update_every_episodes,
        )
        self.set_total_training_steps(estimated_total_steps)

        # Generate timestamp if not provided
        if not run_timestamp:
            run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"[train] run_timestamp: {run_timestamp}", flush=True)

        iter_log_dir: Optional[Path] = None
        if write_iteration_json:
            iter_log_dir = PROJECT_ROOT / "step9" / "output" / "logs" / "iterations"
            iter_log_dir.mkdir(parents=True, exist_ok=True)
        eval_samples = list(eval_samples or [])

        checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        try:
            for it in range(start_iteration, num_iterations):
                self.current_iteration = it + 1
                print(f"\n{'#'*60}", flush=True)
                print(f"# Iteration {it+1}/{num_iterations}", flush=True)
                print(f"{'#'*60}", flush=True)
                print(f"[Iteration {it+1}/{num_iterations}] Collecting trajectories...", flush=True)

                chunk_size = int(update_every_episodes) if int(update_every_episodes) > 0 else int(episodes_per_iteration)
                chunk_size = max(1, min(chunk_size, int(episodes_per_iteration)))
                remaining = int(episodes_per_iteration)
                iteration_episode_samples: List[Dict[str, Any]] = []
                total_needed = int(episodes_per_iteration)
                idxs = list(range(len(train_samples)))
                random.shuffle(idxs)
                if total_needed <= len(idxs):
                    iteration_episode_samples = [train_samples[idx] for idx in idxs[:total_needed]]
                else:
                    while len(iteration_episode_samples) < total_needed:
                        random.shuffle(idxs)
                        for idx in idxs:
                            iteration_episode_samples.append(train_samples[idx])
                            if len(iteration_episode_samples) >= total_needed:
                                break

                episodes: List[Dict[str, Any]] = []
                total_flat_steps = 0
                update_metric_blocks: List[Tuple[Dict[str, float], int]] = []

                chunk_idx = 0
                while remaining > 0:
                    chunk_idx += 1
                    cur_eps = min(chunk_size, remaining)
                    print(
                        f"[Iteration {it+1}] Chunk {chunk_idx}: collect {cur_eps} episodes, remaining={remaining-cur_eps}",
                        flush=True,
                    )

                    global_episode_start = int(it) * int(episodes_per_iteration) + len(episodes)
                    chunk_start = len(episodes)
                    chunk_end = chunk_start + cur_eps
                    chunk_samples = iteration_episode_samples[chunk_start:chunk_end]
                    chunk_episodes = self.collect_trajectories(
                        chunk_samples,
                        cur_eps,
                        deterministic=False,
                        start_episode_index=global_episode_start,
                        run_timestamp=run_timestamp,
                        shuffle_samples=False,
                        record_episode_records=record_episode_records,
                    )
                    episodes.extend(chunk_episodes)

                    flat_steps = self.compute_advantages(chunk_episodes)
                    step_count = int(len(flat_steps))
                    total_flat_steps += step_count

                    if step_count > 0:
                        chunk_metrics = self.update_policy(flat_steps)
                        update_metric_blocks.append((chunk_metrics, step_count))
                    # Release per-step rollout payload once this chunk update is finished.
                    for ep in chunk_episodes:
                        ep.pop("steps", None)
                    remaining -= cur_eps

                # Weighted average of per-chunk update metrics.
                update_metrics: Dict[str, float] = {}
                if update_metric_blocks:
                    key_set = set()
                    for m, _w in update_metric_blocks:
                        key_set.update(m.keys())
                    total_weight = float(sum(w for _m, w in update_metric_blocks))
                    for k in key_set:
                        num = 0.0
                        for m, w in update_metric_blocks:
                            num += float(m.get(k, 0.0)) * float(w)
                        update_metrics[k] = num / max(1.0, total_weight)
                else:
                    update_metrics = {
                        "policy_loss": 0.0,
                        "value_loss": 0.0,
                        "entropy": 0.0,
                        "kl": 0.0,
                        "ref_kl": 0.0,
                        "early_stop": 0.0,
                        "kl_coef": float(self.config.kl_coef),
                    }

                avg_reward = float(np.mean([ep["total_reward"] for ep in episodes])) if episodes else 0.0
                avg_final_r = float(np.mean([ep["final_r_rule"] for ep in episodes])) if episodes else 0.0
                avg_initial_r = float(np.mean([ep["initial_r_rule"] for ep in episodes])) if episodes else 0.0
                avg_rounds = float(np.mean([ep["num_rounds"] for ep in episodes])) if episodes else 0.0
                eval_metrics: Dict[str, float] = {}
                if eval_samples and eval_every_iterations > 0 and ((it + 1) % int(eval_every_iterations) == 0):
                    eval_metrics = self.evaluate_policy(
                        eval_samples,
                        iteration=it + 1,
                        run_timestamp=run_timestamp,
                    )

                history_entry = {
                    "iteration": it + 1,
                    "episodes": float(len(episodes)),
                    "steps": float(total_flat_steps),
                    "avg_reward": avg_reward,
                    "avg_final_r_rule": avg_final_r,
                    "avg_initial_r_rule": avg_initial_r,
                    "avg_rounds": avg_rounds,
                    **update_metrics,
                }
                if eval_metrics:
                    history_entry.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                history.append(history_entry)

                iteration_record = {
                    "iteration": it + 1,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "episodes_per_iteration": episodes_per_iteration,
                    "avg_reward": avg_reward,
                    "avg_initial_r_rule": avg_initial_r,
                    "avg_final_r_rule": avg_final_r,
                    "avg_rounds": avg_rounds,
                    "update_metrics": update_metrics,
                    "eval_metrics": eval_metrics,
                    "episodes_detail": [
                        {
                            "sample_id": ep["sample_id"],
                            "initial_r_rule": ep["initial_r_rule"],
                            "final_r_rule": ep["final_r_rule"],
                            "best_r_rule": ep.get("best_r_rule", ep["final_r_rule"]),
                            "num_rounds": ep["num_rounds"],
                            "total_reward": ep["total_reward"],
                        }
                        for ep in episodes
                    ]
                }

                if iter_log_dir is not None:
                    iter_log_path = iter_log_dir / f"iteration_{it+1:04d}.json"
                    iter_log_path.parent.mkdir(parents=True, exist_ok=True)
                    iter_log_path.write_text(json.dumps(iteration_record, ensure_ascii=False, indent=2), encoding="utf-8")
                    print(f"\nIteration JSON saved: {iter_log_path}", flush=True)

                if checkpoint_dir and save_every_iterations > 0 and ((it + 1) % save_every_iterations == 0):
                    self.save_checkpoint(
                        checkpoint_dir / f"ppo_ckpt_iter_{it+1:04d}.pt",
                        iteration=it + 1,
                        history=history,
                    )
                if checkpoint_dir:
                    self.save_checkpoint(
                        checkpoint_dir / "ppo_ckpt_latest.pt",
                        iteration=it + 1,
                        history=history,
                    )

                print(f"[Iteration {it+1}] avg_reward={avg_reward:.4f}, avg_final_r={avg_final_r:.4f}, avg_rounds={avg_rounds:.1f}, policy_loss={update_metrics.get('policy_loss', 0):.4f}, kl={update_metrics.get('kl', 0):.4f}", flush=True)
                if eval_metrics:
                    print(
                        f"[Iteration {it+1}] held-out eval: "
                        f"avg_initial_r={eval_metrics.get('avg_initial_r_rule', 0.0):.4f}, "
                        f"avg_final_r={eval_metrics.get('avg_final_r_rule', 0.0):.4f}, "
                        f"improvement_rate={eval_metrics.get('improvement_rate', 0.0):.4f}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            if checkpoint_dir:
                self.save_checkpoint(
                    checkpoint_dir / "ppo_ckpt_interrupt_latest.pt",
                    iteration=self.current_iteration,
                    history=history,
                )
            print("[train] Interrupted by user/signal.", flush=True)
        except Exception as exc:
            if checkpoint_dir:
                self.save_checkpoint(
                    checkpoint_dir / "ppo_ckpt_crash_latest.pt",
                    iteration=self.current_iteration,
                    history=history,
                )
            print(
                f"[train] Exception: {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        if checkpoint_dir and self.current_iteration > 0:
            self.save_checkpoint(
                checkpoint_dir / "ppo_ckpt_latest.pt",
                iteration=self.current_iteration,
                history=history,
            )
        return history

    def evaluate_policy(
        self,
        eval_samples: List[Dict[str, Any]],
        iteration: int,
        run_timestamp: str = "",
    ) -> Dict[str, float]:
        if not eval_samples:
            return {}

        print(
            f"[heldout-eval] Iteration {iteration}: evaluating {len(eval_samples)} held-out samples...",
            flush=True,
        )
        eval_episodes = self.collect_trajectories(
            eval_samples,
            num_episodes=len(eval_samples),
            deterministic=True,
            start_episode_index=0,
            run_timestamp=f"{run_timestamp}_heldout_iter_{iteration:04d}",
            shuffle_samples=False,
            record_episode_records=False,
        )
        if not eval_episodes:
            return {}

        improvement_rate = float(
            np.mean(
                [
                    1.0 if ep["final_r_rule"] > ep["initial_r_rule"] + 1e-6 else 0.0
                    for ep in eval_episodes
                ]
            )
        )
        metrics = {
            "samples": float(len(eval_episodes)),
            "avg_reward": float(np.mean([ep["total_reward"] for ep in eval_episodes])),
            "avg_initial_r_rule": float(np.mean([ep["initial_r_rule"] for ep in eval_episodes])),
            "avg_final_r_rule": float(np.mean([ep["final_r_rule"] for ep in eval_episodes])),
            "avg_best_r_rule": float(np.mean([ep.get("best_r_rule", ep["final_r_rule"]) for ep in eval_episodes])),
            "avg_rounds": float(np.mean([ep["num_rounds"] for ep in eval_episodes])),
            "improvement_rate": improvement_rate,
        }
        return metrics

    # ---------------------------
    # Prompt / action utils
    # ---------------------------
    def _compact_history_context(
        self,
        history_context: str,
        max_lines: int = 6,
        max_chars: int = 900,
    ) -> str:
        lines = [line.strip() for line in str(history_context or "").splitlines() if line.strip()]
        if not lines:
            return ""
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        compact = "\n".join(lines)
        if len(compact) > max_chars:
            compact = compact[-max_chars:]
        return compact

    def _build_strategy_prompt(self, state: Dict[str, Any]) -> str:
        prompt = self.strategy_prompt_template
        table_content = str(state.get("table_content", "") or "")
        current_kc = state.get("current_kc", {})
        history_context = self._compact_history_context(state.get("history_context", ""))
        last_teacher_summary = str(state.get("last_teacher_summary", "") or "")
        prompt = prompt.replace("{table_content}", table_content)
        prompt = prompt.replace(
            "{current_knowledge_chain}",
            json.dumps(current_kc, ensure_ascii=False),
        )
        summary_blocks: List[str] = []
        if last_teacher_summary:
            summary_blocks.append(f"## 上一轮Teacher总结\n{last_teacher_summary}")
        if history_context:
            summary_blocks.append(f"## 历史修正\n{history_context}")
        if summary_blocks:
            prompt = prompt.replace(
                "{previous_summary_block}",
                "\n" + "\n\n".join(summary_blocks),
            )
        else:
            prompt = prompt.replace("{previous_summary_block}", "")
        return prompt

    def _should_retry_invalid_action(self, action: Dict[str, Any], retry_count: int = 0) -> bool:
        """判断是否应该重试-invalid action，增加重试次数"""
        if retry_count >= 2:  # 最多重试 2 次（共 3 次尝试）
            return False
        if str(action.get("action", "")).upper() != "INVALID":
            return False
        reason = str(action.get("reason", "") or "").strip().lower()
        return reason in {
            "missing_strategy",
            "invalid_strategy",
            "parse_failed",
            "missing_indices",
        }

    def _build_strategy_retry_prompt(
        self,
        state: Dict[str, Any],
        invalid_reason: str,
        previous_output: str,
    ) -> str:
        table_content = str(state.get("table_content", "") or "")
        current_kc = state.get("current_kc", {})
        previous_output = str(previous_output or "").strip()
        if len(previous_output) > 1200:
            previous_output = previous_output[:1200] + "\n...[TRUNCATED]..."
        allowed = ", ".join(sorted(s for s in self.valid_strategies if s != "Modify/entity_recognition"))
        return (
            "你上一条策略输出无效，必须严格重答。\n"
            "只输出一个 JSON 对象，禁止任何额外文字、markdown、代码块、列表或解释。\n"
            "输出格式必须严格为："
            "{\"strategy\":\"...\",\"chain_indices\":[...],\"reason\":\"...\"}\n"
            f"strategy 只能取以下值之一：{allowed}, STOP\n"
            "规则：\n"
            "- 若 strategy=STOP，则 chain_indices 必须为 []。\n"
            "- 若 strategy=Add/chain，则 chain_indices 必须为 []。\n"
            "- 其他策略必须提供至少一个有效整数 chain_indices。\n"
            "- reason 必须是简短字符串。\n"
            f"上一条无效原因：{invalid_reason}\n"
            f"上一条无效输出：\n{previous_output}\n\n"
            f"表格内容：\n{table_content}\n\n"
            f"当前知识链 JSON：\n{json.dumps(current_kc, ensure_ascii=False)}"
        )

    def _effective_prompt_max_tokens(self, max_tokens: int) -> int:
        limit = max(1, int(max_tokens))
        model_max = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(model_max, int) and model_max > 0:
            limit = min(limit, model_max)
        tokenizer_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 100000:
            limit = min(limit, tokenizer_max)
        return limit

    def _invalid_action_payload(
        self,
        reason: str,
        strategy: str = "",
        chain_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        return {
            "action": "INVALID",
            "strategy": (strategy or "INVALID")[:128],
            "chain_indices": list(chain_indices or []),
            "reason": (reason or "invalid_action")[:256],
        }

    def _render_chat_prompt(self, prompt_text: str) -> str:
        stripped = prompt_text.lstrip()
        # Avoid double-wrapping if prompt is already chat-formatted.
        if (
            "<|im_start|>" in prompt_text
            or "<|assistant|>" in prompt_text
            or "<|start_header_id|>" in prompt_text
            or "<|eot_id|>" in prompt_text
            or stripped.startswith("[INST]")
        ):
            return prompt_text
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(f"[_render_chat_prompt] Chat template failed, using raw prompt: {e}", flush=True)
            return prompt_text

    def _render_strategy_chat_prompt(self, prompt_text: str) -> str:
        stripped = prompt_text.lstrip()
        if (
            "<|im_start|>" in prompt_text
            or "<|assistant|>" in prompt_text
            or "<|start_header_id|>" in prompt_text
            or "<|eot_id|>" in prompt_text
            or stripped.startswith("[INST]")
        ):
            return prompt_text
        messages = [
            {"role": "system", "content": self.strategy_selector_system_message},
            {"role": "user", "content": prompt_text},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception as e:
            print(
                f"[_render_strategy_chat_prompt] Chat template failed, using raw prompt: {e}",
                flush=True,
            )
            return prompt_text

    def _generate_action_text(
        self, prompt_text: str, deterministic: bool
    ) -> Tuple[str, List[int], float, float]:
        print(f"[_generate_action_text] Starting generation, prompt_len={len(prompt_text)}, deterministic={deterministic}", flush=True)

        prompt_ids = self._prepare_prompt_ids(
            prompt_text,
            max_tokens=self.config.max_prompt_tokens,
            context="strategy_prompt",
        )
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self._model_input_device())
        prompt_inputs = {
            "input_ids": prompt_tensor,
            "attention_mask": torch.ones_like(prompt_tensor),
        }
        prompt_len = len(prompt_ids)
        print(f"[_generate_action_text] prompt tokens={prompt_len}", flush=True)

        # Enable output_scores to compute log_prob for PPO
        generation_kwargs = {
            "max_new_tokens": self.config.action_max_new_tokens,
            "do_sample": not deterministic,
            "temperature": self.config.temperature if not deterministic else 0.1,
            "top_p": self.config.top_p if not deterministic else 1.0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_scores": True,  # Enable for PPO log_prob computation
        }

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(**prompt_inputs, **generation_kwargs)

        sequence = outputs.sequences[0]
        action_token_ids = sequence[prompt_len:].tolist()
        action_text = self.tokenizer.decode(action_token_ids, skip_special_tokens=True).strip()
        print(f"[_generate_action_text] Generated action_text (first 500 chars): {action_text[:500]}", flush=True)

        if not action_token_ids:
            action_text = '{"action":"STOP","strategy":"STOP","chain_indices":[],"reason":"empty_generation"}'
            action_token_ids = self.tokenizer(
                action_text, add_special_tokens=False
            )["input_ids"]
            return action_text, action_token_ids, -1e-6, 0.0

        scores = getattr(outputs, "scores", None)
        if scores:
            log_prob, entropy = self._logprob_from_generate_scores(scores, action_token_ids)
        else:
            log_prob, entropy = 0.0, 0.0
        print(f"[_generate_action_text] log_prob={log_prob:.4f}, entropy={entropy:.4f}", flush=True)
        return action_text, action_token_ids, log_prob, entropy

    def _logprob_from_generate_scores(
        self, scores: List[torch.Tensor], action_token_ids: List[int]
    ) -> Tuple[float, float]:
        if not scores or not action_token_ids:
            return -1e-6, 0.0

        steps = min(len(scores), len(action_token_ids))
        token_log_probs: List[float] = []
        entropies: List[float] = []
        for i in range(steps):
            logits = torch.nan_to_num(scores[i][0].float(), nan=0.0, posinf=1e4, neginf=-1e4)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)
            probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs = probs / probs.sum().clamp(min=1e-8)
            safe_log_probs = torch.log(torch.clamp(probs, min=1e-8))
            token_id = int(action_token_ids[i])
            if token_id < 0 or token_id >= safe_log_probs.shape[-1]:
                token_log_probs.append(-20.0)
            else:
                token_log_probs.append(float(safe_log_probs[token_id].detach().cpu().item()))
            ent = torch.nan_to_num((-(probs * safe_log_probs).sum()), nan=0.0, posinf=20.0, neginf=0.0)
            ent = float(ent.detach().cpu().item())
            entropies.append(ent)
        if not token_log_probs:
            return -1e-6, 0.0
        # Keep rollout logprob scale aligned with update-time token_mean logprob.
        return float(np.mean(token_log_probs)), float(np.mean(entropies)) if entropies else 0.0

    def _estimate_value(self, prompt_text: str) -> torch.Tensor:
        prompt_text = self._render_chat_prompt(prompt_text)
        prompt_ids = self._prepare_prompt_ids(
            prompt_text,
            max_tokens=self.config.max_prompt_tokens,
            context="value_prompt",
        )
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self._model_input_device())
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                _, last_hidden = self._forward_with_last_hidden(prompt_tensor)
                hidden = last_hidden[:, -1, :]
                value = self.value_head(hidden)
        finally:
            if was_training:
                self.model.train()
        return value.squeeze(0)

    def _evaluate_action(
        self,
        prompt_text: str,
        action_text: str,
        max_action_tokens: Optional[int] = None,
        logprob_mode: str = "token_mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_text = self._render_chat_prompt(prompt_text)
        prompt_ids = self._prepare_prompt_ids(
            prompt_text,
            max_tokens=self.config.max_prompt_tokens,
            context="policy_eval_prompt",
        )
        action_ids = self.tokenizer(action_text, add_special_tokens=False)["input_ids"]
        return self._evaluate_action_from_ids(
            prompt_ids,
            action_ids,
            logprob_mode=logprob_mode,
            max_action_tokens=max_action_tokens,
        )

    def _evaluate_action_ids(
        self,
        prompt_text: str,
        action_ids: List[int],
        logprob_mode: str = "token_mean",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_text = self._render_chat_prompt(prompt_text)
        prompt_ids = self._prepare_prompt_ids(
            prompt_text,
            max_tokens=self.config.max_prompt_tokens,
            context="policy_eval_prompt",
        )
        return self._evaluate_action_from_ids(
            prompt_ids,
            action_ids,
            logprob_mode=logprob_mode,
            max_action_tokens=len(action_ids) if action_ids else None,
        )

    def _evaluate_action_from_ids(
        self,
        prompt_ids: List[int],
        action_ids: List[int],
        logprob_mode: str = "token_mean",
        max_action_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not action_ids:
            action_ids = self.tokenizer(
                '{"action":"STOP","strategy":"STOP","chain_indices":[]}',
                add_special_tokens=False,
            )["input_ids"]
        token_cap = int(max_action_tokens) if max_action_tokens is not None else int(self.config.action_max_new_tokens)
        token_cap = max(1, token_cap)
        if len(action_ids) > token_cap:
            action_ids = action_ids[:token_cap]

        input_ids = prompt_ids + list(action_ids)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._model_input_device())

        was_training = self.model.training
        grad_enabled = torch.is_grad_enabled()
        if grad_enabled and not was_training:
            self.model.train()
        if not grad_enabled and was_training:
            self.model.eval()
        try:
            outputs, last_hidden = self._forward_with_last_hidden(input_tensor)
        finally:
            if grad_enabled and not was_training:
                self.model.eval()
            if not grad_enabled and was_training:
                self.model.train()
        logits = outputs.logits  # [1, L, V]

        p_len = len(prompt_ids)
        a_len = len(action_ids)
        # logits at [p_len-1 : p_len+a_len-1] predict action tokens.
        action_logits = logits[:, p_len - 1 : p_len + a_len - 1, :]
        target = torch.tensor(action_ids, dtype=torch.long, device=action_logits.device).unsqueeze(0)

        log_probs = F.log_softmax(action_logits.float(), dim=-1)
        log_probs = torch.nan_to_num(log_probs, nan=-20.0, posinf=0.0, neginf=-20.0)
        token_log_probs = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        mode = str(logprob_mode or "token_mean").strip().lower()
        if mode in {"token_mean", "mean"}:
            seq_log_prob = token_log_probs.mean(dim=-1).squeeze(0)
        else:
            seq_log_prob = token_log_probs.sum(dim=-1).squeeze(0)
        seq_log_prob = torch.nan_to_num(seq_log_prob, nan=-20.0, posinf=0.0, neginf=-20.0)

        probs = torch.exp(log_probs)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        entropy = torch.nan_to_num(token_entropy.mean(dim=-1).squeeze(0), nan=0.0, posinf=20.0, neginf=0.0)

        state_hidden = last_hidden[:, p_len - 1, :]
        value = self.value_head(state_hidden).squeeze(0)
        eval_device = self._model_device()
        return (
            seq_log_prob.to(eval_device),
            entropy.to(eval_device),
            value.to(eval_device),
        )

    def _normalize_reason_text(
        self,
        strategy: str,
        chain_indices: List[int],
        reason: str,
        action: str = "CORRECT",
    ) -> str:
        raw_reason = str(reason or "")
        cleaned = re.sub(r"\s+", " ", raw_reason).strip()
        cleaned = cleaned.replace("```", "")
        cleaned = re.sub(r"[{}\[\]<>`\"]", "", cleaned)
        cleaned = re.sub(r"^(理由|原因|修正理由|reason)[:：]\s*", "", cleaned, flags=re.IGNORECASE)
        first_sentence = re.split(r"[。！？\n\r]", cleaned, maxsplit=1)[0].strip(" ，,；;")
        if len(first_sentence) > 48:
            first_sentence = first_sentence[:48]

        if action == "STOP" or strategy.upper() == "STOP":
            return first_sentence or "当前知识链已无明显可修正问题"

        target = "新增链" if strategy == "Add/chain" else (f"链{chain_indices}" if chain_indices else "当前知识链")
        default_issue = {
            "Add/condition_kv": "存在缺失的condition字段",
            "Add/entity_type": "存在缺失的实体类型",
            "Add/entity_value": "存在缺失的实体值",
            "Add/chain": "存在缺失的知识链",
            "Modify/condition_kv": "condition与表格不一致",
            "Modify/result": "result与表格不一致",
            "Modify/entity_type": "实体类型与链内容不一致",
            "Modify/entity_value": "实体值与链内容不一致",
            "Delete/chain": "存在冗余或错误整链",
            "Delete/condition_kv": "存在多余condition片段",
            "Delete/result": "存在多余或错误result",
            "Delete/entity_type": "存在多余实体类型",
            "Delete/entity_value": "存在多余实体值",
        }.get(strategy, "存在待修正问题")
        issue_text = first_sentence or default_issue
        structured = f"目标:{target}；问题:{issue_text}"
        if len(structured) > 96:
            structured = structured[:96]
        return structured

    def _parse_action(self, text: str, round_idx: int = 0) -> Dict[str, Any]:
        all_json_candidates: List[Dict[str, Any]] = []
        for candidate in self._extract_all_json(text):
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        all_json_candidates.append(item)
            elif isinstance(candidate, dict):
                all_json_candidates.append(candidate)

        try:
            extracted = self._extract_json(text)
            parsed_obj = json.loads(extracted)
            if isinstance(parsed_obj, list):
                for item in parsed_obj:
                    if isinstance(item, dict):
                        all_json_candidates.append(item)
            elif isinstance(parsed_obj, dict):
                all_json_candidates.append(parsed_obj)
        except Exception:
            pass

        parsed: Dict[str, Any] = {}
        # Prefer candidate matching current round.
        for candidate in all_json_candidates:
            try:
                if int(candidate.get("round", -1)) == int(round_idx):
                    parsed = candidate
                    break
            except Exception:
                continue
        # Then choose candidate with action/strategy.
        if not parsed:
            for candidate in all_json_candidates:
                if "action" in candidate and "strategy" in candidate:
                    parsed = candidate
                    break
        # Finally choose any candidate containing strategy.
        if not parsed:
            for candidate in all_json_candidates:
                if "strategy" in candidate:
                    parsed = candidate
                    break

        try:
            if not isinstance(parsed, dict):
                raise ValueError("action is not dict")

            action = str(parsed.get("action", "")).strip().upper()
            strategy = str(parsed.get("strategy", "")).strip()
            if not action:
                action = "STOP" if strategy.upper() == "STOP" else "CORRECT"

            raw_indices = parsed.get("chain_indices", [])
            indices: List[int] = []
            seen_indices = set()
            if isinstance(raw_indices, list):
                for x in raw_indices:
                    try:
                        ix = int(x)
                        if ix >= 0 and ix not in seen_indices:
                            indices.append(ix)
                            seen_indices.add(ix)
                            if len(indices) >= 7:
                                break
                    except Exception:
                        continue

            reason = str(parsed.get("reason", "")) if "reason" in parsed else ""
            reason = reason.strip()
            if len(reason) > 256:
                reason = reason[:256]

            strategy_out = strategy if strategy else ("STOP" if action == "STOP" else "")
            strategy_out = re.sub(r"[，。,;；]+$", "", strategy_out).strip()
            for allowed in self.valid_strategies:
                if strategy_out.lower() == allowed.lower():
                    strategy_out = allowed
                    break
            if strategy_out.upper() == "STOP":
                action = "STOP"
            if len(strategy_out) > 128:
                strategy_out = strategy_out[:128]

            # STOP 和 Add/chain 强制清空 chain_indices，与 step8.1 对齐
            if strategy_out.upper() == "STOP" or strategy_out == "Add/chain":
                indices = []

            normalized_reason = self._normalize_reason_text(
                strategy=strategy_out or strategy,
                chain_indices=indices,
                reason=reason,
                action=action,
            )

            if action == "STOP":
                return {
                    "action": "STOP",
                    "strategy": "STOP",
                    "chain_indices": [],
                    "reason": normalized_reason,
                }

            # Invalid corrective action without strategy wastes teacher calls.
            if not strategy_out:
                return self._invalid_action_payload("missing_strategy", chain_indices=indices)
            if strategy_out not in self.valid_strategies:
                return self._invalid_action_payload(
                    "invalid_strategy",
                    strategy=strategy_out,
                    chain_indices=indices,
                )
            # For most corrective strategies, missing indices usually leads to no-op teacher calls.
            if action != "INVALID":
                strategy_low = strategy_out.lower()
                requires_indices = strategy_low not in {"add/chain"}
                if requires_indices and not indices:
                    return self._invalid_action_payload(
                        "missing_indices",
                        strategy=strategy_out,
                        chain_indices=[],
                    )

            return {
                "action": "CORRECT",
                "strategy": strategy_out,
                "chain_indices": indices,
                "reason": normalized_reason,
            }
        except Exception:
            return self._invalid_action_payload("parse_failed")

    def _canonical_action_text(self, action: Dict[str, Any]) -> str:
        """Serialize parsed action to a compact, bounded JSON string for PPO scoring."""
        payload = {
            "action": str(action.get("action", "STOP")),
            "strategy": str(action.get("strategy", "STOP")),
            "chain_indices": action.get("chain_indices", []),
            "reason": str(action.get("reason", "")),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _extract_all_json(self, text: str) -> List[Dict]:
        """提取文本中所有有效的JSON对象，支持更复杂的嵌套结构"""
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

        # 增强的正则：支持更深的嵌套，使用栈来匹配平衡的括号
        import re

        # 方法1：尝试找到所有 { ... } 块，然后用栈验证平衡
        i = 0
        while i < len(cleaned):
            if cleaned[i] == '{':
                # 找到匹配的 }
                end = self._find_matching_brace(cleaned, i)
                if end is not None:
                    candidate = cleaned[i:end+1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            results.append(parsed)
                    except Exception:
                        pass
                    i = end + 1
                else:
                    i += 1
            else:
                i += 1

        return results

    def _extract_json(self, text: str) -> str:
        cleaned = text.strip()

        if "```" in cleaned:
            blocks = cleaned.split("```")
            for block in blocks:
                candidate = block.strip()
                if candidate.lower().startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") or candidate.startswith("["):
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        pass

        obj_start = cleaned.find("{")
        obj_end = cleaned.rfind("}")
        if obj_start != -1 and obj_end > obj_start:
            candidate = cleaned[obj_start : obj_end + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")
        if arr_start != -1 and arr_end > arr_start:
            candidate = cleaned[arr_start : arr_end + 1]
            try:
                json.loads(candidate)
                return candidate
            except Exception:
                pass

        return cleaned

    def _find_matching_brace(self, text: str, start_idx: int) -> Optional[int]:
        """Find matching closing brace for a JSON object starting at start_idx."""
        if start_idx < 0 or start_idx >= len(text) or text[start_idx] != "{":
            return None
        depth = 0
        in_string = False
        escaped = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _recover_truncated_initial_kc(self, text: str) -> Optional[Dict[str, Any]]:
        """Recover complete knowledge_mining entries from truncated initial-KC output."""
        cleaned = text.strip()
        km_pos = cleaned.find("\"knowledge_mining\"")
        if km_pos < 0:
            return None

        arr_start = cleaned.find("[", km_pos)
        if arr_start < 0:
            return None

        items: List[Dict[str, Any]] = []
        i = arr_start + 1
        while i < len(cleaned):
            while i < len(cleaned) and cleaned[i] in " \t\r\n,":
                i += 1
            if i >= len(cleaned) or cleaned[i] == "]":
                break
            if cleaned[i] != "{":
                break

            end_idx = self._find_matching_brace(cleaned, i)
            if end_idx is None:
                # Last object is incomplete due to truncation.
                break

            candidate = cleaned[i : end_idx + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                break

            if isinstance(obj, dict) and "knowledge_chain" in obj:
                items.append(obj)
            i = end_idx + 1

        if not items:
            return None
        return {"knowledge_mining": items}

    def _prepare_prompt_ids(self, prompt_text: str, max_tokens: int, context: str) -> List[int]:
        """Convert prompt text to token ids with step8.1-style tokenizer truncation."""
        effective_max_tokens = self._effective_prompt_max_tokens(max_tokens)
        mode = str(getattr(self.config, "prompt_overflow_mode", "truncate")).strip().lower()
        kwargs: Dict[str, Any] = {
            "add_special_tokens": False,
            "truncation": True,
            "max_length": effective_max_tokens,
        }
        if mode == "error":
            raw_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            if len(raw_ids) > effective_max_tokens:
                raise RuntimeError(
                    f"[{context}] prompt token length {len(raw_ids)} exceeds max_prompt_tokens={effective_max_tokens}; "
                    "set prompt_overflow_mode=truncate or increase max_prompt_tokens."
                )
            ids = raw_ids
        else:
            ids = self.tokenizer(prompt_text, **kwargs)["input_ids"]
            raw_len = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
            if raw_len > effective_max_tokens:
                print(
                    f"[{context}] prompt too long ({raw_len}), truncated to {effective_max_tokens} tokens (tokenizer).",
                    flush=True,
                )
        if not ids:
            return [self.tokenizer.eos_token_id]
        return ids

    def _forward_with_last_hidden(self, input_tensor: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """
        Forward once and capture last hidden states via lm_head input hook.
        This avoids output_hidden_states=True (large memory overhead).
        """
        captured: Dict[str, torch.Tensor] = {}
        lm_head = self.model.get_output_embeddings()
        handle = None

        if lm_head is not None:
            def _hook(_module, module_input, _module_output):
                if "hidden" not in captured and module_input:
                    captured["hidden"] = module_input[0]

            handle = lm_head.register_forward_hook(_hook)

        try:
            outputs = self.model(input_ids=input_tensor, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()

        hidden = captured.get("hidden")
        if hidden is not None:
            return outputs, hidden

        # Safe fallback if hook path fails on some model implementations.
        outputs = self.model(input_ids=input_tensor, output_hidden_states=True, use_cache=False)
        return outputs, outputs.hidden_states[-1]

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self.device)

    def _model_input_device(self) -> torch.device:
        try:
            emb = self.model.get_input_embeddings()
            if emb is not None:
                return next(emb.parameters()).device
        except Exception:
            pass
        return self._model_device()

    def _input_device_for_model(self, model: Any) -> torch.device:
        try:
            emb = model.get_input_embeddings()
            if emb is not None:
                return next(emb.parameters()).device
        except Exception:
            pass
        try:
            return next(model.parameters()).device
        except StopIteration:
            return self._model_device()

    def _model_output_device(self) -> torch.device:
        try:
            head = self.model.get_output_embeddings()
            if head is not None:
                return next(head.parameters()).device
        except Exception:
            pass
        return self._model_device()

    def _prepare_ref_kl_ids(
        self,
        prompt_text: str,
        action_text: str,
        action_token_ids: Optional[List[int]] = None,
        max_action_tokens: Optional[int] = None,
    ) -> Tuple[List[int], List[int]]:
        prompt_text = self._render_chat_prompt(prompt_text)
        prompt_ids = self._prepare_prompt_ids(
            prompt_text,
            max_tokens=self.config.max_prompt_tokens,
            context="ref_kl_prompt",
        )
        if isinstance(action_token_ids, list) and action_token_ids:
            action_ids = list(action_token_ids)
        else:
            action_ids = self.tokenizer(action_text, add_special_tokens=False)["input_ids"]

        if not action_ids:
            action_ids = []
        else:
            token_cap = int(max_action_tokens) if max_action_tokens is not None else int(self.config.action_max_new_tokens)
            token_cap = max(1, token_cap)
            if len(action_ids) > token_cap:
                action_ids = action_ids[:token_cap]

        total_max_tokens = int(getattr(self.config, "ref_kl_max_tokens", 2048))
        if total_max_tokens > 0:
            if len(action_ids) >= total_max_tokens:
                action_ids = action_ids[: max(1, total_max_tokens - 1)]
            prompt_budget = max(1, total_max_tokens - len(action_ids))
            if len(prompt_ids) > prompt_budget:
                prompt_ids = prompt_ids[-prompt_budget:]
        return prompt_ids, action_ids

    def _compute_ref_kl_batch(self, batch_steps: List[Dict[str, Any]]) -> torch.Tensor:
        if self.ref_model is None or not batch_steps:
            return torch.tensor(0.0, device=self._model_device())

        prepared: List[Tuple[List[int], List[int]]] = []
        for step in batch_steps:
            step_token_cap_raw = step.get("action_token_cap", self.config.action_max_new_tokens)
            try:
                step_token_cap = int(step_token_cap_raw)
            except Exception:
                step_token_cap = int(self.config.action_max_new_tokens)
            p_ids, a_ids = self._prepare_ref_kl_ids(
                prompt_text=step["prompt_text"],
                action_text=step["action_text"],
                action_token_ids=step.get("action_token_ids"),
                max_action_tokens=step_token_cap,
            )
            if a_ids:
                prepared.append((p_ids, a_ids))

        if not prepared:
            return torch.tensor(0.0, device=self._model_device())

        seq_ids: List[List[int]] = [p + a for p, a in prepared]
        prompt_lens: List[int] = [len(p) for p, _a in prepared]
        action_ids_list: List[List[int]] = [a for _p, a in prepared]
        max_len = max(len(x) for x in seq_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_device = self._model_input_device()
        input_tensor = torch.full(
            (len(seq_ids), max_len),
            int(pad_id),
            dtype=torch.long,
            device=input_device,
        )
        for i, ids in enumerate(seq_ids):
            input_tensor[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=input_device)

        was_training = self.model.training
        grad_enabled = torch.is_grad_enabled()
        if grad_enabled and not was_training:
            self.model.train()
        if not grad_enabled and was_training:
            self.model.eval()
        try:
            outputs_curr = self.model(input_ids=input_tensor, use_cache=False)
        finally:
            if grad_enabled and not was_training:
                self.model.eval()
            if not grad_enabled and was_training:
                self.model.train()

        logits_curr = outputs_curr.logits.float()
        ref_input = input_tensor.to(self._ref_model_input_device())
        with torch.no_grad():
            outputs_ref = self.ref_model(input_ids=ref_input, use_cache=False)
            logits_ref = outputs_ref.logits.float()

        kl_vals: List[torch.Tensor] = []
        for i, (p_len, action_ids) in enumerate(zip(prompt_lens, action_ids_list)):
            a_len = len(action_ids)
            if a_len <= 0:
                continue

            curr_slice = logits_curr[i : i + 1, p_len - 1 : p_len + a_len - 1, :]
            log_probs_curr = F.log_softmax(curr_slice, dim=-1)
            log_probs_curr = torch.nan_to_num(log_probs_curr, nan=-20.0, posinf=0.0, neginf=-20.0)
            probs_curr = torch.exp(log_probs_curr)
            probs_curr = torch.nan_to_num(probs_curr, nan=0.0, posinf=1.0, neginf=0.0)
            probs_curr = probs_curr / probs_curr.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            ref_slice = logits_ref[i : i + 1, p_len - 1 : p_len + a_len - 1, :].to(
                curr_slice.device, non_blocking=True
            )
            log_probs_ref = F.log_softmax(ref_slice, dim=-1)
            log_probs_ref = torch.nan_to_num(log_probs_ref, nan=-20.0, posinf=0.0, neginf=-20.0)

            token_kl = (probs_curr * (log_probs_curr - log_probs_ref)).sum(dim=-1)
            token_kl = torch.nan_to_num(token_kl, nan=0.0, posinf=50.0, neginf=0.0)
            kl_i = token_kl.mean()
            kl_vals.append(kl_i)

        if not kl_vals:
            return torch.tensor(0.0, device=self._model_device())
        kl = torch.stack(kl_vals).mean()
        # Clear cache after ref_kl computation to free GPU memory
        del outputs_curr, outputs_ref, logits_curr, logits_ref, input_tensor, ref_input
        torch.cuda.empty_cache()
        return torch.nan_to_num(kl, nan=0.0, posinf=50.0, neginf=0.0)

    def _compute_ref_kl(self, prompt_text: str, action_text: str) -> torch.Tensor:
        return self._compute_ref_kl_batch([{"prompt_text": prompt_text, "action_text": action_text}])

    def _ref_model_device(self) -> torch.device:
        if self.ref_model is None:
            return self._model_device()
        try:
            dev = next(self.ref_model.parameters()).device
            if dev.type == "meta":
                return self._model_device()
            return dev
        except StopIteration:
            return self._model_device()

    def _ref_model_input_device(self) -> torch.device:
        if self.ref_model is None:
            return self._model_input_device()
        try:
            emb = self.ref_model.get_input_embeddings()
            if emb is not None:
                return next(emb.parameters()).device
        except Exception:
            pass
        return self._ref_model_device()
