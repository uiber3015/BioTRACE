#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step9 PPO environment for multi-turn KC correction.

Design goals:
1) Keep transition logic consistent with Step8 evaluation.
2) Do NOT leak score / GT signal into state.
3) Keep episodes robust under teacher failures (API timeout, prompt miss, parse fail).
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from step7.r_rule_evaluator_final import compute_r_rule
from step9.step8_reference import compact_summary_for_history, normalize_kc_payload


@dataclass
class RewardConfig:
    # Main reward signal: delta-R.
    delta_pos_scale: float = 6.0
    delta_neg_scale: float = 8.0

    # Safety: strong degrade rollback and penalty.
    rollback_threshold: float = 0.3
    rollback_extra_penalty: float = 1.0

    # High-quality sample behavior.
    high_score_threshold: float = 0.9
    high_score_stop_bonus: float = 1.0
    premature_stop_penalty: float = 0.8
    high_score_continue_penalty: float = 0.1
    teacher_stop_high_score_bonus: float = 0.15

    # Large edit penalty when not improving.
    large_edit_threshold: int = 3
    large_edit_penalty: float = 0.05

    # Turn-level control and final objective.
    step_penalty: float = 0.005
    no_progress_penalty: float = 0.02
    norm_equivalent_noop_penalty: float = 0.08
    positive_correction_bonus: float = 0.05
    final_score_scale: float = 0.20
    no_progress_eps: float = 1e-3
    max_stagnant_rounds: int = 3
    ineffective_termination_penalty: float = 0.12
    invalid_action_penalty: float = 0.25
    max_consecutive_teacher_stops: int = 2


class CorrectionEnvironment:
    """
    Multi-turn correction environment for PPO.

    State (score-free):
    - raw table_content
    - full current_kc
    - history_context (strategy-only text, no scores)
    - round_num / max_rounds
    """

    def __init__(
        self,
        corrector: Any,
        max_rounds: int = 4,
        stop_threshold: float = 0.9,
        reward_config: Optional[RewardConfig] = None,
        focus_window_size: int = 7,
        table_summary_char_budget: int = 1800,
        kc_summary_char_budget: int = 3600,
    ) -> None:
        self.corrector = corrector
        self.max_rounds = max_rounds
        self.stop_threshold = stop_threshold
        self.reward_cfg = reward_config or RewardConfig()
        self.focus_window_size = max(1, int(focus_window_size))
        self.table_summary_char_budget = max(400, int(table_summary_char_budget))
        self.kc_summary_char_budget = max(800, int(kc_summary_char_budget))

        self.table_content: str = ""
        self.ground_truth: Dict[str, Any] = {"knowledge_mining": []}
        self.current_kc: Dict[str, Any] = {"knowledge_mining": []}
        self.history_context: str = ""
        self.round_num: int = 0
        self.stagnant_rounds: int = 0
        self.consecutive_teacher_stops: int = 0
        self.last_teacher_summary: str = ""

        self.initial_r_rule: float = 0.0
        self.current_r_rule: float = 0.0
        self.best_r_rule: float = 0.0
        self.best_kc: Dict[str, Any] = {"knowledge_mining": []}
        self.last_target_indices: List[int] = []
        self.last_action_type: str = ""
        self.last_strategy: str = ""

        self.trajectory: Dict[str, Any] = {}
        self.supported_strategies = {
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
            # Backward-compat alias used in older prompts/models.
            "Modify/entity_recognition",
        }

    def reset(
        self,
        table_content: str,
        initial_kc: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.table_content = table_content or ""
        self.current_kc = initial_kc if isinstance(initial_kc, dict) else {"knowledge_mining": []}
        self.ground_truth = ground_truth if isinstance(ground_truth, dict) else {"knowledge_mining": []}
        if "knowledge_mining" not in self.ground_truth:
            self.ground_truth = {"knowledge_mining": self.ground_truth}

        self.history_context = ""
        self.round_num = 0
        self.stagnant_rounds = 0
        self.consecutive_teacher_stops = 0
        self.last_teacher_summary = ""
        self.last_target_indices = []
        self.last_action_type = ""
        self.last_strategy = ""
        # 与 step8.1 对齐：用于判断连续无进展停机
        self.prev_strategy: Optional[str] = None
        self.prev_chain_indices: Optional[List[int]] = None
        self.prev_kc_changed: Optional[bool] = None

        self.initial_r_rule = self._compute_r_rule(self.current_kc)
        self.current_r_rule = self.initial_r_rule
        self.best_r_rule = self.initial_r_rule
        self.best_kc = copy.deepcopy(self.current_kc)

        self.trajectory = {
            "table_content": self.table_content,
            "initial_kc": copy.deepcopy(self.current_kc),
            "initial_r_rule": self.initial_r_rule,
            "steps": [],
            "rewards": [],
        }
        return self._get_state()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        action_type = str(action.get("action", "CORRECT")).upper().strip()
        strategy = str(action.get("strategy", "")).strip()
        reason = str(action.get("reason", "")).strip()
        chain_indices = self._normalize_indices(action.get("chain_indices", []))
        km_items = self.current_kc.get("knowledge_mining", [])
        km_len = len(km_items) if isinstance(km_items, list) else 0
        chain_indices = [idx for idx in chain_indices if idx < km_len]
        self.last_target_indices = list(chain_indices)
        self.last_action_type = action_type
        self.last_strategy = strategy
        strategy_lower = strategy.lower()
        if (
            action_type == "CORRECT"
            and strategy in self.supported_strategies
            and strategy_lower not in {"add/chain", "stop"}
            and not chain_indices
        ):
            action_type = "INVALID"
            reason = "missing_valid_indices"

        print(f"[ENV step] action_type={action_type}, strategy={strategy}, round={self.round_num}", flush=True)
        print(f"[ENV step] action JSON: {json.dumps(action, ensure_ascii=False)[:500]}", flush=True)
        print(f"[ENV step] current_kc (first 500 chars): {json.dumps(self.current_kc, ensure_ascii=False)[:500]}", flush=True)

        if action_type not in {"CORRECT", "STOP", "INVALID"}:
            action_type = "STOP" if strategy.upper() == "STOP" else "INVALID"

        r_before = self.current_r_rule
        kc_before = copy.deepcopy(self.current_kc)
        rolled_back = False
        rollback_to_best = False
        raw_delta = 0.0
        effective_delta = 0.0
        stop_reason: Optional[str] = None
        teacher_summary = ""
        teacher_failed = False
        teacher_error = ""
        teacher_stop = False
        teacher_no_change = False
        teacher_invalid_response = False
        invalid_action = False
        kc_changed = False
        history_outcome = ""
        r_after_for_info = r_before
        norm_exact_same = False
        norm_equivalent = False

        if action_type == "STOP" or strategy.upper() == "STOP":
            done = True
            stop_reason = "model_stop"
            print(f"[ENV step] STOP action, r_before={r_before:.4f}", flush=True)
            reward = self._compute_reward(
                action_type="STOP",
                r_before=r_before,
                r_after=self.current_r_rule,
                raw_delta=0.0,
                effective_delta=0.0,
                chain_indices=[],
                done=True,
                rolled_back=False,
                rollback_to_best=False,
                teacher_stop=False,
                stop_reason=stop_reason,
                norm_equivalent=False,
            )
            info = {
                "action": "STOP",
                "strategy": "STOP",
                "chain_indices": [],
                "r_before": r_before,
                "r_after": self.current_r_rule,
                "final_r_after_aggregation": self.current_r_rule,
                "best_r_rule": self.best_r_rule,
                "raw_delta": 0.0,
                "effective_delta": 0.0,
                "rolled_back": False,
                "rollback_to_best": False,
                "stop_reason": stop_reason,
                "reward": reward,
            }
            self._record_step(info, reward)
            return self._get_state(), reward, done, info

        if action_type == "INVALID":
            invalid_action = True
            teacher_error = reason or "invalid_action"
            teacher_summary = teacher_error
            history_outcome = "INVALID_ACTION"
        elif strategy not in self.supported_strategies:
            action_type = "INVALID"
            invalid_action = True
            teacher_error = f"invalid_strategy:{strategy}"
            teacher_summary = teacher_error
            history_outcome = "INVALID_ACTION"
            print(f"[ENV step] Invalid strategy, skip teacher call: {strategy}", flush=True)
        else:
            try:
                print("[ENV step] Calling DeepSeek corrector...", flush=True)
                corrected_kc, summary = self.corrector.correct(
                    self.table_content,
                    self.current_kc,
                    {
                        "action": action_type,
                        "strategy": strategy,
                        "chain_indices": chain_indices,
                        "reason": reason,
                    },
                    self.last_teacher_summary,
                )

                teacher_summary = summary or ""
                print(f"[ENV step] DeepSeek returned summary (first 500): {teacher_summary[:500] if teacher_summary else 'None'}", flush=True)
                print(f"[ENV step] corrected_kc (first 500): {json.dumps(corrected_kc, ensure_ascii=False)[:500] if corrected_kc else 'None'}", flush=True)
                teacher_stop = isinstance(teacher_summary, str) and teacher_summary.strip().upper().startswith("TEACHER_STOP")
                teacher_no_change = isinstance(teacher_summary, str) and teacher_summary.strip().upper() == "TEACHER_NO_CHANGE"
                teacher_invalid_response = isinstance(teacher_summary, str) and teacher_summary.strip().upper().startswith("INVALID_RESPONSE")

                if not isinstance(corrected_kc, dict):
                    teacher_failed = True
                    teacher_error = "invalid_corrected_kc"
                    corrected_kc = kc_before
                if isinstance(teacher_summary, str) and teacher_summary.startswith("[TEACHER_"):
                    teacher_failed = True
                    teacher_error = teacher_summary[:200]

                if teacher_failed:
                    teacher_summary = teacher_error or teacher_summary
                    history_outcome = "TEACHER_FAILED"
                elif teacher_stop:
                    history_outcome = "TEACHER_STOP"
                elif teacher_no_change:
                    history_outcome = "NO_CHANGE"
                elif teacher_invalid_response:
                    history_outcome = "INVALID"
                else:
                    r_after_raw = self._compute_r_rule(corrected_kc)
                    raw_delta = r_after_raw - r_before
                    r_after_for_info = r_after_raw
                    print(f"[ENV step] r_before={r_before:.4f}, r_after_raw={r_after_raw:.4f}, raw_delta={raw_delta:+.4f}", flush=True)

                    # 新增：区分 post-normalize exact same 和 normalize-equivalent no-op
                    try:
                        from step9.step8_reference import normalize_kc_payload
                        kc_before_norm = normalize_kc_payload(kc_before)
                        kc_after_norm = normalize_kc_payload(corrected_kc)
                        norm_exact_same = self._kc_payload_equal(kc_before, corrected_kc)
                        norm_equivalent = not self._kc_payload_equal(kc_before, corrected_kc) and self._kc_payload_equal(kc_before_norm, kc_after_norm)
                    except Exception:
                        norm_exact_same = False
                        norm_equivalent = False

                    if raw_delta < -self.reward_cfg.rollback_threshold:
                        rolled_back = True
                        self.current_kc = kc_before
                        self.current_r_rule = r_before
                        effective_delta = 0.0
                        rollback_note = (
                            f"[ROLLBACK] 上轮策略 {strategy} 退化 {r_before:.4f} -> {r_after_raw:.4f}，"
                            "已回滚为之前的 KC，当前轮仍可继续修正。"
                        )
                        teacher_summary = rollback_note
                        history_outcome = "ROLLBACK"
                    else:
                        self.current_kc = corrected_kc
                        self.current_r_rule = r_after_raw
                        effective_delta = self.current_r_rule - r_before
                        kc_changed = not self._kc_payload_equal(kc_before, corrected_kc)
                        history_outcome = "CHANGED" if kc_changed else "NO_CHANGE"
                        if self.current_r_rule > self.best_r_rule:
                            self.best_r_rule = self.current_r_rule
                            self.best_kc = copy.deepcopy(self.current_kc)

            except Exception as exc:
                teacher_failed = True
                teacher_error = f"{type(exc).__name__}: {str(exc)}"
                print(f"[ENV ERROR] Exception in corrector: {teacher_error}", flush=True)
                import traceback

                print(f"[ENV ERROR] Traceback: {traceback.format_exc()}", flush=True)
                teacher_summary = teacher_error
                history_outcome = "TEACHER_FAILED"
                norm_exact_same = False
                norm_equivalent = False

        # 如果在非 else 分支（teacher_stop/teacher_failed/rollback等）中没有定义，设为 False
        try:
            _ = norm_exact_same
        except NameError:
            norm_exact_same = False
            norm_equivalent = False

        self.round_num += 1
        no_change_outcomes = {
            "INVALID_ACTION",
            "INVALID",
            "ROLLBACK",
            "TEACHER_FAILED",
            "TEACHER_STOP",
        }
        if teacher_no_change:
            no_change_outcomes.add("NO_CHANGE")

        repeat_strategy = (
            self.prev_strategy is not None
            and strategy == self.prev_strategy
            and chain_indices == (self.prev_chain_indices or [])
        )
        repeat_no_change = repeat_strategy and self.prev_kc_changed is False

        if history_outcome == "ROLLBACK":
            self.consecutive_teacher_stops = 0
            self.stagnant_rounds = 0
            kc_changed = False
        elif history_outcome == "TEACHER_STOP":
            kc_changed = False
            self.consecutive_teacher_stops += 1
            self.stagnant_rounds += 1
        elif history_outcome in no_change_outcomes:
            self.consecutive_teacher_stops = 0
            kc_changed = False
            if repeat_no_change:
                self.stagnant_rounds += 1
            else:
                self.stagnant_rounds = 0
        else:
            self.consecutive_teacher_stops = 0
            delta = self.current_r_rule - r_before
            if abs(delta) < self.reward_cfg.no_progress_eps and repeat_strategy and not kc_changed:
                self.stagnant_rounds += 1
            else:
                self.stagnant_rounds = 0

        self.prev_strategy = strategy
        self.prev_chain_indices = list(chain_indices) if chain_indices else []
        self.prev_kc_changed = kc_changed

        if history_outcome in no_change_outcomes or history_outcome == "ROLLBACK":
            # 高分 teacher_stop 允许直接终止；低分 teacher_stop 若连续出现，
            # 说明策略层卡在不可执行或低价值修正上，应尽快截断无效 rollout。
            if teacher_stop and r_before >= self.stop_threshold:
                done = True
                stop_reason = "teacher_stop"
            elif teacher_stop and self.consecutive_teacher_stops >= self.reward_cfg.max_consecutive_teacher_stops:
                done = True
                stop_reason = "teacher_stop_stagnation"
            elif self.stagnant_rounds >= self.reward_cfg.max_stagnant_rounds:
                done = True
                stop_reason = "stagnation"
            elif self.round_num >= self.max_rounds:
                done = True
                stop_reason = "max_rounds"
            else:
                done = False
            if teacher_stop and not done:
                print(
                    f"[ENV step] TEACHER_STOP (low score): continuing to next round "
                    f"(r={r_before:.4f} < {self.stop_threshold}, consecutive={self.consecutive_teacher_stops})",
                    flush=True,
                )
        else:
            if self.current_r_rule >= self.stop_threshold:
                done = True
                stop_reason = "auto_threshold"
            elif self.stagnant_rounds >= self.reward_cfg.max_stagnant_rounds:
                done = True
                stop_reason = "stagnation"
            elif self.round_num >= self.max_rounds:
                done = True
                stop_reason = "max_rounds"
            else:
                done = False

        self.last_teacher_summary = teacher_summary or ""

        if done:
            rollback_to_best = self.best_r_rule > r_after_for_info + self.reward_cfg.no_progress_eps

        reward = self._compute_reward(
            action_type=action_type,
            r_before=r_before,
            r_after=self.current_r_rule,
            raw_delta=raw_delta,
            effective_delta=effective_delta,
            chain_indices=chain_indices,
            done=done,
            rolled_back=rolled_back,
            rollback_to_best=rollback_to_best,
            teacher_stop=teacher_stop,
            stop_reason=stop_reason,
            norm_equivalent=norm_equivalent,
        )
        self._append_history_line(
            action_type,
            strategy,
            chain_indices,
            outcome=history_outcome or ("CHANGED" if kc_changed else "NO_CHANGE"),
            kc_changed=kc_changed,
        )

        info = {
            "action": action_type,
            "strategy": strategy,
            "chain_indices": chain_indices,
            "teacher_summary": teacher_summary,
            "teacher_failed": teacher_failed,
            "teacher_stop": teacher_stop,
            "teacher_no_change": teacher_no_change,
            "teacher_invalid_response": teacher_invalid_response,
            "teacher_error": teacher_error,
            "invalid_action": invalid_action,
            "r_before": r_before,
            "r_after": r_after_for_info,
            "final_r_after_aggregation": self.current_r_rule if done else r_after_for_info,
            "best_r_rule": self.best_r_rule,
            "raw_delta": raw_delta,
            "effective_delta": effective_delta,
            "rolled_back": rolled_back,
            "rollback_to_best": rollback_to_best,
            "stop_reason": stop_reason,
            "reward": reward,
            "r_score": r_before,  # 用于判断是否是高分 teacher_stop
        }

        # 6 类统计输出
        step_category = "UNKNOWN"
        if teacher_stop:
            step_category = "TEACHER_STOP"
        elif teacher_invalid_response:
            step_category = "INVALID_RESPONSE"
        elif rolled_back:
            step_category = "ROLLBACK"
        elif teacher_failed:
            step_category = "TEACHER_FAILED"
        elif norm_exact_same:
            step_category = "POST_NORM_EXACT_SAME"
        elif norm_equivalent:
            step_category = "NORM_EQUIVALENT_NOOP"
        elif raw_delta > 1e-6:
            step_category = "REAL_IMPROVEMENT"
        else:
            step_category = "NO_CHANGE"

        info["norm_exact_same"] = norm_exact_same
        info["norm_equivalent"] = norm_equivalent
        info["step_category"] = step_category

        print("[ENV step] === STEP COMPLETE ===", flush=True)
        print(
            f"[ENV step] r_before={r_before:.4f}, r_after={self.current_r_rule:.4f}, "
            f"raw_delta={raw_delta:+.4f}, effective_delta={effective_delta:+.4f}",
            flush=True,
        )
        print(
            f"[ENV step] rolled_back={rolled_back}, rollback_to_best={rollback_to_best}, "
            f"teacher_failed={teacher_failed}, "
            f"done={done}, stop_reason={stop_reason}",
            flush=True,
        )
        print(f"[ENV step] reward={reward:.4f}", flush=True)
        print(f"[ENV step] step_category={step_category} (norm_exact_same={norm_exact_same}, norm_equiv={norm_equivalent})", flush=True)
        print(f"[ENV step] current_kc (full): {json.dumps(self.current_kc, ensure_ascii=False)[:1000]}", flush=True)

        self._record_step(info, reward)
        return self._get_state(), reward, done, info

    def _kc_payload_equal(self, kc_a: Dict[str, Any], kc_b: Dict[str, Any]) -> bool:
        try:
            return json.dumps(kc_a, ensure_ascii=False, sort_keys=True) == json.dumps(
                kc_b,
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            return kc_a == kc_b

    def _append_history_line(
        self,
        action_type: str,
        strategy: str,
        chain_indices: List[int],
        outcome: str = "",
        kc_changed: bool = False,
    ) -> None:
        line = (
            f"Round {self.round_num}: strategy={strategy}, "
            f"indices={chain_indices}, action={action_type}, "
            f"kc_changed={'yes' if kc_changed else 'no'}, outcome={outcome}"
        )
        self.history_context += f"\n{line}"

    def _policy_last_teacher_summary(self) -> str:
        summary = str(self.last_teacher_summary or "").strip()
        if not summary:
            return ""
        try:
            return compact_summary_for_history(summary, max_chars=500)
        except Exception:
            summary = " ".join(summary.split())
            return summary[:500]

    def _get_state(self) -> Dict[str, Any]:
        return {
            "table_content": self.table_content,
            "current_kc": copy.deepcopy(self.current_kc),
            "history_context": self.history_context,
            "last_teacher_summary": self._policy_last_teacher_summary(),
            "round_num": self.round_num,
            "max_rounds": self.max_rounds,
        }

    def _compute_r_rule(self, kc: Dict[str, Any]) -> float:
        # 与 step8.1 评估对齐：评分前先归一化 KC
        normalized_kc = normalize_kc_payload(kc)
        sample = {
            "ground_truth": self.ground_truth,
            "generated_json": normalized_kc,
        }
        try:
            result = compute_r_rule(sample, "generated_json")
            return float(result.get("R_rule", 0.0))
        except Exception:
            return 0.0

    def _compute_reward(
        self,
        action_type: str,
        r_before: float,
        r_after: float,
        raw_delta: float,
        effective_delta: float,
        chain_indices: List[int],
        done: bool,
        rolled_back: bool,
        rollback_to_best: bool,
        teacher_stop: bool,
        stop_reason: Optional[str],
        norm_equivalent: bool,
    ) -> float:
        cfg = self.reward_cfg
        reward = 0.0
        # Only true rollback should force reward_delta to use the raw degraded score.
        # Episode-end aggregation back to best_kc is a reporting/terminal policy choice,
        # not a rollback of the current action itself.
        used_rollback_delta = bool(rolled_back and raw_delta < -cfg.no_progress_eps)
        reward_delta = raw_delta if used_rollback_delta else effective_delta

        # Main signal.
        if reward_delta >= 0:
            reward += cfg.delta_pos_scale * reward_delta
        else:
            reward += cfg.delta_neg_scale * reward_delta

        # Safety penalties: 只在强退化 (delta < -0.3) 时回滚才惩罚
        # 与 step8.1 评估逻辑对齐
        if rolled_back:
            reward -= cfg.rollback_extra_penalty
        elif raw_delta < -cfg.rollback_threshold:
            reward -= cfg.rollback_extra_penalty
        # 移除任意负 delta 回滚的额外惩罚，与 step8.1 对齐

        # Stop/continue behavior.
        if action_type == "STOP":
            stop_eval_score = max(r_before, r_after)
            if stop_eval_score >= cfg.high_score_threshold:
                reward += cfg.high_score_stop_bonus
            else:
                reward -= cfg.premature_stop_penalty * (
                    cfg.high_score_threshold - stop_eval_score
                )
        elif action_type == "INVALID":
            reward -= cfg.invalid_action_penalty
        else:
            reward -= cfg.step_penalty
            if abs(reward_delta) <= cfg.no_progress_eps:
                reward -= cfg.no_progress_penalty
            elif reward_delta > cfg.no_progress_eps:
                reward += cfg.positive_correction_bonus
            if norm_equivalent:
                reward -= cfg.norm_equivalent_noop_penalty
            if r_before >= cfg.high_score_threshold:
                reward -= cfg.high_score_continue_penalty
            if teacher_stop:
                # 【修复】低分 teacher_stop 不再立即终止，所以不再给予强终止惩罚
                # 如果 r >= high_score_threshold，说明真的是"无需修改"，给奖励
                if r_after >= cfg.high_score_threshold:
                    reward += cfg.teacher_stop_high_score_bonus
                # 否则：低分 teacher_stop 视为"本轮无变化"，走下面的 no_change 逻辑
                # 不再额外惩罚，让模型有机会在下一轮换个策略继续修

        # Efficiency penalty for broad-but-ineffective edits.
        if action_type != "STOP" and len(chain_indices) > cfg.large_edit_threshold and reward_delta <= 0:
            extra_count = max(0, len(chain_indices) - cfg.large_edit_threshold)
            reward -= min(cfg.large_edit_penalty * float(extra_count), 0.3)

        # Terminal shaping: keep terminal bonuses tied to the terminal state itself.
        # This avoids hidden dependence on best/initial score trackers.
        if done:
            terminal_score = max(r_before, r_after) if action_type == "STOP" else r_after
            reward += cfg.final_score_scale * terminal_score
            if stop_reason == "auto_threshold":
                reward += 0.1
            if stop_reason in {"stagnation", "teacher_stop_stagnation"}:
                reward -= 0.02
            if stop_reason == "max_rounds" and reward_delta <= cfg.no_progress_eps:
                reward -= 0.03
            if (
                stop_reason in {"stagnation", "max_rounds", "teacher_stop_stagnation"}
                and (terminal_score + cfg.no_progress_eps < cfg.high_score_threshold)
            ):
                reward -= cfg.ineffective_termination_penalty

        if used_rollback_delta:
            reward = min(reward, 0.0)

        return float(reward)

    def _normalize_indices(self, raw: Any) -> List[int]:
        if not isinstance(raw, list):
            return []
        out: List[int] = []
        seen = set()
        for x in raw:
            try:
                idx = int(x)
                if idx >= 0 and idx not in seen:
                    out.append(idx)
                    seen.add(idx)
            except Exception:
                continue
        return out

    def _record_step(self, info: Dict[str, Any], reward: float) -> None:
        self.trajectory["steps"].append(copy.deepcopy(info))
        self.trajectory["rewards"].append(float(reward))

    def get_trajectory(self) -> Dict[str, Any]:
        out = copy.deepcopy(self.trajectory)
        out["final_kc"] = copy.deepcopy(self.current_kc)
        out["final_r_rule"] = self.current_r_rule
        out["best_kc"] = copy.deepcopy(self.best_kc)
        out["best_r_rule"] = self.best_r_rule
        out["total_improvement"] = self.best_r_rule - self.initial_r_rule
        out["num_rounds"] = self.round_num
        out["total_reward"] = float(sum(self.trajectory.get("rewards", [])))
        out["stagnant_rounds"] = self.stagnant_rounds
        return out
