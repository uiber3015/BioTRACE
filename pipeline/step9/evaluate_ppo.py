#!/usr/bin/env python3
import os
# -*- coding: utf-8 -*-
"""Step9 PPO policy evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from step9.deepseek_corrector import DeepSeekCorrector
from step9.environment import CorrectionEnvironment, RewardConfig
from step9.ppo_trainer import PPOConfig, PPOTrainer


def load_test_data(data_dir: Path, num_samples: int | None = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    def _sort_key(p: Path):
        try:
            return (0, int(p.stem))
        except Exception:
            return (1, p.stem)

    files = sorted(data_dir.glob("*.json"), key=_sort_key)
    if num_samples is not None:
        files = files[:num_samples]

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            table_content = str(data.get("table_content", "")).strip()
            km = data.get("knowledge_mining", [])
            if table_content and isinstance(km, list) and km:
                samples.append(
                    {
                        "file": f.stem,
                        "table_content": table_content,
                        "knowledge_mining": km,
                    }
                )
        except Exception:
            continue
    return samples


def evaluate_policy(
    trainer: PPOTrainer,
    env: CorrectionEnvironment,
    test_samples: List[Dict[str, Any]],
    max_rounds: int,
    deterministic: bool,
) -> Dict[str, Any]:
    sample_results: List[Dict[str, Any]] = []
    for sample in test_samples:
        table_content = sample["table_content"]
        gt = {"knowledge_mining": sample["knowledge_mining"]}

        initial_kc = trainer.generate_initial_kc(table_content)
        state = env.reset(table_content, initial_kc, gt)
        initial_r = env.initial_r_rule

        done = False
        steps = 0
        rollbacks = 0
        action_log: List[Dict[str, Any]] = []
        while not done and steps < max_rounds:
            action, _meta = trainer.select_action(state, deterministic=deterministic)
            next_state, reward, done, info = env.step(action)
            if info.get("rolled_back"):
                rollbacks += 1
            action_log.append(
                {
                    "round": steps,
                    "action": action,
                    "reward": reward,
                    "info": info,
                }
            )
            state = next_state
            steps += 1

        final_r = env.current_r_rule
        sample_results.append(
            {
                "sample_id": sample["file"],
                "initial_r_rule": initial_r,
                "final_r_rule": final_r,
                "improvement": final_r - initial_r,
                "num_rounds": steps,
                "rollbacks": rollbacks,
                "stop_reason": action_log[-1]["info"].get("stop_reason") if action_log else None,
                "trajectory": action_log,
            }
        )

    if not sample_results:
        return {"samples": [], "summary": {}}

    initial_vals = [x["initial_r_rule"] for x in sample_results]
    final_vals = [x["final_r_rule"] for x in sample_results]
    improve_vals = [x["improvement"] for x in sample_results]
    rounds_vals = [x["num_rounds"] for x in sample_results]
    rollback_vals = [x["rollbacks"] for x in sample_results]
    summary = {
        "num_samples": len(sample_results),
        "avg_initial_r_rule": float(np.mean(initial_vals)),
        "avg_final_r_rule": float(np.mean(final_vals)),
        "avg_improvement": float(np.mean(improve_vals)),
        "avg_num_rounds": float(np.mean(rounds_vals)),
        "avg_rollbacks": float(np.mean(rollback_vals)),
        "stop_accuracy_at_0_9": float(np.mean([1.0 if x["final_r_rule"] >= 0.9 else 0.0 for x in sample_results])),
    }
    return {"samples": sample_results, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Step9 PPO policy")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, default="/home/weijie_yu/yjw/ECLO-RL/step6/Test")
    parser.add_argument("--output_dir", type=str, default="step9/output/evaluation")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--stop_threshold", type=float, default=0.9)
    parser.add_argument("--deterministic", action="store_true", help="Use greedy decoding for strategy")
    parser.add_argument("--ref_kl_coef", type=float, default=0.0, help="Ref KL coefficient (for consistency with training)")
    parser.add_argument(
        "--deepseek_api_url",
        type=str,
        default=os.environ.get("LLM_API_URL", ""),
    )
    parser.add_argument("--deepseek_api_key", type=str, required=True)
    parser.add_argument("--deepseek_model", type=str, default="DeepSeek-V3.2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    corrector = DeepSeekCorrector()
    corrector.api_url = args.deepseek_api_url
    corrector.api_key = args.deepseek_api_key
    corrector.model_name = args.deepseek_model

    env = CorrectionEnvironment(
        corrector=corrector,
        max_rounds=args.max_rounds,
        stop_threshold=args.stop_threshold,
        reward_config=RewardConfig(high_score_threshold=args.stop_threshold),
    )
    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        env=env,
        config=PPOConfig(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    value_head_path = Path(args.model_path) / "value_head.pt"
    if value_head_path.exists():
        state_dict = torch.load(value_head_path, map_location=next(model.parameters()).device)
        trainer.value_head.load_state_dict(state_dict, strict=False)

    test_samples = load_test_data(Path(args.test_data), args.num_samples)
    if not test_samples:
        raise RuntimeError(f"No test samples loaded from {args.test_data}")

    results = evaluate_policy(
        trainer=trainer,
        env=env,
        test_samples=test_samples,
        max_rounds=args.max_rounds,
        deterministic=args.deterministic,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"evaluation_{ts}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Evaluation finished.")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
