#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step9 PPO training entry."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from peft import LoraConfig, get_peft_model, TaskType
except Exception:
    LoraConfig = None
    get_peft_model = None
    TaskType = None

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from step9.deepseek_corrector import DeepSeekCorrector
from step9.environment import CorrectionEnvironment, RewardConfig
from step9.ppo_trainer import PPOConfig, PPOTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_device_map_arg(raw_value: str | None) -> Any:
    """Parse CLI device_map string to a HF-compatible object."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None

    low = text.lower()
    if low in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return low
    if low in {"cpu", "disk"}:
        return {"": low}
    if low.isdigit():
        return {"": int(low)}

    m = re.fullmatch(r"cuda:(\d+)", low)
    if m:
        return {"": int(m.group(1))}

    if text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    raise ValueError(
        f"Unsupported device_map value: {raw_value!r}. "
        'Use one of: auto/balanced/balanced_low_0/sequential/cuda:N/N/cpu/or a JSON dict.'
    )


def _parse_cuda_device_id(value: Any) -> int | None:
    """Return CUDA device index for a single-device spec, else None."""
    if isinstance(value, int):
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        text = value.strip().lower()
        if text.isdigit():
            return int(text)
        m = re.fullmatch(r"cuda:(\d+)", text)
        if m:
            return int(m.group(1))
        return None
    if isinstance(value, torch.device):
        if value.type == "cuda" and value.index is not None and int(value.index) >= 0:
            return int(value.index)
    return None


def infer_single_cuda_device_from_map(device_map: Any) -> int | None:
    """
    Infer single CUDA device id from a parsed device_map.
    Returns None if device_map is auto/mixed/non-CUDA/ambiguous.
    """
    if isinstance(device_map, dict):
        if not device_map:
            return None
        ids = set()
        for target in device_map.values():
            dev_id = _parse_cuda_device_id(target)
            if dev_id is None:
                return None
            ids.add(dev_id)
        if len(ids) == 1:
            return int(next(iter(ids)))
        return None
    return _parse_cuda_device_id(device_map)


def resolve_prompt_root(raw_value: str | None = None) -> Path:
    if raw_value:
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path
    preferred = (PROJECT_ROOT / "step9" / "prompt_packs" / "train_v1").resolve()
    if not preferred.exists():
        raise FileNotFoundError(f"train_v1 prompt pack not found at {preferred}")
    return preferred


def load_training_data(
    data_dir: Path,
    num_samples: int | None = None,
    sample_seed: int = 42,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    def _sort_key(p: Path):
        try:
            return (0, int(p.stem))
        except Exception:
            return (1, p.stem)

    files = sorted(data_dir.glob("*.json"), key=_sort_key)
    if num_samples is not None:
        if num_samples < len(files):
            rng = random.Random(sample_seed)
            files = rng.sample(files, k=num_samples)
            files = sorted(files, key=_sort_key)
        else:
            files = files[:num_samples]

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            table_content = str(data.get("table_content", "")).strip()
            km = data.get("knowledge_mining", [])
            if km is None:
                km = []
            if table_content and isinstance(km, list):
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


def split_train_eval_samples(
    samples: List[Dict[str, Any]],
    eval_ratio: float = 0.1,
    eval_max_samples: int | None = 128,
    split_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = len(samples)
    if total < 2 or eval_ratio <= 0:
        return list(samples), []

    eval_count = int(round(float(total) * float(eval_ratio)))
    if eval_count <= 0:
        eval_count = 1
    eval_count = min(eval_count, total - 1)
    if eval_max_samples is not None and int(eval_max_samples) > 0:
        eval_count = min(eval_count, int(eval_max_samples))
    if eval_count <= 0:
        return list(samples), []

    rng = random.Random(split_seed)
    eval_indices = set(rng.sample(range(total), k=eval_count))
    train_samples = [sample for idx, sample in enumerate(samples) if idx not in eval_indices]
    eval_samples = [sample for idx, sample in enumerate(samples) if idx in eval_indices]
    return train_samples, eval_samples


def main() -> None:
    # Keep realtime logs and force UTF-8 to avoid Windows console mojibake.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
    
    parser = argparse.ArgumentParser(description="Step9 PPO training")

    parser.add_argument("--model_path", type=str, required=True, help="Step8 policy model path")
    parser.add_argument("--tokenizer_path", type=str, default="", help="Optional tokenizer/chat_template path")
    parser.add_argument("--data_dir", type=str, default="step6/RL")
    parser.add_argument("--output_dir", type=str, default="step9/output/ppo_model")
    parser.add_argument("--prompt_root", type=str, default="", help="Prompt pack root; defaults to step9/prompt_packs/train_v1 when not specified")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--eval_max_samples", type=int, default=128)
    parser.add_argument("--eval_every_iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--episodes_per_iteration", type=int, default=16, help="0 means one full pass over the train split after held-out split")
    parser.add_argument("--update_every_episodes", type=int, default=20, help="Collect N episodes then do one PPO update; 0 means update once per iteration")
    parser.add_argument("--max_rounds", type=int, default=6)
    parser.add_argument("--stop_threshold", type=float, default=0.9,
                        help="Stop episode when R-rule >= threshold (与 step8.1 评估对齐，默认 0.9)")
    parser.add_argument("--resume_from", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="Directory to save PPO checkpoints")
    parser.add_argument("--save_every_iterations", type=int, default=1, help="Save checkpoint every N iterations")
    parser.add_argument("--auto_resume_latest", type=int, default=1, help="If resume_from is empty, auto load latest checkpoint from checkpoint_dir")
    parser.add_argument("--record_episode_records", type=int, default=1, help="Whether to persist per-episode trajectory JSON files")
    parser.add_argument("--write_iteration_json", type=int, default=1, help="Whether to persist per-iteration JSON summaries")

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps for larger effective batch size")
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.1)
    parser.add_argument("--value_loss_coef", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--ref_kl_coef", type=float, default=0.05, help="KL penalty relative to Step8 baseline model")
    parser.add_argument("--ref_kl_sample_rate", type=float, default=0.25, help="Sample rate for ref_kl computation to reduce overhead")
    parser.add_argument("--ref_kl_update_interval", type=int, default=4, help="Compute ref_kl once every N mini-batches")
    parser.add_argument("--target_kl", type=float, default=0.05)
    parser.add_argument("--early_stop_kl", type=float, default=0.2, help="Early stop KL threshold")
    parser.add_argument("--ref_kl_max_tokens", type=int, default=1024, help="Max tokens for ref_kl computation (to avoid OOM)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Learning rate warmup steps")
    parser.add_argument("--lr_scheduler", type=str, default="constant", choices=["linear", "cosine", "constant"])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_prompt_tokens", type=int, default=12000,
                        help="Max prompt tokens for strategy selection (与 step8.1 对齐，默认 12000)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--action_max_new_tokens", type=int, default=512)
    parser.add_argument("--initial_max_prompt_tokens", type=int, default=12000,
                        help="Max prompt tokens for initial KC generation (与 step8.1 对齐，默认 12000)")
    parser.add_argument("--initial_max_new_tokens", type=int, default=4096)
    parser.add_argument("--initial_action_max_eval_tokens", type=int, default=4096,
                        help="Max tokens for initial KC PPO evaluation (should match initial_max_new_tokens)")
    parser.add_argument("--initial_r_reward_scale", type=float, default=0.40)
    parser.add_argument(
        "--initial_generation_loss_weight",
        type=float,
        default=1.35,
        help="Additional PPO loss weight multiplier for the initial-KC generation step.",
    )
    parser.add_argument(
        "--initial_generation_temperature",
        type=float,
        default=0.10,
        help="Sampling temperature for PPO initial-KC rollout generation.",
    )
    parser.add_argument(
        "--initial_generation_top_p",
        type=float,
        default=0.95,
        help="Top-p for PPO initial-KC rollout generation.",
    )
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--optimize_initial_kc", type=int, default=1, help="Whether to include initial-KC generation in PPO optimization")
    parser.add_argument("--focus_window_size", type=int, default=7, help="Deprecated compatibility knob; full state is now exposed to the policy")
    parser.add_argument("--table_summary_char_budget", type=int, default=1800, help="Deprecated compatibility knob; raw table_content is used")
    parser.add_argument("--kc_summary_char_budget", type=int, default=3600, help="Deprecated compatibility knob; full current_kc is used")

    # Reward/safety knobs
    parser.add_argument("--delta_pos_scale", type=float, default=6.0)
    parser.add_argument("--delta_neg_scale", type=float, default=8.0)
    parser.add_argument("--rollback_threshold", type=float, default=0.3)
    parser.add_argument("--rollback_extra_penalty", type=float, default=1.5)
    parser.add_argument("--high_score_threshold", type=float, default=0.9,
                        help="High score threshold for STOP bonus (与 step8.1 评估对齐，默认 0.9)")
    parser.add_argument("--high_score_stop_bonus", type=float, default=1.0)
    parser.add_argument("--premature_stop_penalty", type=float, default=1.0)
    parser.add_argument("--high_score_continue_penalty", type=float, default=0.1)
    parser.add_argument("--teacher_stop_high_score_bonus", type=float, default=0.15)
    parser.add_argument("--large_edit_threshold", type=int, default=3)
    parser.add_argument("--large_edit_penalty", type=float, default=0.05)
    parser.add_argument("--step_penalty", type=float, default=0.005)
    parser.add_argument("--no_progress_penalty", type=float, default=0.02)
    parser.add_argument("--norm_equivalent_noop_penalty", type=float, default=0.08)
    parser.add_argument("--positive_correction_bonus", type=float, default=0.05)
    parser.add_argument("--final_score_scale", type=float, default=0.20)
    parser.add_argument("--no_progress_eps", type=float, default=1e-3)
    parser.add_argument("--max_stagnant_rounds", type=int, default=3)
    parser.add_argument("--max_consecutive_teacher_stops", type=int, default=2)
    parser.add_argument("--ineffective_termination_penalty", type=float, default=0.12)
    parser.add_argument(
        "--teacher_failed_adv_weight",
        type=float,
        default=0.5,
        help="PPO optimization weight for teacher_failed non-terminal steps (0~1).",
    )
    parser.add_argument(
        "--prompt_overflow_mode",
        type=str,
        default="truncate",
        choices=["truncate", "error"],
        help="How to handle overlong prompts before tokenization into policy/value/ref paths.",
    )

    parser.add_argument(
        "--deepseek_api_url",
        type=str,
        default="https://api.deepseek.com/v1/chat/completions",
    )
    parser.add_argument("--deepseek_api_key", type=str, default="")
    parser.add_argument("--deepseek_model", type=str, default="DeepSeek-V3.2")
    parser.add_argument("--teacher_max_retries", type=int, default=3)
    parser.add_argument("--teacher_timeout", type=int, default=180)
    parser.add_argument("--teacher_backoff_base", type=float, default=2.0)

    parser.add_argument("--enable_ref_model", type=int, default=1, help="Enable frozen reference model for ref_kl")
    parser.add_argument("--ref_model_path", type=str, default="", help="Reference model path; default=model_path")
    parser.add_argument("--ref_model_device_map", type=str, default="auto", help='e.g. "auto" or "cuda:0"')
    parser.add_argument(
        "--allow_shared_ref_device",
        type=int,
        default=0,
        help="Allow policy model and ref_model to share the same CUDA device for single-GPU runs.",
    )

    parser.add_argument("--device_map", type=str, default="auto", help='e.g. "auto" or "cuda:0"')
    args = parser.parse_args()

    if LoraConfig is None or get_peft_model is None or TaskType is None:
        raise RuntimeError(
            "Missing dependency `peft`. Please install it in your training environment."
        )

    if not args.deepseek_api_key:
        args.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not args.deepseek_api_key:
        raise RuntimeError("DeepSeek API key is empty. Set --deepseek_api_key or DEEPSEEK_API_KEY.")

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 注意：step8.1 评估没有 high_score_threshold 概念，只有 stop_threshold
    # 训练时设为与 stop_threshold 相同，让 STOP 奖励在达到阈值时触发
    if args.high_score_threshold >= args.stop_threshold:
        # 与 step8.1 对齐：不自动调整，直接使用 stop_threshold 的值
        print(
            f"[INFO] high_score_threshold ({args.high_score_threshold:.3f}) >= stop_threshold ({args.stop_threshold:.3f}); "
            f"using high_score_threshold = stop_threshold = {args.stop_threshold:.3f} (与 step8.1 评估对齐)",
            flush=True,
        )
        args.high_score_threshold = args.stop_threshold

    data_dir = Path(args.data_dir)
    if not data_dir.exists() and not data_dir.is_absolute():
        data_dir = (PROJECT_ROOT / data_dir).resolve()
    prompt_root = resolve_prompt_root(args.prompt_root)
    all_samples = load_training_data(data_dir, args.num_samples, sample_seed=args.seed)
    if not all_samples:
        raise RuntimeError(f"No training samples loaded from {data_dir}")
    train_samples, eval_samples = split_train_eval_samples(
        all_samples,
        eval_ratio=args.eval_ratio,
        eval_max_samples=args.eval_max_samples,
        split_seed=args.seed,
    )
    if not train_samples:
        raise RuntimeError("Train split is empty after held-out partitioning.")
    print(
        f"[DATA] Loaded {len(all_samples)} samples from {data_dir}: "
        f"train={len(train_samples)}, eval={len(eval_samples)}",
        flush=True,
    )
    if int(args.episodes_per_iteration) <= 0:
        args.episodes_per_iteration = len(train_samples)
        print(
            f"[DATA] episodes_per_iteration auto-set to train split size: {args.episodes_per_iteration}",
            flush=True,
        )

    tokenizer_fallback = Path(args.tokenizer_path) if args.tokenizer_path else None
    if tokenizer_fallback is not None and not tokenizer_fallback.is_absolute():
        tokenizer_fallback = (PROJECT_ROOT / tokenizer_fallback).resolve()
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        print(f"Tokenizer loaded from model_path: {args.model_path}", flush=True)
    except Exception as exc:
        if tokenizer_fallback is None:
            raise RuntimeError(
                f"Failed to load tokenizer from {args.model_path}: {exc}. "
                "Please provide --tokenizer_path."
            ) from exc
        print(
            f"Failed to load tokenizer from {args.model_path}, trying {tokenizer_fallback} ...",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_fallback), trust_remote_code=True)

    # 鍔犺浇chat_template
    chat_template_path = Path(args.model_path) / "chat_template.jinja"
    if not chat_template_path.exists() and tokenizer_fallback is not None:
        chat_template_path = tokenizer_fallback / "chat_template.jinja"
    if chat_template_path.exists():
        tokenizer.chat_template = chat_template_path.read_text(encoding='utf-8')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    parsed_policy_device_map = parse_device_map_arg(args.device_map)
    parsed_ref_device_map = None
    if int(args.enable_ref_model) == 1 and args.ref_kl_coef > 0:
        parsed_ref_device_map = parse_device_map_arg(args.ref_model_device_map)
        policy_cuda_id = infer_single_cuda_device_from_map(parsed_policy_device_map)
        ref_cuda_id = infer_single_cuda_device_from_map(parsed_ref_device_map)

        allow_shared_ref_device = bool(int(args.allow_shared_ref_device))
        # By default we keep policy/ref on different CUDA devices.
        # Single-GPU 48GB runs can explicitly opt into shared placement.
        if (
            policy_cuda_id is not None
            and ref_cuda_id is not None
            and policy_cuda_id == ref_cuda_id
            and not allow_shared_ref_device
        ):
            raise RuntimeError(
                "Policy model and reference model resolve to the same CUDA device "
                f"(cuda:{policy_cuda_id}). "
                f"device_map={args.device_map!r}, ref_model_device_map={args.ref_model_device_map!r}. "
                "Please assign different GPUs or pass --allow_shared_ref_device 1 for a single-GPU run."
            )

        # Log device assignment
        if ref_cuda_id is not None:
            print(f"[INFO] ref_model will be loaded on cuda:{ref_cuda_id}", flush=True)
            if policy_cuda_id == ref_cuda_id and allow_shared_ref_device:
                print(
                    f"[WARN] policy model and ref_model will share cuda:{ref_cuda_id}; "
                    "this increases VRAM pressure but is allowed by config.",
                    flush=True,
                )
        else:
            # ref_model is on CPU or other device
            print(f"[INFO] ref_model will be loaded on device: {args.ref_model_device_map}", flush=True)

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if parsed_policy_device_map is not None:
        model_kwargs["device_map"] = parsed_policy_device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    # Apply LoRA to reduce trainable parameter count.
    print("Applying LoRA to model...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Enable gradient checkpointing for memory saving.
    model.gradient_checkpointing_enable()

    ref_model = None
    if int(args.enable_ref_model) == 1 and args.ref_kl_coef > 0:
        ref_model_path = args.ref_model_path if args.ref_model_path else args.model_path
        ref_model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        # Use 4-bit quantization for ref_model to save memory.
        ref_model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if parsed_ref_device_map is not None:
            ref_model_kwargs["device_map"] = parsed_ref_device_map
        try:
            print(f"Loading frozen ref_model from: {ref_model_path}", flush=True)
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, **ref_model_kwargs)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
            print("Frozen ref_model loaded.", flush=True)
        except Exception as exc:
            print(f"[WARN] Failed to load ref_model: {exc}. Continue without ref_kl.", flush=True)
            ref_model = None
    else:
        print("ref_model disabled by config or ref_kl_coef <= 0.", flush=True)

    corrector = DeepSeekCorrector(
        teacher_prompts_dir=prompt_root / "prompts" / "teacher",
        max_retries=args.teacher_max_retries,
        timeout_seconds=args.teacher_timeout,
        backoff_base_seconds=args.teacher_backoff_base,
    )
    corrector.api_url = args.deepseek_api_url
    corrector.api_key = args.deepseek_api_key
    corrector.model_name = args.deepseek_model

    reward_cfg = RewardConfig(
        delta_pos_scale=args.delta_pos_scale,
        delta_neg_scale=args.delta_neg_scale,
        rollback_threshold=args.rollback_threshold,
        rollback_extra_penalty=args.rollback_extra_penalty,
        high_score_threshold=args.high_score_threshold,
        high_score_stop_bonus=args.high_score_stop_bonus,
        premature_stop_penalty=args.premature_stop_penalty,
        high_score_continue_penalty=args.high_score_continue_penalty,
        teacher_stop_high_score_bonus=args.teacher_stop_high_score_bonus,
        large_edit_threshold=args.large_edit_threshold,
        large_edit_penalty=args.large_edit_penalty,
        step_penalty=args.step_penalty,
        no_progress_penalty=args.no_progress_penalty,
        norm_equivalent_noop_penalty=args.norm_equivalent_noop_penalty,
        positive_correction_bonus=args.positive_correction_bonus,
        final_score_scale=args.final_score_scale,
        no_progress_eps=args.no_progress_eps,
        max_stagnant_rounds=args.max_stagnant_rounds,
        max_consecutive_teacher_stops=args.max_consecutive_teacher_stops,
        ineffective_termination_penalty=args.ineffective_termination_penalty,
    )
    env = CorrectionEnvironment(
        corrector=corrector,
        max_rounds=args.max_rounds,
        stop_threshold=args.stop_threshold,
        reward_config=reward_cfg,
        focus_window_size=args.focus_window_size,
        table_summary_char_budget=args.table_summary_char_budget,
        kc_summary_char_budget=args.kc_summary_char_budget,
    )

    ppo_cfg = PPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        ref_kl_coef=args.ref_kl_coef,
        ref_kl_sample_rate=args.ref_kl_sample_rate,
        ref_kl_update_interval=args.ref_kl_update_interval,
        target_kl=args.target_kl,
        early_stop_kl=args.early_stop_kl,
        ref_kl_max_tokens=args.ref_kl_max_tokens,
        warmup_steps=args.warmup_steps,
        lr_scheduler=args.lr_scheduler,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        action_max_new_tokens=args.action_max_new_tokens,
        initial_max_prompt_tokens=args.initial_max_prompt_tokens,
        initial_max_new_tokens=args.initial_max_new_tokens,
        initial_action_max_eval_tokens=args.initial_action_max_eval_tokens,
        initial_r_reward_scale=args.initial_r_reward_scale,
        initial_generation_loss_weight=args.initial_generation_loss_weight,
        initial_generation_temperature=args.initial_generation_temperature,
        initial_generation_top_p=args.initial_generation_top_p,
        temperature=args.temperature,
        top_p=args.top_p,
        teacher_failed_adv_weight=args.teacher_failed_adv_weight,
        high_score_threshold=args.high_score_threshold,
        prompt_overflow_mode=args.prompt_overflow_mode,
        optimize_initial_kc=bool(int(args.optimize_initial_kc)),
    )
    trainer = PPOTrainer(
        model=model,
        tokenizer=tokenizer,
        env=env,
        config=ppo_cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
        ref_model=ref_model,
        prompt_root=prompt_root,
    )

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (out_dir / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = 0
    initial_history: List[Dict[str, float]] = []
    resume_path: Path | None = Path(args.resume_from) if args.resume_from else None
    if resume_path is None and int(args.auto_resume_latest) == 1:
        latest_ckpts = sorted(checkpoint_dir.glob("ppo_ckpt_iter_*.pt"))
        if latest_ckpts:
            resume_path = latest_ckpts[-1]
            print(f"Auto-resume found latest checkpoint: {resume_path}", flush=True)

    if resume_path is not None:
        resume_state = trainer.load_checkpoint(resume_path)
        start_iteration = int(resume_state.get("iteration", 0))
        initial_history = list(resume_state.get("history", []))
        print(f"Resuming from checkpoint: {resume_path}, start_iteration={start_iteration}", flush=True)

    def _interrupt_handler(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)

    # Generate timestamp for this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"[main] run_timestamp: {run_timestamp}", flush=True)

    history = trainer.train(
        train_samples=train_samples,
        eval_samples=eval_samples,
        num_iterations=args.num_iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        update_every_episodes=args.update_every_episodes,
        eval_every_iterations=args.eval_every_iterations,
        start_iteration=start_iteration,
        initial_history=initial_history,
        checkpoint_dir=checkpoint_dir,
        save_every_iterations=args.save_every_iterations,
        run_timestamp=run_timestamp,
        record_episode_records=bool(int(args.record_episode_records)),
        write_iteration_json=bool(int(args.write_iteration_json)),
    )

    final_dir = out_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    torch.save(trainer.value_head.state_dict(), final_dir / "value_head.pt")

    run_meta = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "data_dir": str(data_dir),
        "prompt_root": str(prompt_root),
        "num_total_samples": len(all_samples),
        "num_train_samples": len(train_samples),
        "num_eval_samples": len(eval_samples),
        "eval_ratio": args.eval_ratio,
        "eval_max_samples": args.eval_max_samples,
        "eval_every_iterations": args.eval_every_iterations,
        "num_iterations": args.num_iterations,
        "episodes_per_iteration": args.episodes_per_iteration,
        "update_every_episodes": args.update_every_episodes,
        "record_episode_records": bool(int(args.record_episode_records)),
        "write_iteration_json": bool(int(args.write_iteration_json)),
        "max_rounds": args.max_rounds,
        "stop_threshold": args.stop_threshold,
        "ppo_config": vars(ppo_cfg),
        "reward_config": vars(reward_cfg),
        "history": history,
    }
    (out_dir / "training_summary.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Step9 PPO training finished.", flush=True)
    print(f"Model saved to: {final_dir}", flush=True)
    print(f"Summary saved to: {out_dir / 'training_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
