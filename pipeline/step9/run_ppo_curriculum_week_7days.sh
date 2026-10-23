#!/bin/bash

# ============================================================
# Step9 PPO Curriculum - 7天完成版本
# ============================================================
# 基于当前 0.79 chunks/hour 的训练速度
# 目标：在 7 天内完成训练
# ============================================================

set -euo pipefail

echo "=========================================="
echo "Step9 PPO Curriculum (7-Day Budget)"
echo "=========================================="

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    export PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    export PYTHON_BIN="python"
  else
    echo "Error: neither python3 nor python was found in PATH."
    exit 1
  fi
fi

# 训练入口前预检查，避免后台启动后才发现基础环境缺失
export MODEL_PATH="${MODEL_PATH:-data/checkpoints/step8_merged}"
export EASY_DATA_DIR="${EASY_DATA_DIR:-$WORKDIR/step6/RL_easy}"
export MEDIUM_DATA_DIR="${MEDIUM_DATA_DIR:-$WORKDIR/step6/RL_medium}"
export HARD_DATA_DIR="${HARD_DATA_DIR:-$WORKDIR/step6/RL_hard}"
export PROMPT_ROOT="${PROMPT_ROOT:-$WORKDIR/step9/prompt_packs/train_v1}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: MODEL_PATH not found: $MODEL_PATH"
  exit 1
fi
if [ ! -d "$EASY_DATA_DIR" ]; then
  echo "Error: EASY_DATA_DIR not found: $EASY_DATA_DIR"
  exit 1
fi
if [ ! -d "$MEDIUM_DATA_DIR" ]; then
  echo "Error: MEDIUM_DATA_DIR not found: $MEDIUM_DATA_DIR"
  exit 1
fi
if [ ! -d "$HARD_DATA_DIR" ]; then
  echo "Error: HARD_DATA_DIR not found: $HARD_DATA_DIR"
  exit 1
fi
if [ ! -d "$PROMPT_ROOT" ]; then
  echo "Error: PROMPT_ROOT not found: $PROMPT_ROOT"
  exit 1
fi

# 输出路径
export OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/step9/output/ppo_curriculum_week_7days}"
export CURRICULUM_LOG_DIR="${CURRICULUM_LOG_DIR:-$OUTPUT_DIR/logs}"
export CURRICULUM_LOG_FILE="${CURRICULUM_LOG_FILE:-$CURRICULUM_LOG_DIR/ppo_curriculum_week_7days.log}"
export LATEST_CURRICULUM_RUN_FILE="${LATEST_CURRICULUM_RUN_FILE:-$OUTPUT_DIR/latest_curriculum_run.txt}"
export STAGE_LOG_DIR="${STAGE_LOG_DIR:-$CURRICULUM_LOG_DIR}"

# GPU 配置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
export REF_MODEL_DEVICE_MAP="${REF_MODEL_DEVICE_MAP:-cuda:1}"
export ALLOW_SHARED_REF_DEVICE="${ALLOW_SHARED_REF_DEVICE:-0}"
export ENABLE_REF_MODEL="${ENABLE_REF_MODEL:-1}"
export DEEPSEEK_API_URL="${LLM_API_URL:-}"
export DEEPSEEK_API_URL_FALLBACKS="${DEEPSEEK_API_URL_FALLBACKS:-}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY}"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-DeepSeek-V3.2}"

# 后台运行模式
# CURRICULUM_BACKGROUND: 仅课程总控脚本在后台运行
# RUN_IN_FOREGROUND: 各 stage 在课程主进程内前台顺序执行，并统一追加到主日志
# 为保证“单主日志 + 后台启动 + 顺序跑完整课程”，这里强制固定关键行为开关。
export CURRICULUM_BACKGROUND=1
export RUN_IN_FOREGROUND=1
export RECORD_EPISODE_RECORDS=0
export WRITE_ITERATION_JSON=0
export REBUILD_STAGE_DIRS_FROM_BUCKETS=1
export AUTO_RESUME_LATEST=1

# 评测关闭（训练完成后单独评测）
export EVAL_RATIO=0
export EVAL_MAX_SAMPLES=0
export EVAL_EVERY_ITERATIONS=0

# ============================================================
# 【关键参数】为了 7 天内跑完，同时保留后期 refinement：
#
# 1. 压缩前期 Easy 预算，把更多 budget 留给 Stage3
# 2. Stage3 增加到 2 个 iteration，避免所有 stage 都只扫一遍
# 3. 恢复 PPO 友好的生成长度，避免超长生成拖慢吞吐
# ============================================================

# ---------- 控制总耗时的关键参数 ----------
export MAX_ROUNDS="${MAX_ROUNDS:-4}"
export MAX_STAGNANT_ROUNDS="${MAX_STAGNANT_ROUNDS:-3}"
export MAX_CONSECUTIVE_TEACHER_STOPS="${MAX_CONSECUTIVE_TEACHER_STOPS:-2}"
export UPDATE_EVERY_EPISODES="${UPDATE_EVERY_EPISODES:-16}"
export SAVE_EVERY_ITERATIONS="${SAVE_EVERY_ITERATIONS:-1}"
export KEEP_ITER_CHECKPOINTS="${KEEP_ITER_CHECKPOINTS:-4}"

# ---------- PPO 参数 ----------
export REF_KL_COEF="${REF_KL_COEF:-0.05}"
export REF_KL_SAMPLE_RATE="${REF_KL_SAMPLE_RATE:-0.10}"
export REF_KL_UPDATE_INTERVAL="${REF_KL_UPDATE_INTERVAL:-8}"
export REF_KL_MAX_TOKENS="${REF_KL_MAX_TOKENS:-512}"

# ---------- Generation parameters (reduced for OOM fix) ----------
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8000}"
export INITIAL_MAX_PROMPT_TOKENS="${INITIAL_MAX_PROMPT_TOKENS:-8000}"
export INITIAL_MAX_NEW_TOKENS="${INITIAL_MAX_NEW_TOKENS:-4096}"
export INITIAL_ACTION_MAX_EVAL_TOKENS="${INITIAL_ACTION_MAX_EVAL_TOKENS:-$INITIAL_MAX_NEW_TOKENS}"
export ACTION_MAX_NEW_TOKENS="${ACTION_MAX_NEW_TOKENS:-512}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export INITIAL_R_REWARD_SCALE="${INITIAL_R_REWARD_SCALE:-0.40}"
export INITIAL_GENERATION_LOSS_WEIGHT="${INITIAL_GENERATION_LOSS_WEIGHT:-1.35}"
export INITIAL_GENERATION_TEMPERATURE="${INITIAL_GENERATION_TEMPERATURE:-0.10}"
export INITIAL_GENERATION_TOP_P="${INITIAL_GENERATION_TOP_P:-0.95}"
export TEMPERATURE="${TEMPERATURE:-0.15}"
export TOP_P="${TOP_P:-0.95}"
export NORM_EQUIVALENT_NOOP_PENALTY="${NORM_EQUIVALENT_NOOP_PENALTY:-0.08}"

# ============================================================
# 【Iterations 配置】前轻后重
# ============================================================
export STAGE1_TOTAL_ITERS="${STAGE1_TOTAL_ITERS:-1}"
export STAGE2_TOTAL_ITERS="${STAGE2_TOTAL_ITERS:-1}"
export STAGE3_TOTAL_ITERS="${STAGE3_TOTAL_ITERS:-2}"
export STAGE4_TOTAL_ITERS="${STAGE4_TOTAL_ITERS:-1}"

# ============================================================
# 【样本量配置】把预算从 Easy 前期转给中后期难样本
#
# 计算依据：
# - 128 total chunks
# - 当前速度：~0.79 chunks/hour
# - 7天 = 168小时 → 保守可处理 ~132 chunks
#
# 分配策略：
# - Stage 1 (Easy): 只保留必要 bootstrap
# - Stage 2 (Easy+Medium): 过渡到中等难度
# - Stage 3 (Easy+Medium+Hard): 主训练阶段，跑两遍但缩小 easy/medium
# - Stage 4 (Hard refine): 把最终预算集中到 hard，恢复 late-stage hard coverage
# ============================================================

# ---------- Stage 1: Easy Only ----------
# 256 samples → 16 chunks (1 iter)
export STAGE1_EASY_COUNT="${STAGE1_EASY_COUNT:-256}"
export STAGE1_EASY_OFFSET="${STAGE1_EASY_OFFSET:-0}"

# ---------- Stage 2: Easy + Medium ----------
# (64 + 192) samples → 16 chunks (1 iter)
export STAGE2_EASY_COUNT="${STAGE2_EASY_COUNT:-64}"
export STAGE2_EASY_OFFSET="${STAGE2_EASY_OFFSET:-256}"
export STAGE2_MEDIUM_COUNT="${STAGE2_MEDIUM_COUNT:-192}"
export STAGE2_MEDIUM_OFFSET="${STAGE2_MEDIUM_OFFSET:-0}"

# ---------- Stage 3: Easy + Medium + Hard ----------
# (32 + 96 + 256) samples → 24 chunks per iter, 48 chunks total
export STAGE3_EASY_COUNT="${STAGE3_EASY_COUNT:-32}"
export STAGE3_EASY_OFFSET="${STAGE3_EASY_OFFSET:-320}"
export STAGE3_MEDIUM_COUNT="${STAGE3_MEDIUM_COUNT:-96}"
export STAGE3_MEDIUM_OFFSET="${STAGE3_MEDIUM_OFFSET:-192}"
export STAGE3_HARD_COUNT="${STAGE3_HARD_COUNT:-256}"
export STAGE3_HARD_OFFSET="${STAGE3_HARD_OFFSET:-0}"

# ---------- Stage 4: Hard refine ----------
# 768 hard samples → 48 chunks (1 iter)
# 与旧版“Stage4 大量 hard refine”的目标保持一致；
# 在 128 chunks 总预算下，Stage3 + Stage4 合计覆盖 1024 个 hard 样本。
export STAGE4_MEDIUM_COUNT="${STAGE4_MEDIUM_COUNT:-0}"
export STAGE4_MEDIUM_OFFSET="${STAGE4_MEDIUM_OFFSET:-0}"
export STAGE4_HARD_COUNT="${STAGE4_HARD_COUNT:-768}"
export STAGE4_HARD_OFFSET="${STAGE4_HARD_OFFSET:-256}"

# ============================================================
# 打印配置摘要
# ============================================================
echo ""
echo "=== 7-Day Budget Configuration ==="
echo ""
echo "Output: $OUTPUT_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Easy bucket: $EASY_DATA_DIR"
echo "Medium bucket: $MEDIUM_DATA_DIR"
echo "Hard bucket: $HARD_DATA_DIR"
echo "Prompt root: $PROMPT_ROOT"
echo ""
echo "--- Stage Samples (total chunks) ---"

# 计算各 Stage 的 chunks
s1_chunks=$((STAGE1_EASY_COUNT / 16 * STAGE1_TOTAL_ITERS))
s2_chunks=$(((STAGE2_EASY_COUNT + STAGE2_MEDIUM_COUNT) / 16 * STAGE2_TOTAL_ITERS))
s3_chunks=$(((STAGE3_EASY_COUNT + STAGE3_MEDIUM_COUNT + STAGE3_HARD_COUNT) / 16 * STAGE3_TOTAL_ITERS))
s4_chunks=$(((STAGE4_MEDIUM_COUNT + STAGE4_HARD_COUNT) / 16 * STAGE4_TOTAL_ITERS))
total_chunks=$((s1_chunks + s2_chunks + s3_chunks + s4_chunks))

echo "  Stage1 (Easy):              ${STAGE1_EASY_COUNT} samples → ${s1_chunks} chunks"
echo "  Stage2 (Easy+Medium):        $((STAGE2_EASY_COUNT + STAGE2_MEDIUM_COUNT)) samples → ${s2_chunks} chunks"
echo "  Stage3 (Easy+Medium+Hard):  $((STAGE3_EASY_COUNT + STAGE3_MEDIUM_COUNT + STAGE3_HARD_COUNT)) samples → ${s3_chunks} chunks"
echo "  Stage4 (Hard refine):        $((STAGE4_MEDIUM_COUNT + STAGE4_HARD_COUNT)) samples → ${s4_chunks} chunks"
echo "  ─────────────────────────────────────────"
echo "  TOTAL:                       $((STAGE1_EASY_COUNT + STAGE2_EASY_COUNT + STAGE2_MEDIUM_COUNT + STAGE3_EASY_COUNT + STAGE3_MEDIUM_COUNT + STAGE3_HARD_COUNT + STAGE4_MEDIUM_COUNT + STAGE4_HARD_COUNT)) samples → ${total_chunks} chunks"
echo ""
echo "--- Time Estimate (at 0.79 chunks/hour) ---"
estimated_hours=$("$PYTHON_BIN" -c "total_chunks=$total_chunks; print(f'{total_chunks / 0.79:.1f}')")
estimated_days=$("$PYTHON_BIN" -c "total_chunks=$total_chunks; print(f'{total_chunks / 0.79 / 24:.1f}')")
echo "  Estimated time: ${estimated_hours} hours ≈ ${estimated_days} days"
echo ""
echo "--- Key Parameters ---"
echo "  MAX_ROUNDS=$MAX_ROUNDS"
echo "  UPDATE_EVERY_EPISODES=$UPDATE_EVERY_EPISODES"
echo "  SAVE_EVERY_ITERATIONS=$SAVE_EVERY_ITERATIONS"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  CURRICULUM_LOG_FILE=$CURRICULUM_LOG_FILE"
echo "  RECORD_EPISODE_RECORDS=$RECORD_EPISODE_RECORDS"
echo "  WRITE_ITERATION_JSON=$WRITE_ITERATION_JSON"
echo ""

# 重要：这里只后台化课程总控脚本；stage 必须保持前台顺序执行，
# 否则四个阶段会各自再 nohup 一次，无法保证课程顺序和单主日志。
export RUN_IN_FOREGROUND=1

exec bash "$WORKDIR/step9/run_ppo_curriculum.sh" "$@"
