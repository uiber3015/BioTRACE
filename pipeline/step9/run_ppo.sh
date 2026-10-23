#!/bin/bash

set -euo pipefail

echo "=========================================="
echo "Step9 PPO Smoke Test"
echo "=========================================="

# Paths
WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="data/checkpoints/step8_merged"
DATA_DIR="$WORKDIR/step6/RL"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/step9/output/ppo_model}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
RESUME_FROM="${RESUME_FROM:-}"
AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-0}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: neither python3 nor python was found in PATH."
    exit 1
  fi
fi

# Training schedule (smoke test defaults)
NUM_SAMPLES="${NUM_SAMPLES:-60}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
EVAL_EVERY_ITERATIONS="${EVAL_EVERY_ITERATIONS:-1}"
NUM_ITERATIONS="${NUM_ITERATIONS:-12}"
EPISODES_PER_ITERATION="${EPISODES_PER_ITERATION:-0}"  # 0 => auto use train split size after held-out split
UPDATE_EVERY_EPISODES="${UPDATE_EVERY_EPISODES:-15}"  # Increased to reduce update frequency for more stable training
MAX_ROUNDS="${MAX_ROUNDS:-5}"
STOP_THRESHOLD="${STOP_THRESHOLD:-0.9}"  # 与 step8.1 评估对齐
SAVE_EVERY_ITERATIONS="${SAVE_EVERY_ITERATIONS:-2}"

# PPO hyperparameters
LEARNING_RATE="${LEARNING_RATE:-5e-6}"  # Smoke-test default
BATCH_SIZE="${BATCH_SIZE:-1}"  # Smoke-test default
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"  # Smoke-test default
NUM_EPOCHS="${NUM_EPOCHS:-2}"
CLIP_EPSILON="${CLIP_EPSILON:-0.2}"  # Smoke-test default
VALUE_LOSS_COEF="${VALUE_LOSS_COEF:-0.2}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"  # Smoke-test default
KL_COEF="${KL_COEF:-0.05}"
REF_KL_COEF="${REF_KL_COEF:-0.05}"
REF_KL_SAMPLE_RATE="${REF_KL_SAMPLE_RATE:-0.25}"  # Smoke-test default
REF_KL_UPDATE_INTERVAL="${REF_KL_UPDATE_INTERVAL:-4}"  # Smoke-test default
TARGET_KL="${TARGET_KL:-0.1}"  # Smoke-test default
EARLY_STOP_KL="${EARLY_STOP_KL:-0.5}"  # Smoke-test default
REF_KL_MAX_TOKENS="${REF_KL_MAX_TOKENS:-768}"  # Smoke-test default
WARMUP_STEPS="${WARMUP_STEPS:-20}"  # Smoke-test default
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-12000}"  # 与 step8.1 对齐 (PROMPT_MAX_LEN=12000)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
ACTION_MAX_NEW_TOKENS="${ACTION_MAX_NEW_TOKENS:-512}"
INITIAL_MAX_PROMPT_TOKENS=12000  # 与 step8.1 对齐
INITIAL_MAX_NEW_TOKENS="${INITIAL_MAX_NEW_TOKENS:-4096}"  # Reduced from 8192
INITIAL_ACTION_MAX_EVAL_TOKENS="${INITIAL_ACTION_MAX_EVAL_TOKENS:-$INITIAL_MAX_NEW_TOKENS}"
INITIAL_R_REWARD_SCALE="${INITIAL_R_REWARD_SCALE:-0.40}"
INITIAL_GENERATION_LOSS_WEIGHT="${INITIAL_GENERATION_LOSS_WEIGHT:-1.35}"
INITIAL_GENERATION_TEMPERATURE="${INITIAL_GENERATION_TEMPERATURE:-0.10}"
INITIAL_GENERATION_TOP_P="${INITIAL_GENERATION_TOP_P:-0.95}"
TEMPERATURE="${TEMPERATURE:-0.15}"
TOP_P="${TOP_P:-0.95}"
TEACHER_FAILED_ADV_WEIGHT="${TEACHER_FAILED_ADV_WEIGHT:-0.5}"
PROMPT_OVERFLOW_MODE="${PROMPT_OVERFLOW_MODE:-truncate}"
OPTIMIZE_INITIAL_KC="${OPTIMIZE_INITIAL_KC:-1}"
FOCUS_WINDOW_SIZE="${FOCUS_WINDOW_SIZE:-7}"
TABLE_SUMMARY_CHAR_BUDGET="${TABLE_SUMMARY_CHAR_BUDGET:-1800}"
KC_SUMMARY_CHAR_BUDGET="${KC_SUMMARY_CHAR_BUDGET:-3600}"

# Reward config - Smoke-test defaults
DELTA_POS_SCALE="${DELTA_POS_SCALE:-7.0}"
DELTA_NEG_SCALE="${DELTA_NEG_SCALE:-5.0}"
ROLLBACK_THRESHOLD="${ROLLBACK_THRESHOLD:-0.3}"
ROLLBACK_EXTRA_PENALTY="${ROLLBACK_EXTRA_PENALTY:-1.5}"
HIGH_SCORE_THRESHOLD="${HIGH_SCORE_THRESHOLD:-0.9}"  # 与 step8.1 评估对齐 (stop_threshold=0.9)
HIGH_SCORE_STOP_BONUS="${HIGH_SCORE_STOP_BONUS:-0.8}"
PREMATURE_STOP_PENALTY="${PREMATURE_STOP_PENALTY:-1.0}"
HIGH_SCORE_CONTINUE_PENALTY="${HIGH_SCORE_CONTINUE_PENALTY:-0.15}"
TEACHER_STOP_HIGH_SCORE_BONUS="${TEACHER_STOP_HIGH_SCORE_BONUS:-0.2}"
# Low-score teacher_stop is handled by generic no-progress / termination shaping.
LARGE_EDIT_THRESHOLD="${LARGE_EDIT_THRESHOLD:-3}"
LARGE_EDIT_PENALTY="${LARGE_EDIT_PENALTY:-0.05}"
STEP_PENALTY="${STEP_PENALTY:-0.002}"
NO_PROGRESS_PENALTY="${NO_PROGRESS_PENALTY:-0.01}"
NORM_EQUIVALENT_NOOP_PENALTY="${NORM_EQUIVALENT_NOOP_PENALTY:-0.08}"
NO_PROGRESS_EPS="${NO_PROGRESS_EPS:-0.001}"
MAX_STAGNANT_ROUNDS="${MAX_STAGNANT_ROUNDS:-3}"
MAX_CONSECUTIVE_TEACHER_STOPS="${MAX_CONSECUTIVE_TEACHER_STOPS:-2}"
POSITIVE_CORRECTION_BONUS="${POSITIVE_CORRECTION_BONUS:-0.15}"
FINAL_SCORE_SCALE="${FINAL_SCORE_SCALE:-0.35}"
INEFFECTIVE_TERMINATION_PENALTY="${INEFFECTIVE_TERMINATION_PENALTY:-0.15}"

# Teacher robustness
TEACHER_MAX_RETRIES="${TEACHER_MAX_RETRIES:-3}"
TEACHER_TIMEOUT="${TEACHER_TIMEOUT:-180}"
TEACHER_BACKOFF_BASE="${TEACHER_BACKOFF_BASE:-2.0}"

# Reference model KL
ENABLE_REF_MODEL="${ENABLE_REF_MODEL:-1}"  # Default enabled
REF_MODEL_PATH="${REF_MODEL_PATH:-$MODEL_PATH}"
# Dual-GPU configuration: Policy on GPU 0, Ref on GPU 1
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"  # Use GPU 0 and 1
DEVICE_MAP="${DEVICE_MAP:-cuda:0}"                   # Policy model on GPU 0
REF_MODEL_DEVICE_MAP="${REF_MODEL_DEVICE_MAP:-cuda:1}"  # Ref model on GPU 1
ALLOW_SHARED_REF_DEVICE="${ALLOW_SHARED_REF_DEVICE:-0}"  # Disabled for dual GPU
PROMPT_ROOT="${PROMPT_ROOT:-$WORKDIR/step9/prompt_packs/train_v1}"

# DeepSeek API (use env var if set, otherwise default)
DEEPSEEK_API_URL="${DEEPSEEK_API_URL:-https://api.deepseek.com/v1/chat/completions}"
DEEPSEEK_API_URL_FALLBACKS="${DEEPSEEK_API_URL_FALLBACKS:-}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-DeepSeek-V3.2}"

# Logging
LOG_DIR="$WORKDIR/step9/output/logs"
mkdir -p "$LOG_DIR"
RUN_IN_FOREGROUND="${RUN_IN_FOREGROUND:-0}"
LATEST_STAGE_RUN_FILE="${LATEST_STAGE_RUN_FILE:-}"
STAGE_NAME="${STAGE_NAME:-}"
MASTER_CURRICULUM_LOG_FILE="${MASTER_CURRICULUM_LOG_FILE:-}"
RECORD_EPISODE_RECORDS="${RECORD_EPISODE_RECORDS:-1}"
WRITE_ITERATION_JSON="${WRITE_ITERATION_JSON:-1}"

# 课程模式下使用主日志，否则生成独立日志
if [ -n "$MASTER_CURRICULUM_LOG_FILE" ]; then
  LOG_FILE="$MASTER_CURRICULUM_LOG_FILE"
else
  LOG_FILE="$LOG_DIR/ppo_train_$(date +%Y%m%d_%H%M%S).log"
fi

write_latest_stage_run_file() {
  local status="${1:-stage_started}"
  if [ -z "$LATEST_STAGE_RUN_FILE" ]; then
    return
  fi
  mkdir -p "$(dirname "$LATEST_STAGE_RUN_FILE")"
  {
    echo "time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "status=$status"
    if [ -n "$MASTER_CURRICULUM_LOG_FILE" ]; then
      echo "master_log=$MASTER_CURRICULUM_LOG_FILE"
    fi
    echo "stage_name=$STAGE_NAME"
    echo "latest_stage_log=$LOG_FILE"
    echo "stage_log=$LOG_FILE"
    echo "checkpoint_dir=$CHECKPOINT_DIR"
    echo "output_dir=$OUTPUT_DIR"
    echo "data_dir=$DATA_DIR"
    echo "num_samples=$NUM_SAMPLES"
  } > "$LATEST_STAGE_RUN_FILE"
}

echo "Model: $MODEL_PATH"
echo "Data: $DATA_DIR"
echo "Samples: $NUM_SAMPLES"
echo "Eval ratio: $EVAL_RATIO"
echo "Eval max samples: $EVAL_MAX_SAMPLES"
echo "Eval every iterations: $EVAL_EVERY_ITERATIONS"
echo "Iterations: $NUM_ITERATIONS"
if [ "$EPISODES_PER_ITERATION" = "0" ]; then
  echo "Episodes/iter: auto(train split size)"
else
  echo "Episodes/iter: $EPISODES_PER_ITERATION"
fi
echo "Update every episodes: $UPDATE_EVERY_EPISODES"
echo "LR: $LEARNING_RATE"
echo "KL coef: $KL_COEF"
echo "Python: $PYTHON_BIN"
echo "Policy device_map: $DEVICE_MAP"
echo "Ref device_map: $REF_MODEL_DEVICE_MAP"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Prompt root: $PROMPT_ROOT"
echo "Output log: $LOG_FILE"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Record episode records: $RECORD_EPISODE_RECORDS"
echo "Write iteration JSON: $WRITE_ITERATION_JSON"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: model path not found: $MODEL_PATH"
  exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
  echo "Error: data dir not found: $DATA_DIR"
  exit 1
fi

if [ "${ENABLE_REF_MODEL}" = "1" ] && [ "${REF_KL_COEF}" != "0" ]; then
  # Check if both are on the same CUDA device.
  if [ "$DEVICE_MAP" != "cpu" ] && [ "$REF_MODEL_DEVICE_MAP" != "cpu" ] && [ "$DEVICE_MAP" = "$REF_MODEL_DEVICE_MAP" ]; then
    if [ "${ALLOW_SHARED_REF_DEVICE}" != "1" ]; then
      echo "Error: DEVICE_MAP and REF_MODEL_DEVICE_MAP must be different unless ALLOW_SHARED_REF_DEVICE=1."
      echo "Current: DEVICE_MAP=$DEVICE_MAP, REF_MODEL_DEVICE_MAP=$REF_MODEL_DEVICE_MAP"
      exit 1
    fi
    echo "Warning: policy model and ref_model will share the same visible CUDA device."
  fi
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$CHECKPOINT_DIR"

cd "$WORKDIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES
export DEEPSEEK_API_KEY
export DEEPSEEK_API_URL_FALLBACKS

RESUME_ARG=""
if [ -n "$RESUME_FROM" ]; then
  RESUME_ARG="--resume_from $RESUME_FROM"
fi

CMD=(
  "$PYTHON_BIN" step9/train_ppo.py
  --model_path "$MODEL_PATH"
  --tokenizer_path "$MODEL_PATH"
  --data_dir "$DATA_DIR"
  --output_dir "$OUTPUT_DIR"
  --prompt_root "$PROMPT_ROOT"
  --num_samples "$NUM_SAMPLES"
  --eval_ratio "$EVAL_RATIO"
  --eval_max_samples "$EVAL_MAX_SAMPLES"
  --eval_every_iterations "$EVAL_EVERY_ITERATIONS"
  --num_iterations "$NUM_ITERATIONS"
  --episodes_per_iteration "$EPISODES_PER_ITERATION"
  --update_every_episodes "$UPDATE_EVERY_EPISODES"
  --max_rounds "$MAX_ROUNDS"
  --stop_threshold "$STOP_THRESHOLD"
  --checkpoint_dir "$CHECKPOINT_DIR"
  --save_every_iterations "$SAVE_EVERY_ITERATIONS"
  --auto_resume_latest "$AUTO_RESUME_LATEST"
  --record_episode_records "$RECORD_EPISODE_RECORDS"
  --write_iteration_json "$WRITE_ITERATION_JSON"
  --learning_rate "$LEARNING_RATE"
  --batch_size "$BATCH_SIZE"
  --mini_batch_size "$MINI_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --ppo_epochs "$NUM_EPOCHS"
  --clip_epsilon "$CLIP_EPSILON"
  --value_loss_coef "$VALUE_LOSS_COEF"
  --entropy_coef "$ENTROPY_COEF"
  --kl_coef "$KL_COEF"
  --ref_kl_coef "$REF_KL_COEF"
  --ref_kl_sample_rate "$REF_KL_SAMPLE_RATE"
  --ref_kl_update_interval "$REF_KL_UPDATE_INTERVAL"
  --target_kl "$TARGET_KL"
  --early_stop_kl "$EARLY_STOP_KL"
  --ref_kl_max_tokens "$REF_KL_MAX_TOKENS"
  --warmup_steps "$WARMUP_STEPS"
  --lr_scheduler "$LR_SCHEDULER"
  --gamma "$GAMMA"
  --gae_lambda "$GAE_LAMBDA"
  --max_grad_norm "$MAX_GRAD_NORM"
  --max_prompt_tokens "$MAX_PROMPT_TOKENS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --action_max_new_tokens "$ACTION_MAX_NEW_TOKENS"
  --initial_max_prompt_tokens "$INITIAL_MAX_PROMPT_TOKENS"
  --initial_max_new_tokens "$INITIAL_MAX_NEW_TOKENS"
  --initial_action_max_eval_tokens "$INITIAL_ACTION_MAX_EVAL_TOKENS"
  --initial_r_reward_scale "$INITIAL_R_REWARD_SCALE"
  --initial_generation_loss_weight "$INITIAL_GENERATION_LOSS_WEIGHT"
  --initial_generation_temperature "$INITIAL_GENERATION_TEMPERATURE"
  --initial_generation_top_p "$INITIAL_GENERATION_TOP_P"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --optimize_initial_kc "$OPTIMIZE_INITIAL_KC"
  --focus_window_size "$FOCUS_WINDOW_SIZE"
  --table_summary_char_budget "$TABLE_SUMMARY_CHAR_BUDGET"
  --kc_summary_char_budget "$KC_SUMMARY_CHAR_BUDGET"
  --teacher_failed_adv_weight "$TEACHER_FAILED_ADV_WEIGHT"
  --prompt_overflow_mode "$PROMPT_OVERFLOW_MODE"
  --delta_pos_scale "$DELTA_POS_SCALE"
  --delta_neg_scale "$DELTA_NEG_SCALE"
  --rollback_threshold "$ROLLBACK_THRESHOLD"
  --rollback_extra_penalty "$ROLLBACK_EXTRA_PENALTY"
  --high_score_threshold "$HIGH_SCORE_THRESHOLD"
  --high_score_stop_bonus "$HIGH_SCORE_STOP_BONUS"
  --premature_stop_penalty "$PREMATURE_STOP_PENALTY"
  --high_score_continue_penalty "$HIGH_SCORE_CONTINUE_PENALTY"
  --teacher_stop_high_score_bonus "$TEACHER_STOP_HIGH_SCORE_BONUS"
  --large_edit_threshold "$LARGE_EDIT_THRESHOLD"
  --large_edit_penalty "$LARGE_EDIT_PENALTY"
  --step_penalty "$STEP_PENALTY"
  --no_progress_penalty "$NO_PROGRESS_PENALTY"
  --norm_equivalent_noop_penalty "$NORM_EQUIVALENT_NOOP_PENALTY"
  --no_progress_eps "$NO_PROGRESS_EPS"
  --max_stagnant_rounds "$MAX_STAGNANT_ROUNDS"
  --max_consecutive_teacher_stops "$MAX_CONSECUTIVE_TEACHER_STOPS"
  --positive_correction_bonus "$POSITIVE_CORRECTION_BONUS"
  --final_score_scale "$FINAL_SCORE_SCALE"
  --ineffective_termination_penalty "$INEFFECTIVE_TERMINATION_PENALTY"
  --teacher_max_retries "$TEACHER_MAX_RETRIES"
  --teacher_timeout "$TEACHER_TIMEOUT"
  --teacher_backoff_base "$TEACHER_BACKOFF_BASE"
  --enable_ref_model "$ENABLE_REF_MODEL"
  --ref_model_path "$REF_MODEL_PATH"
  --ref_model_device_map "$REF_MODEL_DEVICE_MAP"
  --allow_shared_ref_device "$ALLOW_SHARED_REF_DEVICE"
  --device_map "$DEVICE_MAP"
  --deepseek_api_url "$DEEPSEEK_API_URL"
  --deepseek_model "$DEEPSEEK_MODEL"
)

if [ -n "$RESUME_FROM" ]; then
  CMD+=(--resume_from "$RESUME_FROM")
fi

if [ "$RUN_IN_FOREGROUND" = "1" ]; then
  echo "Training started in foreground."
  echo "Log file: $LOG_FILE"
  write_latest_stage_run_file "stage_started"
  if [ -n "$MASTER_CURRICULUM_LOG_FILE" ]; then
    "${CMD[@]}" >> "$LOG_FILE" 2>&1
  else
    "${CMD[@]}" > "$LOG_FILE" 2>&1
  fi
  EXIT_CODE=$?
  if [ "$EXIT_CODE" = "0" ]; then
    write_latest_stage_run_file "stage_finished"
  else
    write_latest_stage_run_file "stage_failed"
  fi
  echo "Training finished with exit code: $EXIT_CODE"
  echo "Log file: $LOG_FILE"
  echo "After training: bash step9/run_training_health_report.sh"
  echo "=========================================="
  exit $EXIT_CODE
else
  if [ -n "$MASTER_CURRICULUM_LOG_FILE" ]; then
    nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
  else
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  fi
  TRAIN_PID=$!
  write_latest_stage_run_file "stage_started"
  echo "Training started. PID: $TRAIN_PID"
  echo "Log file: $LOG_FILE"
  echo "Watch log: tail -f $LOG_FILE"
  echo "Stop: kill $TRAIN_PID"
  echo "After training: bash step9/run_training_health_report.sh"
  echo "=========================================="
fi
