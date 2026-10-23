#!/bin/bash

set -euo pipefail

echo "=========================================="
echo "Step9 PPO Curriculum Training"
echo "=========================================="

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKDIR/step9/output/ppo_curriculum_model}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
CURRICULUM_BACKGROUND="${CURRICULUM_BACKGROUND:-0}"
CURRICULUM_LOG_DIR="${CURRICULUM_LOG_DIR:-$OUTPUT_DIR/logs}"
CURRICULUM_LOG_FILE="${CURRICULUM_LOG_FILE:-$CURRICULUM_LOG_DIR/ppo_curriculum.log}"
CURRICULUM_BG_CHILD="${CURRICULUM_BG_CHILD:-0}"
LATEST_CURRICULUM_RUN_FILE="${LATEST_CURRICULUM_RUN_FILE:-$OUTPUT_DIR/latest_curriculum_run.txt}"
STAGE_LOG_DIR="${STAGE_LOG_DIR:-$CURRICULUM_LOG_DIR}"
KEEP_ITER_CHECKPOINTS="${KEEP_ITER_CHECKPOINTS:-3}"

read_latest_curriculum_field() {
  local key="$1"
  if [ ! -f "$LATEST_CURRICULUM_RUN_FILE" ]; then
    return
  fi
  grep -m1 "^${key}=" "$LATEST_CURRICULUM_RUN_FILE" | cut -d= -f2- || true
}

write_latest_curriculum_run_file() {
  local status="$1"
  local stage_name="${2:-$(read_latest_curriculum_field stage_name)}"
  local latest_stage_log="${3:-$(read_latest_curriculum_field latest_stage_log)}"
  local data_dir="${4:-$(read_latest_curriculum_field data_dir)}"
  local num_samples="${5:-$(read_latest_curriculum_field num_samples)}"

  mkdir -p "$(dirname "$LATEST_CURRICULUM_RUN_FILE")"
  {
    echo "time=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "status=$status"
    echo "master_log=$CURRICULUM_LOG_FILE"
    if [ -n "${CURRICULUM_PID:-}" ]; then
      echo "curriculum_pid=$CURRICULUM_PID"
    fi
    if [ -n "$stage_name" ]; then
      echo "stage_name=$stage_name"
    fi
    if [ -n "$latest_stage_log" ]; then
      echo "latest_stage_log=$latest_stage_log"
      echo "stage_log=$latest_stage_log"
    fi
    echo "checkpoint_dir=$CHECKPOINT_DIR"
    echo "output_dir=$OUTPUT_DIR"
    echo "stage_logs_dir=$STAGE_LOG_DIR"
    if [ -n "${CURRICULUM_ROOT:-}" ]; then
      echo "curriculum_root=$CURRICULUM_ROOT"
    fi
    if [ -n "$data_dir" ]; then
      echo "data_dir=$data_dir"
    fi
    if [ -n "$num_samples" ]; then
      echo "num_samples=$num_samples"
    fi
  } > "$LATEST_CURRICULUM_RUN_FILE"
}

if [ "$CURRICULUM_BACKGROUND" = "1" ] && [ "$CURRICULUM_BG_CHILD" != "1" ]; then
  mkdir -p "$CURRICULUM_LOG_DIR"
  export CURRICULUM_BG_CHILD=1
  export CURRICULUM_BACKGROUND=0
  nohup bash "$0" "$@" > "$CURRICULUM_LOG_FILE" 2>&1 &
  CURRICULUM_PID=$!
  disown "$CURRICULUM_PID" 2>/dev/null || true
  write_latest_curriculum_run_file "curriculum_background_started"
  echo "Curriculum training started in background."
  echo "PID: $CURRICULUM_PID"
  echo "Master log: $CURRICULUM_LOG_FILE"
  echo "Latest run file: $LATEST_CURRICULUM_RUN_FILE"
  echo "Stage logs dir: $STAGE_LOG_DIR"
  echo "Stop: kill $CURRICULUM_PID"
  echo "=========================================="
  exit 0
fi

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

# Base paths
MODEL_PATH="${MODEL_PATH:-data/checkpoints/step8_merged}"
CURRICULUM_ROOT="${CURRICULUM_ROOT:-$WORKDIR/step9/curriculum_data}"

# Optional bucket-style curriculum inputs. If stage directories do not exist,
# the script will build them from these three buckets.
EASY_DATA_DIR="${EASY_DATA_DIR:-$WORKDIR/step6/RL_easy}"
MEDIUM_DATA_DIR="${MEDIUM_DATA_DIR:-$WORKDIR/step6/RL_medium}"
HARD_DATA_DIR="${HARD_DATA_DIR:-$WORKDIR/step6/RL_hard}"
REBUILD_STAGE_DIRS_FROM_BUCKETS="${REBUILD_STAGE_DIRS_FROM_BUCKETS:-1}"

# Expected stage datasets. These should already be prepared.
STAGE1_DATA_DIR="${STAGE1_DATA_DIR:-$CURRICULUM_ROOT/stage1_bootstrap}"
STAGE2_DATA_DIR="${STAGE2_DATA_DIR:-$CURRICULUM_ROOT/stage2_transition}"
STAGE3_DATA_DIR="${STAGE3_DATA_DIR:-$CURRICULUM_ROOT/stage3_main}"
STAGE4_DATA_DIR="${STAGE4_DATA_DIR:-$CURRICULUM_ROOT/stage4_hard_refine}"

# Optional final evaluation.
FINAL_EVAL_DATA_DIR="${FINAL_EVAL_DATA_DIR:-}"
FINAL_EVAL_NUM_SAMPLES="${FINAL_EVAL_NUM_SAMPLES:-300}"

# Shared runtime settings
export MODEL_PATH
export OUTPUT_DIR
export CHECKPOINT_DIR
export PROMPT_ROOT="${PROMPT_ROOT:-$WORKDIR/step9/prompt_packs/train_v1}"
export SAVE_EVERY_ITERATIONS="${SAVE_EVERY_ITERATIONS:-1}"
export AUTO_RESUME_LATEST="${AUTO_RESUME_LATEST:-1}"
export RUN_IN_FOREGROUND="${RUN_IN_FOREGROUND:-1}"
export RECORD_EPISODE_RECORDS="${RECORD_EPISODE_RECORDS:-1}"
export WRITE_ITERATION_JSON="${WRITE_ITERATION_JSON:-1}"
export LATEST_STAGE_RUN_FILE="$LATEST_CURRICULUM_RUN_FILE"
export MASTER_CURRICULUM_LOG_FILE="$CURRICULUM_LOG_FILE"
export PYTHON_BIN

# Dual-GPU configuration: Policy on GPU 0, Ref on GPU 1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export DEVICE_MAP="${DEVICE_MAP:-cuda:0}"
export REF_MODEL_DEVICE_MAP="${REF_MODEL_DEVICE_MAP:-cuda:1}"
export ALLOW_SHARED_REF_DEVICE="${ALLOW_SHARED_REF_DEVICE:-0}"
export ENABLE_REF_MODEL="${ENABLE_REF_MODEL:-1}"

# ---------- CUDA Memory Optimization ----------
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Stable common defaults
export BATCH_SIZE="${BATCH_SIZE:-1}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-1}"
export UPDATE_EVERY_EPISODES="${UPDATE_EVERY_EPISODES:-64}"
export MAX_ROUNDS="${MAX_ROUNDS:-5}"
export STOP_THRESHOLD="${STOP_THRESHOLD:-0.9}"
export MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-8000}"
export INITIAL_MAX_PROMPT_TOKENS="${INITIAL_MAX_PROMPT_TOKENS:-8000}"
export INITIAL_MAX_NEW_TOKENS="${INITIAL_MAX_NEW_TOKENS:-4096}"
export INITIAL_ACTION_MAX_EVAL_TOKENS="${INITIAL_ACTION_MAX_EVAL_TOKENS:-$INITIAL_MAX_NEW_TOKENS}"
export ACTION_MAX_NEW_TOKENS="${ACTION_MAX_NEW_TOKENS:-512}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export OPTIMIZE_INITIAL_KC="${OPTIMIZE_INITIAL_KC:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
export NUM_EPOCHS="${NUM_EPOCHS:-2}"
export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export REF_KL_COEF="${REF_KL_COEF:-0.05}"
export REF_KL_SAMPLE_RATE="${REF_KL_SAMPLE_RATE:-0.25}"
export REF_KL_UPDATE_INTERVAL="${REF_KL_UPDATE_INTERVAL:-4}"
export REF_KL_MAX_TOKENS="${REF_KL_MAX_TOKENS:-768}"

# Reward defaults
export DELTA_POS_SCALE="${DELTA_POS_SCALE:-7.0}"
export DELTA_NEG_SCALE="${DELTA_NEG_SCALE:-5.0}"
export ROLLBACK_THRESHOLD="${ROLLBACK_THRESHOLD:-0.3}"
export ROLLBACK_EXTRA_PENALTY="${ROLLBACK_EXTRA_PENALTY:-1.5}"
export HIGH_SCORE_THRESHOLD="${HIGH_SCORE_THRESHOLD:-0.9}"
export HIGH_SCORE_STOP_BONUS="${HIGH_SCORE_STOP_BONUS:-0.8}"
export PREMATURE_STOP_PENALTY="${PREMATURE_STOP_PENALTY:-1.0}"
export HIGH_SCORE_CONTINUE_PENALTY="${HIGH_SCORE_CONTINUE_PENALTY:-0.15}"
export TEACHER_STOP_HIGH_SCORE_BONUS="${TEACHER_STOP_HIGH_SCORE_BONUS:-0.2}"
export STEP_PENALTY="${STEP_PENALTY:-0.002}"
export NO_PROGRESS_PENALTY="${NO_PROGRESS_PENALTY:-0.01}"
export NO_PROGRESS_EPS="${NO_PROGRESS_EPS:-0.001}"
export MAX_STAGNANT_ROUNDS="${MAX_STAGNANT_ROUNDS:-4}"
export POSITIVE_CORRECTION_BONUS="${POSITIVE_CORRECTION_BONUS:-0.15}"
export FINAL_SCORE_SCALE="${FINAL_SCORE_SCALE:-0.35}"
export INEFFECTIVE_TERMINATION_PENALTY="${INEFFECTIVE_TERMINATION_PENALTY:-0.15}"

# To maximize useful rollout budget, the curriculum runner defaults to no per-stage held-out split.
# Run step9/evaluate_ppo.py after training for final evaluation.
export EVAL_RATIO="${EVAL_RATIO:-0}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
export EVAL_EVERY_ITERATIONS="${EVAL_EVERY_ITERATIONS:-0}"

CURRENT_STAGE_NAME=""
CURRENT_STAGE_DATA_DIR=""
CURRENT_STAGE_NUM_SAMPLES=""

on_curriculum_exit() {
  local exit_code=$?
  if [ "$exit_code" -ne 0 ]; then
    write_latest_curriculum_run_file \
      "curriculum_failed" \
      "$CURRENT_STAGE_NAME" \
      "$(read_latest_curriculum_field latest_stage_log)" \
      "$CURRENT_STAGE_DATA_DIR" \
      "$CURRENT_STAGE_NUM_SAMPLES"
    echo "Curriculum training failed with exit code: $exit_code"
  fi
}

trap on_curriculum_exit EXIT

prune_old_iter_checkpoints() {
  local keep_n="$1"
  if [ "$keep_n" -le 0 ]; then
    return
  fi

  mapfile -t iter_ckpts < <(find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name 'ppo_ckpt_iter_*.pt' | sort)
  local total="${#iter_ckpts[@]}"
  if [ "$total" -le "$keep_n" ]; then
    return
  fi

  local remove_n=$((total - keep_n))
  local idx
  for ((idx=0; idx<remove_n; idx++)); do
    rm -f "${iter_ckpts[$idx]}"
  done
  echo "Pruned $remove_n old iteration checkpoints. Kept latest $keep_n iterative checkpoints plus latest snapshots."
}

count_json_files() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' '
}

ensure_bucket_dir() {
  local label="$1"
  local dir="$2"
  if [ ! -d "$dir" ]; then
    echo "Error: $label bucket dir not found: $dir"
    exit 1
  fi
  local count
  count="$(count_json_files "$dir")"
  if [ "$count" = "0" ]; then
    echo "Error: $label bucket dir contains no json files: $dir"
    exit 1
  fi
}

copy_json_slice() {
  local src_dir="$1"
  local dst_dir="$2"
  local start_idx="$3"
  local take_n="$4"

  # Use natural sort so offsets follow numeric file ids (1,2,3...) instead of
  # lexicographic order (1,10,100...), otherwise curriculum slices drift away
  # from the intended contiguous ranges.
  mapfile -t files < <(find "$src_dir" -maxdepth 1 -type f -name '*.json' | sort -V)
  local available="${#files[@]}"
  if [ "$start_idx" -lt 0 ]; then
    echo "Error: start_idx must be >= 0 for $src_dir"
    exit 1
  fi
  local end_idx=$((start_idx + take_n))
  if [ "$available" -lt "$end_idx" ]; then
    echo "Error: requested slice [$start_idx, $end_idx) from $src_dir, but only $available files are available"
    exit 1
  fi
  local idx
  for ((idx=start_idx; idx<end_idx; idx++)); do
    cp "${files[$idx]}" "$dst_dir/"
  done
}

prepare_stage_from_buckets() {
  local target_dir="$1"
  local easy_n="$2"
  local medium_n="$3"
  local hard_n="$4"
  local easy_offset="${5:-0}"
  local medium_offset="${6:-0}"
  local hard_offset="${7:-0}"

  mkdir -p "$target_dir"
  find "$target_dir" -maxdepth 1 -type f -name '*.json' -delete

  if [ "$easy_n" -gt 0 ]; then
    copy_json_slice "$EASY_DATA_DIR" "$target_dir" "$easy_offset" "$easy_n"
  fi
  if [ "$medium_n" -gt 0 ]; then
    copy_json_slice "$MEDIUM_DATA_DIR" "$target_dir" "$medium_offset" "$medium_n"
  fi
  if [ "$hard_n" -gt 0 ]; then
    copy_json_slice "$HARD_DATA_DIR" "$target_dir" "$hard_offset" "$hard_n"
  fi
}

ensure_stage_dir() {
  local label="$1"
  local dir="$2"
  if [ ! -d "$dir" ]; then
    echo "Error: $label data dir not found: $dir"
    exit 1
  fi
  local count
  count="$(count_json_files "$dir")"
  if [ "$count" = "0" ]; then
    echo "Error: $label data dir contains no json files: $dir"
    exit 1
  fi
}

prepare_curriculum_dirs_if_needed() {
  if [ "$REBUILD_STAGE_DIRS_FROM_BUCKETS" = "1" ]; then
    echo "Rebuilding stage directories from easy/medium/hard buckets..."
    ensure_bucket_dir "Easy" "$EASY_DATA_DIR"
    ensure_bucket_dir "Medium" "$MEDIUM_DATA_DIR"
    ensure_bucket_dir "Hard" "$HARD_DATA_DIR"

    prepare_stage_from_buckets "$STAGE1_DATA_DIR" "${STAGE1_EASY_COUNT:-768}" 0 0 "${STAGE1_EASY_OFFSET:-0}" 0 0
    prepare_stage_from_buckets "$STAGE2_DATA_DIR" "${STAGE2_EASY_COUNT:-512}" "${STAGE2_MEDIUM_COUNT:-512}" 0 "${STAGE2_EASY_OFFSET:-0}" "${STAGE2_MEDIUM_OFFSET:-0}" 0
    prepare_stage_from_buckets "$STAGE3_DATA_DIR" "${STAGE3_EASY_COUNT:-256}" "${STAGE3_MEDIUM_COUNT:-640}" "${STAGE3_HARD_COUNT:-640}" "${STAGE3_EASY_OFFSET:-0}" "${STAGE3_MEDIUM_OFFSET:-0}" "${STAGE3_HARD_OFFSET:-0}"
    prepare_stage_from_buckets "$STAGE4_DATA_DIR" 0 "${STAGE4_MEDIUM_COUNT:-256}" "${STAGE4_HARD_COUNT:-1024}" 0 "${STAGE4_MEDIUM_OFFSET:-0}" "${STAGE4_HARD_OFFSET:-0}"
    return
  fi

  local missing_stage_dirs=0
  for d in "$STAGE1_DATA_DIR" "$STAGE2_DATA_DIR" "$STAGE3_DATA_DIR" "$STAGE4_DATA_DIR"; do
    if [ ! -d "$d" ]; then
      missing_stage_dirs=1
      break
    fi
  done

  if [ "$missing_stage_dirs" = "0" ]; then
    return
  fi

  echo "Stage directories missing. Building them from easy/medium/hard buckets..."
  ensure_bucket_dir "Easy" "$EASY_DATA_DIR"
  ensure_bucket_dir "Medium" "$MEDIUM_DATA_DIR"
  ensure_bucket_dir "Hard" "$HARD_DATA_DIR"

  prepare_stage_from_buckets "$STAGE1_DATA_DIR" "${STAGE1_EASY_COUNT:-768}" 0 0 "${STAGE1_EASY_OFFSET:-0}" 0 0
  prepare_stage_from_buckets "$STAGE2_DATA_DIR" "${STAGE2_EASY_COUNT:-512}" "${STAGE2_MEDIUM_COUNT:-512}" 0 "${STAGE2_EASY_OFFSET:-0}" "${STAGE2_MEDIUM_OFFSET:-0}" 0
  prepare_stage_from_buckets "$STAGE3_DATA_DIR" "${STAGE3_EASY_COUNT:-256}" "${STAGE3_MEDIUM_COUNT:-640}" "${STAGE3_HARD_COUNT:-640}" "${STAGE3_EASY_OFFSET:-0}" "${STAGE3_MEDIUM_OFFSET:-0}" "${STAGE3_HARD_OFFSET:-0}"
  prepare_stage_from_buckets "$STAGE4_DATA_DIR" 0 "${STAGE4_MEDIUM_COUNT:-256}" "${STAGE4_HARD_COUNT:-1024}" 0 "${STAGE4_MEDIUM_OFFSET:-0}" "${STAGE4_HARD_OFFSET:-0}"
}

run_stage() {
  local label="$1"
  local data_dir="$2"
  local total_iters="$3"
  local lr="$4"
  local clip="$5"
  local entropy="$6"
  local target_kl="$7"
  local early_stop_kl="$8"

  local num_samples
  num_samples="$(count_json_files "$data_dir")"
  CURRENT_STAGE_NAME="$label"
  CURRENT_STAGE_DATA_DIR="$data_dir"
  CURRENT_STAGE_NUM_SAMPLES="$num_samples"

  echo "------------------------------------------"
  echo "Running $label"
  echo "  data_dir=$data_dir"
  echo "  num_samples=$num_samples"
  echo "  total_iterations=$total_iters"
  echo "  learning_rate=$lr"
  echo "  clip_epsilon=$clip"
  echo "  entropy_coef=$entropy"
  echo "  target_kl=$target_kl"
  echo "  early_stop_kl=$early_stop_kl"
  echo "------------------------------------------"

  # 每个阶段使用独立的 checkpoint 目录，避免 auto_resume 冲突
  export CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/${label// /_}"

  export DATA_DIR="$data_dir"
  export NUM_SAMPLES="$num_samples"
  export NUM_ITERATIONS="$total_iters"
  export EPISODES_PER_ITERATION=0
  export STAGE_NAME="$label"
  export LEARNING_RATE="$lr"
  export CLIP_EPSILON="$clip"
  export ENTROPY_COEF="$entropy"
  export TARGET_KL="$target_kl"
  export EARLY_STOP_KL="$early_stop_kl"

  write_latest_curriculum_run_file "stage_pending" "$label" "" "$data_dir" "$num_samples"
  if bash "$WORKDIR/step9/run_ppo.sh"; then
    prune_old_iter_checkpoints "$KEEP_ITER_CHECKPOINTS"
    write_latest_curriculum_run_file "stage_finished" "$label" "$(read_latest_curriculum_field latest_stage_log)" "$data_dir" "$num_samples"
  else
    local exit_code=$?
    write_latest_curriculum_run_file "stage_failed" "$label" "$(read_latest_curriculum_field latest_stage_log)" "$data_dir" "$num_samples"
    echo "Stage failed: $label (exit code: $exit_code)"
    return "$exit_code"
  fi
}

prepare_curriculum_dirs_if_needed
ensure_stage_dir "Stage1" "$STAGE1_DATA_DIR"
ensure_stage_dir "Stage2" "$STAGE2_DATA_DIR"
ensure_stage_dir "Stage3" "$STAGE3_DATA_DIR"
ensure_stage_dir "Stage4" "$STAGE4_DATA_DIR"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$(dirname "$LATEST_CURRICULUM_RUN_FILE")"

echo "Curriculum root: $CURRICULUM_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Python: $PYTHON_BIN"
echo "Latest run file: $LATEST_CURRICULUM_RUN_FILE"
echo "Easy bucket: $EASY_DATA_DIR"
echo "Medium bucket: $MEDIUM_DATA_DIR"
echo "Hard bucket: $HARD_DATA_DIR"
echo "Rebuild stages from buckets: $REBUILD_STAGE_DIRS_FROM_BUCKETS"
echo "Prompt root: $PROMPT_ROOT"
echo "Foreground mode: $RUN_IN_FOREGROUND"
echo "Curriculum background mode: $CURRICULUM_BACKGROUND"
echo "Curriculum log dir: $CURRICULUM_LOG_DIR"
echo "Stage logs dir: $STAGE_LOG_DIR"
echo "Keep iterative checkpoints: $KEEP_ITER_CHECKPOINTS"
echo "Record episode records: $RECORD_EPISODE_RECORDS"
echo "Write iteration JSON: $WRITE_ITERATION_JSON"

write_latest_curriculum_run_file "curriculum_started"

# Cumulative iteration schedule defaults to 1 -> 2 -> 4 -> 5 and can be overridden.
run_stage "Stage1 Bootstrap" "$STAGE1_DATA_DIR" "${STAGE1_TOTAL_ITERS:-1}" "${STAGE1_LEARNING_RATE:-5e-6}" "${STAGE1_CLIP_EPSILON:-0.20}" "${STAGE1_ENTROPY_COEF:-0.010}" "${STAGE1_TARGET_KL:-0.10}" "${STAGE1_EARLY_STOP_KL:-0.50}"
run_stage "Stage2 Transition" "$STAGE2_DATA_DIR" "${STAGE2_TOTAL_ITERS:-2}" "${STAGE2_LEARNING_RATE:-4e-6}" "${STAGE2_CLIP_EPSILON:-0.15}" "${STAGE2_ENTROPY_COEF:-0.008}" "${STAGE2_TARGET_KL:-0.07}" "${STAGE2_EARLY_STOP_KL:-0.35}"
run_stage "Stage3 Main" "$STAGE3_DATA_DIR" "${STAGE3_TOTAL_ITERS:-4}" "${STAGE3_LEARNING_RATE:-3e-6}" "${STAGE3_CLIP_EPSILON:-0.15}" "${STAGE3_ENTROPY_COEF:-0.005}" "${STAGE3_TARGET_KL:-0.05}" "${STAGE3_EARLY_STOP_KL:-0.25}"
run_stage "Stage4 Hard Refinement" "$STAGE4_DATA_DIR" "${STAGE4_TOTAL_ITERS:-5}" "${STAGE4_LEARNING_RATE:-2e-6}" "${STAGE4_CLIP_EPSILON:-0.12}" "${STAGE4_ENTROPY_COEF:-0.003}" "${STAGE4_TARGET_KL:-0.04}" "${STAGE4_EARLY_STOP_KL:-0.20}"

if [ -n "$FINAL_EVAL_DATA_DIR" ]; then
  echo "------------------------------------------"
  echo "Running final evaluation"
  echo "  test_data=$FINAL_EVAL_DATA_DIR"
  echo "  num_samples=$FINAL_EVAL_NUM_SAMPLES"
  echo "------------------------------------------"
  "$PYTHON_BIN" "$WORKDIR/step9/evaluate_ppo.py" \
    --model_path "$OUTPUT_DIR/final" \
    --test_data "$FINAL_EVAL_DATA_DIR" \
    --output_dir "$OUTPUT_DIR/evaluation" \
    --num_samples "$FINAL_EVAL_NUM_SAMPLES" \
    --max_rounds "$MAX_ROUNDS" \
    --stop_threshold "$STOP_THRESHOLD" \
    --deepseek_api_url "$DEEPSEEK_API_URL" \
    --deepseek_api_key "$DEEPSEEK_API_KEY" \
    --deepseek_model "$DEEPSEEK_MODEL"
fi

echo "=========================================="
echo "Curriculum PPO training finished."
echo "Final model: $OUTPUT_DIR/final"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Latest run file: $LATEST_CURRICULUM_RUN_FILE"
echo "=========================================="
write_latest_curriculum_run_file "curriculum_finished"
