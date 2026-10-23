#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Step8 evaluate_comparison (nohup)"
echo "=========================================="

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WORKDIR"

# Runtime knobs (override by env vars)
MODEL_PATH="${MODEL_PATH:-data/checkpoints/step8_merged}"
MODEL_NAME="${MODEL_NAME:-Step8_merged_eval}"
# 每张GPU评测样本数（总样本数会按 GPU 数自动计算）
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-100}"
MAX_ROUNDS="${MAX_ROUNDS:-5}"
# Backward-compatible single-device knob; preferred knob is GPU_LIST (e.g. "0,1").
CUDA_DEVICE="${CUDA_DEVICE:-}"
GPU_LIST="${GPU_LIST:-${CUDA_DEVICE:-0,1}}"
CONDA_ENV="${CONDA_ENV:-openr1}"

LOG_DIR="${LOG_DIR:-$WORKDIR/step8/logs}"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

if [ ! -f "$WORKDIR/step8/evaluate_comparison.py" ]; then
  echo "Error: missing $WORKDIR/step8/evaluate_comparison.py"
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: model path not found: $MODEL_PATH"
  exit 1
fi

# Activate conda env if available
if [ -f "/home/weijie_yu/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/weijie_yu/anaconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

export PYTHONUNBUFFERED=1

echo "Model: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Max rounds: $MAX_ROUNDS"
echo "GPU list: $GPU_LIST"

IFS=',' read -r -a RAW_GPU_ARRAY <<< "$GPU_LIST"
GPU_ARRAY=()
for raw_gpu in "${RAW_GPU_ARRAY[@]}"; do
  gpu="$(echo "$raw_gpu" | xargs)"
  if [ -n "$gpu" ]; then
    GPU_ARRAY+=("$gpu")
  fi
done

NUM_SHARDS="${#GPU_ARRAY[@]}"
if [ "$NUM_SHARDS" -le 0 ]; then
  echo "Error: no valid GPUs parsed from GPU_LIST=$GPU_LIST"
  exit 1
fi

# 通过“总样本 = 每GPU样本 * GPU数”保证每张GPU拿到固定样本数
TEST_SAMPLES=$((SAMPLES_PER_GPU * NUM_SHARDS))
if [ "$TEST_SAMPLES" -le 0 ]; then
  echo "Error: invalid computed TEST_SAMPLES=$TEST_SAMPLES (SAMPLES_PER_GPU=$SAMPLES_PER_GPU, NUM_SHARDS=$NUM_SHARDS)"
  exit 1
fi

echo "Samples per GPU: $SAMPLES_PER_GPU"
echo "Total test samples: $TEST_SAMPLES"
echo "Parallel shards: $NUM_SHARDS"
echo "Starting parallel evaluation in background..."

PIDS=()
LOG_FILES=()
for shard_id in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$shard_id]}"
  LOG_FILE="$LOG_DIR/evaluate_comparison_${TIMESTAMP}_shard${shard_id}of${NUM_SHARDS}_gpu${gpu}.log"
  OUTPUT_TAG="shard${shard_id}of${NUM_SHARDS}_gpu${gpu}_${TIMESTAMP}"

  echo "  -> Launch shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u step8/evaluate_comparison.py \
    --model_path "$MODEL_PATH" \
    --model_name "${MODEL_NAME}_shard${shard_id}of${NUM_SHARDS}" \
    --test_samples "$TEST_SAMPLES" \
    --max_rounds "$MAX_ROUNDS" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$shard_id" \
    --output_dir "step8/logs" \
    --output_tag "$OUTPUT_TAG" \
    > "$LOG_FILE" 2>&1 &

  pid=$!
  PIDS+=("$pid")
  LOG_FILES+=("$LOG_FILE")
  echo "     PID=$pid, log=$LOG_FILE"
done

echo ""
echo "All shards started."
for idx in "${!PIDS[@]}"; do
  echo "  shard $idx -> PID=${PIDS[$idx]}, log=${LOG_FILES[$idx]}"
done
echo "Watch logs:"
for log_file in "${LOG_FILES[@]}"; do
  echo "  tail -f $log_file"
done
echo "Stop all shards: kill ${PIDS[*]}"
echo "=========================================="
