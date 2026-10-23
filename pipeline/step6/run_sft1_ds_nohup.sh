#!/usr/bin/env bash
# 直接可 nohup 的双卡 DeepSpeed 运行脚本，max_len=30k
set -euo pipefail

DATA_DIR="/home/weijie_yu/yjw/ECLO-RL/step6/SFT1"
MODEL_PATH="/home/new_disk/yjw/Model/Qwen3-8B"
# 使用时间戳创建新的checkpoint目录，避免覆盖
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/home/new_disk/yjw/checkpoints/qwen3_sft1_${TIMESTAMP}"
LOG_DIR="/home/weijie_yu/yjw/ECLO-RL/step6/logs"
LOG_PATH="${LOG_DIR}/train_ds_nohup_${TIMESTAMP}.log"
DS_CONFIG="./deepspeed_zero3_2gpu.json"

LR="2e-5"
EPOCHS=3
BS=1
GA=8
MAX_LEN=30000
EVAL_RATIO=0.02

# 激活 conda 环境（非交互 shell 也能用）
if [ -f "/home/weijie_yu/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/weijie_yu/anaconda3/etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1090
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate yjw

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# 优化 PyTorch 内存分配，减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup deepspeed --num_gpus=2 train_sft1.py \
  --data_dir "${DATA_DIR}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_train_batch_size ${BS} \
  --gradient_accumulation_steps ${GA} \
  --num_train_epochs ${EPOCHS} \
  --learning_rate ${LR} \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_length ${MAX_LEN} \
  --eval_ratio ${EVAL_RATIO} \
  --deepspeed "${DS_CONFIG}" \
  > "${LOG_PATH}" 2>&1 &

echo "已后台启动，日志: ${LOG_PATH}"

