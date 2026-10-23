#!/usr/bin/env bash
# 使用openr1环境的DeepSpeed运行脚本（PyTorch 2.5.1 + DeepSpeed 0.15.4）
set -euo pipefail

DATA_DIR="/home/weijie_yu/yjw/ECLO-RL/step6/SFT1"
MODEL_PATH="/home/new_disk/yjw/Model/Qwen3-8B"
# 使用时间戳创建新的checkpoint目录，避免覆盖
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/home/new_disk/yjw/checkpoints/qwen3_sft1_${TIMESTAMP}"
LOG_DIR="/home/weijie_yu/yjw/ECLO-RL/step6/logs"
LOG_PATH="${LOG_DIR}/train_ds_${TIMESTAMP}.log"
DS_CONFIG="./deepspeed_zero3_2gpu.json"

LR="2e-5"
EPOCHS=3
BS=1  # 每卡 batch size
GA=4  # 梯度累积（双卡时总 effective_bs = BS * GA * 2 = 8）
MAX_LEN=30000  # 降低序列长度以适应双卡显存
EVAL_RATIO=0.02
SAVE_STEPS=600

# 激活 openr1 环境（兼容环境）
if [ -f "/home/weijie_yu/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/weijie_yu/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate openr1

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# 优化 PyTorch 内存分配，减少碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "使用openr1环境启动双卡训练..."
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "DeepSpeed版本: $(python -c 'import deepspeed; print(deepspeed.__version__)')"

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
  --save_steps ${SAVE_STEPS} \
  --deepspeed "${DS_CONFIG}" \
  > "${LOG_PATH}" 2>&1 &

echo "已后台启动，日志: ${LOG_PATH}"
echo "使用openr1环境: PyTorch 2.5.1 + DeepSpeed 0.15.4"
echo "双卡训练配置: batch_size=${BS}, gradient_accumulation=${GA}, effective_bs=$((BS*GA*2))"
