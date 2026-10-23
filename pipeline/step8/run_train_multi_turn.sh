#!/usr/bin/env bash
# Step8 多轮修正增量学习训练脚本 - 使用 LoRA
set -euo pipefail

# 基座模型 (Step6 训练后的模型)
BASE_MODEL_PATH="/home/new_disk/yjw/checkpoints/qwen3_sft1_20260225_183025"

# 数据路径
SFT1_DATA_DIR="/home/weijie_yu/yjw/ECLO-RL/step6/SFT1"
MULTI_TURN_DATA="/home/weijie_yu/yjw/ECLO-RL/step8/training_data/sft_training_data_full_trajectory.json"

# 输出路径
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="/home/new_disk/yjw/checkpoints/step8_multi_turn_lora_${TIMESTAMP}"
LOG_DIR="/home/weijie_yu/yjw/ECLO-RL/step8/logs"
LOG_PATH="${LOG_DIR}/train_multi_turn_${TIMESTAMP}.log"

# 训练超参数 - 防遗忘策略
BATCH_SIZE=1
GRADIENT_ACCUMULATION=8
LEARNING_RATE=5e-5  # 降低学习率，减少灾难性遗忘
NUM_EPOCHS=1  # 减少训练轮数
MAX_SEQ_LENGTH=8192

# 混合比例 - 提高SFT1比例，保留原始能力
MIX_RATIO=0.8

# STOP 信号过采样倍数
STOP_OVERSAMPLE_RATIO=5

# LoRA 配置 - 保守配置
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.1

# 激活 openr1 环境
if [ -f "/home/weijie_yu/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/weijie_yu/anaconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
conda activate openr1

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# 优化 PyTorch 内存分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Step8 多轮修正增量学习训练 (LoRA)"
echo "=========================================="
echo "基座模型: ${BASE_MODEL_PATH}"
echo "SFT1 数据: ${SFT1_DATA_DIR}"
echo "多轮修正数据: ${MULTI_TURN_DATA}"
echo "输出目录: ${OUTPUT_DIR}"
echo "学习率: ${LEARNING_RATE}"
echo "Epochs: ${NUM_EPOCHS}"
echo "LoRA R: ${LORA_R}"
echo "混合比例: ${MIX_RATIO} (SFT1 : 多轮修正)"
echo "STOP 过采样: ${STOP_OVERSAMPLE_RATIO}x"
echo "有效 Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION))"
echo ""
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "DeepSpeed版本: $(python -c 'import deepspeed; print(deepspeed.__version__)')"
echo "=========================================="
echo "日志: ${LOG_PATH}"
echo ""

# 后台运行训练
nohup python -u train_multi_turn.py \
  --base_model_path "${BASE_MODEL_PATH}" \
  --sft1_data_dir "${SFT1_DATA_DIR}" \
  --multi_turn_data "${MULTI_TURN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
  --learning_rate ${LEARNING_RATE} \
  --num_epochs ${NUM_EPOCHS} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --mix_ratio ${MIX_RATIO} \
  --stop_oversample_ratio ${STOP_OVERSAMPLE_RATIO} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  > "${LOG_PATH}" 2>&1 &

TRAIN_PID=$!
echo "训练已后台启动，PID: ${TRAIN_PID}"
echo "使用 openr1 环境: PyTorch + DeepSpeed"
echo ""
echo "查看日志: tail -f ${LOG_PATH}"
echo "查看进程: ps aux | grep train_multi_turn"

