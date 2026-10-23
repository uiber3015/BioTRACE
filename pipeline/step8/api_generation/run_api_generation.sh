#!/bin/bash
# 四线程 API 知识链生成启动脚本

# 设置工作目录
cd "$(dirname "$0")"

# 创建必要的目录
mkdir -p logs

# 设置 Python 环境（根据实际情况修改）
# export PATH="/path/to/conda/bin:$PATH"
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate your_env

# 运行参数说明：
# --num_samples: 处理的样本数量（默认全部）
# --workers: 并行线程数（默认4）

echo "=========================================="
echo "  四线程 API 知识链生成启动"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "工作目录: $(pwd)"
echo ""

# 启动脚本（修改 num_samples 参数）
nohup python -u api_kc_generation.py \
    --num_samples 2749 \
    --workers 4 \
    > logs/api_kc_generation_$(date '+%Y%m%d_%H%M%S').log 2>&1 &

# 获取进程 ID
PID=$!

echo "进程已启动"
echo "PID: $PID"
echo "日志: logs/api_kc_generation_*.log"
echo ""
echo "查看进度: tail -f logs/api_kc_generation_*.log"
echo "查看进程: ps aux | grep api_kc_generation"
echo "停止进程: kill $PID"
