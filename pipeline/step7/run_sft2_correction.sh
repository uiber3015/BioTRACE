#!/bin/bash
# ============================================================================#
#                                                                          #
#   ECLO-RL SFT2 修正循环批量运行脚本                                       #
#                                                                          #
#   功能：                                                                #
#   1. 加载SFT1/SFT2模型（Qwen3）                                         #
#   2. 使用DeepSeek API进行策略选择和修正                                    #
#   3. 最大修正轮数：4轮                                                   #
#   4. 双卡并行运行                                                        #
#   5. 生成完整的修正轨迹数据（SFT2训练集）                                 #
#                                                                          #
#   数据来源：                                                            #
#   - 默认从 step6/SFT2 目录读取所有 JSON 文件                             #
#   - 或使用 --test_jsonl 参数指定 JSONL 文件                               #
#                                                                          #
#   使用方法：                                                             #
#   bash run_sft2_correction.sh                                            #
#                                                                          #
#   后台运行：                                                             #
#   nohup bash run_sft2_correction.sh > logs/sft2_correction.log 2>&1 &   #
#                                                                          #
#   命令行参数：                                                           #
#   --sft2_dir     SFT2数据目录（默认: step6/SFT2）                        #
#   --test_jsonl   测试集JSONL文件（可选，与sft2_dir二选一）               #
#   --output_dir   输出目录（默认: step7/sft2_correction_results）         #
#   --max_rounds   最大修正轮数（默认: 4）                                   #
#   --sample_limit 样本数限制（默认: 0 表示全部）                            #
#   --workers      并行数（默认: 2）                                        #
#                                                                          #
#   示例：                                                                #
#   bash run_sft2_correction.sh                           # 使用默认SFT2目录  #
#   bash run_sft2_correction.sh --sample_limit 100        # 只处理100个样本  #
#   bash run_sft2_correction.sh --test_jsonl data.jsonl  # 使用JSONL文件     #
#                                                                          #
# ============================================================================#

# ==================== 环境配置 ====================
export CUDA_VISIBLE_DEVICES=0,1

# 路径配置
WORKSPACE="/home/weijie_yu/yjw/ECLO-RL"
cd $WORKSPACE

# Conda环境
CONDA_ENV="openr1"

# ==================== 默认参数 ====================
SFT2_DIR="step6/SFT2"      # SFT2数据目录
TEST_JSONL=""              # 留空表示使用SFT2_DIR
OUTPUT_DIR="step7/sft2_correction_results"
LOG_DIR="logs"
MAX_ROUNDS=4
SAMPLE_LIMIT=0
WORKERS=2

# ==================== 解析命令行参数 ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --sft2_dir)
            SFT2_DIR="$2"
            shift 2
            ;;
        --test_jsonl)
            TEST_JSONL="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_rounds)
            MAX_ROUNDS="$2"
            shift 2
            ;;
        --sample_limit)
            SAMPLE_LIMIT="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 创建目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sft2_correction_${TIMESTAMP}.log"

# ==================== 打印配置 ====================
echo "=========================================================================="
echo "ECLO-RL SFT2 修正循环批量运行"
echo "时间: $(date)"
echo "=========================================================================="
echo ""
echo "配置参数:"
echo "  - 测试集: $TEST_JSONL"
echo "  - 输出目录: $OUTPUT_DIR"
echo "  - 最大修正轮数: $MAX_ROUNDS"
echo "  - 样本限制: $SAMPLE_LIMIT $([ $SAMPLE_LIMIT -eq 0 ] && echo '(全部)' || echo '')"
echo "  - 并行数: $WORKERS"
echo "  - 日志文件: $LOG_FILE"
echo ""

# ==================== 环境检查 ====================
echo "检查环境..."

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: conda 未安装"
    exit 1
fi

# 检查数据源
SAMPLE_COUNT=0
if [ -d "$SFT2_DIR" ]; then
    SAMPLE_COUNT=$(ls -1 "$SFT2_DIR"/*.json 2>/dev/null | wc -l)
    echo "  ✅ SFT2目录存在: $SFT2_DIR"
    echo "  ✅ 样本数: $SAMPLE_COUNT"
elif [ -f "$TEST_JSONL" ]; then
    SAMPLE_COUNT=$(wc -l < "$TEST_JSONL")
    echo "  ✅ 测试JSONL存在: $TEST_JSONL"
    echo "  ✅ 样本数: $SAMPLE_COUNT"
else
    echo "❌ 错误: 未找到有效的数据源"
    echo "   - SFT2_DIR: $SFT2_DIR $([ -d "$SFT2_DIR" ] && echo '存在' || echo '不存在')"
    echo "   - TEST_JSONL: $TEST_JSONL $([ -f "$TEST_JSONL" ] && echo '存在' || echo '不存在')"
    exit 1
fi

echo "  ✅ conda环境: $CONDA_ENV"
echo ""

# ==================== 激活环境 ====================
# ==================== 预估时间 ====================
if [ $SAMPLE_LIMIT -gt 0 ]; then
    PROCESS_SAMPLES=$SAMPLE_LIMIT
else
    PROCESS_SAMPLES=$SAMPLE_COUNT
fi

echo "预估运行时间:"
echo "  - 样本数: $PROCESS_SAMPLES"
ESTIMATE_MIN=$((PROCESS_SAMPLES * 4 / 60))
ESTIMATE_MAX=$((PROCESS_SAMPLES * 8 / 60))
if [ $ESTIMATE_MIN -lt 1 ]; then
    ESTIMATE_MIN=1
fi
if [ $ESTIMATE_MAX -lt 1 ]; then
    ESTIMATE_MAX=1
fi
echo "  - 预计: ${ESTIMATE_MIN}-${ESTIMATE_MAX} 分钟"
echo ""

# ==================== 开始运行 ====================
echo "=========================================================================="
echo "开始运行..."
echo "=========================================================================="
echo ""

# 构建命令
CMD="python step7/sft2_correction_full.py"
if [ -d "$SFT2_DIR" ]; then
    CMD="$CMD --sft2_dir $SFT2_DIR"
elif [ -f "$TEST_JSONL" ]; then
CMD="$CMD --test_jsonl $TEST_JSONL"
fi
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --max_rounds $MAX_ROUNDS"
CMD="$CMD --workers $WORKERS"
if [ $SAMPLE_LIMIT -gt 0 ]; then
    CMD="$CMD --sample_limit $SAMPLE_LIMIT"
fi

echo "运行命令: $CMD"
echo ""
echo "日志将保存到: $LOG_FILE"
echo ""
echo "按 Ctrl+C 可中断运行"
echo ""

# 激活conda环境（使用正确的路径）
source /home/weijie_yu/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

# 实际运行
eval "$CMD 2>&1" | tee "$LOG_FILE"

# ==================== 运行完成 ====================
EXIT_CODE=$?

echo ""
echo "=========================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 运行完成: $(date)"
else
    echo "❌ 运行失败: $(date)，退出码: $EXIT_CODE"
fi
echo "=========================================================================="
echo ""
echo "结果文件:"
echo "  - 原始数据: $OUTPUT_DIR/sft2_correction_raw_*.json"
echo "  - 分析报告: $OUTPUT_DIR/sft2_correction_report_*.md"
echo "  - 运行日志: $LOG_FILE"
echo ""
echo "中间结果（每10个样本保存一次）:"
echo "  - 临时文件: $OUTPUT_DIR/sft2_correction_temp_*.json"
echo ""
echo "查看结果:"
echo "  cat $OUTPUT_DIR/sft2_correction_report_*.md"
echo ""
echo "查看日志:"
echo "  tail -f $LOG_FILE"
echo ""
echo "查看中间结果:"
echo "  cat $OUTPUT_DIR/sft2_correction_temp_10.json | head -100"
echo ""

exit $EXIT_CODE

