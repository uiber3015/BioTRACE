#!/bin/bash
# 双向并行 API 生成脚本
# 用于加速 DeepSeek 和 Gemini 模型的评测

echo "=========================================="
echo "  双向并行 API 生成"
echo "=========================================="

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查是否有正在运行的进程
check_process() {
    local model=$1
    if pgrep -f "api_kc_bidirectional.*--model $model" > /dev/null; then
        echo "⚠️  $model 双向进程正在运行中"
        return 1
    fi
    return 0
}

# 启动双向处理
start_bidirectional() {
    local model=$1
    local workers=${2:-2}
    
    echo ""
    echo "启动 $model 双向处理 (workers=$workers)..."
    echo "----------------------------------------"
    
    nohup python3 api_kc_bidirectional.py \
        --model "$model" \
        --workers "$workers" \
        > "logs/bidirectional_${model}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    
    echo "✅ 已启动 $model 双向处理"
    echo "   进程 PID: $!"
    echo "   日志: logs/bidirectional_${model}_*.log"
}

# 主菜单
echo ""
echo "请选择操作:"
echo "1) 启动 DeepSeek-V3.2 双向处理"
echo "2) 启动 Gemini-3-Flash 双向处理"
echo "3) 同时启动 DeepSeek 和 Gemini 双向处理"
echo "4) 查看当前进程状态"
echo "5) 退出"
echo ""

read -p "请输入选项 [1-5]: " choice

case $choice in
    1)
        if check_process "deepseek-v3.2"; then
            start_bidirectional "deepseek-v3.2" 2
        fi
        ;;
    2)
        if check_process "gemini-3-flash-preview"; then
            start_bidirectional "gemini-3-flash-preview" 2
        fi
        ;;
    3)
        if check_process "deepseek-v3.2"; then
            start_bidirectional "deepseek-v3.2" 2
        fi
        sleep 2
        if check_process "gemini-3-flash-preview"; then
            start_bidirectional "gemini-3-flash-preview" 2
        fi
        ;;
    4)
        echo ""
        echo "当前运行的双向进程:"
        ps aux | grep "api_kc_bidirectional" | grep -v grep || echo "无"
        ;;
    5)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
