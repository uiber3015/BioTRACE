#!/bin/bash
# 自动检测Task2完成后启动知识链判断
# 用途：等待Task2运行完毕，自动启动judge_kc_potential.py

echo "========================================"
echo "自动检测并启动知识链判断"
echo "========================================"
echo ""

# 进入脚本所在目录
cd "$(dirname "$0")"

# 配置
CHECK_INTERVAL=300  # 检查间隔（秒），默认5分钟
MAX_WAIT_TIME=86400 # 最大等待时间（秒），默认24小时

echo "配置信息:"
echo "  检查间隔: ${CHECK_INTERVAL}秒 ($(($CHECK_INTERVAL / 60))分钟)"
echo "  最大等待: ${MAX_WAIT_TIME}秒 ($(($MAX_WAIT_TIME / 3600))小时)"
echo ""

# 检查Task2是否在运行
check_task2_running() {
    # 查找run_task2_parallel.py进程
    ps aux | grep -E 'python.*run_task2_parallel.py' | grep -v grep > /dev/null
    return $?
}

# 获取Task2进度
get_task2_progress() {
    # 从日志文件获取最后的进度信息
    if [ -f "logs/task2_regeneration_parallel.log" ]; then
        tail -100 logs/task2_regeneration_parallel.log | grep -E "进度:|任务2 重生成汇总" | tail -1
    else
        echo "日志文件不存在"
    fi
}

# 开始监控
start_time=$(date +%s)
check_count=0

echo "[监控开始] $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

while true; do
    check_count=$((check_count + 1))
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    echo "----------------------------------------"
    echo "[检查 #$check_count] $(date '+%Y-%m-%d %H:%M:%S')"
    echo "已等待: $(($elapsed / 60))分钟"
    echo ""
    
    # 检查Task2是否还在运行
    if check_task2_running; then
        echo "状态: Task2 正在运行中..."
        
        # 显示进度（如果有）
        progress=$(get_task2_progress)
        if [ ! -z "$progress" ]; then
            echo "进度: $progress"
        fi
        
        echo ""
        echo "等待 ${CHECK_INTERVAL}秒 后再次检查..."
        echo ""
        
    else
        echo "状态: ✓ Task2 已完成！"
        echo ""
        
        # 等待2秒确保进程完全结束
        sleep 2
        
        # 再次确认
        if check_task2_running; then
            echo "进程仍在运行，继续等待..."
            sleep $CHECK_INTERVAL
            continue
        fi
        
        echo "========================================"
        echo "Task2 确认完成，开始启动知识链判断"
        echo "========================================"
        echo ""
        
        # 检查judge_kc_potential.py是否存在
        if [ ! -f "judge_kc_potential.py" ]; then
            echo "❌ 错误: judge_kc_potential.py 不存在"
            exit 1
        fi
        
        # 检查是否已经在运行
        if ps aux | grep -E 'python.*judge_kc_potential.py' | grep -v grep > /dev/null; then
            echo "⚠️ 警告: judge_kc_potential.py 已在运行"
            echo "现有进程:"
            ps aux | grep -E 'python.*judge_kc_potential.py' | grep -v grep
            echo ""
            read -p "是否停止现有进程并重新启动? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                pkill -9 -f "judge_kc_potential.py"
                sleep 2
            else
                echo "保持现有进程，退出监控"
                exit 0
            fi
        fi
        
        # 创建日志目录
        mkdir -p logs
        
        # 启动知识链判断
        echo "启动命令: nohup python -u judge_kc_potential.py > logs/kc_judgment_nohup.log 2>&1 &"
        nohup python -u judge_kc_potential.py > logs/kc_judgment_nohup.log 2>&1 &
        
        NEW_PID=$!
        
        echo ""
        echo "========================================"
        echo "✓ 知识链判断已启动！"
        echo "========================================"
        echo "进程 PID: $NEW_PID"
        echo "日志文件:"
        echo "  - logs/kc_potential_judgment.log"
        echo "  - logs/kc_judgment_nohup.log"
        echo ""
        echo "监控命令:"
        echo "  tail -f logs/kc_potential_judgment.log"
        echo "  python view_kc_judgments.py"
        echo ""
        echo "停止命令:"
        echo "  kill -9 $NEW_PID"
        echo "========================================"
        
        # 任务完成，退出监控
        exit 0
    fi
    
    # 检查是否超过最大等待时间
    if [ $elapsed -gt $MAX_WAIT_TIME ]; then
        echo ""
        echo "========================================"
        echo "⚠️ 已达到最大等待时间 ($(($MAX_WAIT_TIME / 3600))小时)"
        echo "========================================"
        echo "Task2 仍在运行，退出监控"
        echo "请手动检查任务状态"
        exit 1
    fi
    
    # 等待下次检查
    sleep $CHECK_INTERVAL
done
