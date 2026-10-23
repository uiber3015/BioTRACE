#!/bin/bash
# 重新生成失败样本 - 双卡运行脚本
# 从评测结果中自动检测需要重新生成的样本

cd "$(dirname "$0")"

# 日志目录
LOG_DIR="logs"
mkdir -p $LOG_DIR

# regenerate_failures.py 显式参数
INITIAL_PROMPT_MAX_LENGTH=30000
INITIAL_MAX_NEW_TOKENS=16384
STRATEGY_PROMPT_MAX_LENGTH=25000

# 时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 从评测结果中提取失败样本ID
echo "正在检测需要重新生成的样本..."
python3 -c "
import json
import sys

# 评测结果文件（最新的一次）
eval_files = [
    'data/logs/evaluation_20260313_105950_659133_shard0of2_shard0of2_gpu0_20260313_105947.json',
    'data/logs/evaluation_20260313_105950_655276_shard1of2_shard1of2_gpu1_20260313_105947.json'
]

failed_ids = set()
for fname in eval_files:
    try:
        with open(fname) as f:
            data = json.load(f)
        samples = data.get('model', {}).get('samples', [])
        for s in samples:
            sid = s.get('sample_id', '')
            error = s.get('error', '')
            
            # 检查初始生成是否为空
            initial_kc = s.get('initial_kc', {})
            is_initial_empty = (
                not initial_kc or 
                initial_kc == {} or 
                initial_kc.get('knowledge_mining') is None or
                initial_kc.get('knowledge_mining') == []
            )
            
            # 检查是否是API/网络错误
            is_api_error = (
                'HTTPSConnectionPool' in error or
                'API call failed' in error or
                'dial tcp' in error or
                'No available channel' in error or
                'model_not_found' in error or
                'Connection' in error or
                'timeout' in error.lower()
            )
            
            if is_initial_empty or is_api_error:
                failed_ids.add(sid)
    except Exception as e:
        print(f'警告: 处理 {fname} 时出错: {e}', file=sys.stderr)

# 保存失败样本ID列表
failed_list = sorted(list(failed_ids), key=lambda x: int(x) if x.isdigit() else x)
with open('${LOG_DIR}/failed_sample_ids_${TIMESTAMP}.json', 'w') as f:
    json.dump(failed_list, f)

print(f'找到 {len(failed_list)} 个需要重新生成的样本')
print(f'样本ID: {failed_list}')
"

FAILED_IDS_FILE="${LOG_DIR}/failed_sample_ids_${TIMESTAMP}.json"

# GPU 0 - shard 0/2
echo "启动 GPU0 (shard 0/2)..."
nohup python -u regenerate_failures.py \
  --model_path data/checkpoints/step8_merged \
  --test_dir data/step6/Test \
  --output_dir $LOG_DIR \
  --device cuda:0 \
  --num_shards 2 \
  --shard_id 0 \
  --failed_sample_ids $FAILED_IDS_FILE \
  --initial_prompt_max_length $INITIAL_PROMPT_MAX_LENGTH \
  --initial_max_new_tokens $INITIAL_MAX_NEW_TOKENS \
  --strategy_prompt_max_length $STRATEGY_PROMPT_MAX_LENGTH \
  > $LOG_DIR/regen_${TIMESTAMP}_shard0of2_gpu0.log 2>&1 &

# GPU 1 - shard 1/2
echo "启动 GPU1 (shard 1/2)..."
nohup python -u regenerate_failures.py \
  --model_path data/checkpoints/step8_merged \
  --test_dir data/step6/Test \
  --output_dir $LOG_DIR \
  --device cuda:1 \
  --num_shards 2 \
  --shard_id 1 \
  --failed_sample_ids $FAILED_IDS_FILE \
  --initial_prompt_max_length $INITIAL_PROMPT_MAX_LENGTH \
  --initial_max_new_tokens $INITIAL_MAX_NEW_TOKENS \
  --strategy_prompt_max_length $STRATEGY_PROMPT_MAX_LENGTH \
  > $LOG_DIR/regen_${TIMESTAMP}_shard1of2_gpu1.log 2>&1 &

echo "已启动两个任务:"
echo "  GPU0: $LOG_DIR/regen_${TIMESTAMP}_shard0of2_gpu0.log"
echo "  GPU1: $LOG_DIR/regen_${TIMESTAMP}_shard1of2_gpu1.log"
echo ""
echo "查看运行状态: tail -f $LOG_DIR/regen_${TIMESTAMP}_shard0of2_gpu0.log"
echo "查看GPU进程: nvidia-smi"
