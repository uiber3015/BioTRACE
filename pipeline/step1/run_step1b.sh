#!/bin/bash
# 步骤 1B：用 RapidTable 把表格图片转成 Markdown

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT_ROOT="${TABLE_IMAGE_DIR:-$ROOT/data/step1/output}"

echo "========================================"
echo "步骤 1B: 表格图片转 Markdown"
echo "========================================"

if [ ! -d "$INPUT_ROOT/table_images" ]; then
    echo "未找到表格图片目录: $INPUT_ROOT/table_images"
    echo "请先运行步骤 1A，或设置 TABLE_IMAGE_DIR"
    exit 1
fi

if [ ! -d "$INPUT_ROOT/table_captions" ]; then
    echo "未找到 caption 目录: $INPUT_ROOT/table_captions"
    exit 1
fi

python -c "from rapid_table import RapidTable" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "未安装 rapid_table。请先执行: pip install rapid_table html2text"
    exit 1
fi

cd "$ROOT"
read -p "是否使用 GPU 加速? (y/N): " use_gpu

if [[ "$use_gpu" =~ ^[Yy]$ ]]; then
    read -p "GPU ID (默认 0): " gpu_id
    gpu_id=${gpu_id:-0}
    python pipeline/step1/convert_tables_to_markdown.py \
        --input-root "$INPUT_ROOT" \
        --use-cuda \
        --gpu-id "$gpu_id"
else
    python pipeline/step1/convert_tables_to_markdown.py \
        --input-root "$INPUT_ROOT"
fi
