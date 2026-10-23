#!/bin/bash
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INPUT_ROOT="${TABLE_IMAGE_DIR:-$ROOT/data/step1/output}"

cd "$ROOT"
python pipeline/step1/convert_tables_to_markdown.py \
    --input-root "$INPUT_ROOT" \
    --use-cuda \
    --gpu-id 0
