#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本：为统一 JSON 加上 context 和 table_content 字段
匹配策略：
1. 首先使用 table_title 精确匹配 step3 的 caption（精确匹配）
2. 如果精确匹配失败，使用 rows 的 key/value 信息匹配 step3 的 table_content（基于内容的匹配）
3. 添加数据一致性验证，避免 rows 和 table_content 不匹配
"""

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 复用表格规范化与覆盖率计算逻辑
from normalize_and_compare import normalize_table_content, compare_rows  # type: ignore

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("enrich_json")

# 路径配置 - 使用相对路径，基于脚本所在目录
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 可配置路径（优先使用本地目录）
unified_input_dir = SCRIPT_DIR / "unified_json_input"
default_unified_dir = SCRIPT_DIR / "unified_json_enriched"
if unified_input_dir.exists():
    unified_dir = unified_input_dir
else:
    unified_dir = default_unified_dir

step3_local_dir = SCRIPT_DIR / "all_in"
default_step3_dir = PROJECT_ROOT / "step3" / "step3" / "output" / "all_in"
if step3_local_dir.exists():
    step3_all_in_dir = step3_local_dir
else:
    step3_all_in_dir = default_step3_dir

output_dir = SCRIPT_DIR / "unified_json_enriched"

# 如果 step3_all_in_dir 不存在，尝试其他可能的位置
if not step3_all_in_dir.exists():
    # 尝试在 step5 目录下查找 step3_data 文件夹
    alt_step3_dir = SCRIPT_DIR / "step3_data"
    if alt_step3_dir.exists():
        step3_all_in_dir = alt_step3_dir
        logger.info(f"使用备用 step3 目录: {step3_all_in_dir}")
    else:
        logger.warning(f"⚠️ Step3 目录不存在: {step3_all_in_dir}")
        logger.warning(f"   请将 step3 的原始数据放到: {alt_step3_dir} 或修改脚本中的路径配置")

output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🚀 统一 JSON 增强脚本（双重匹配策略 + 数据一致性验证）")
print("=" * 80)
print(f"📂 统一 JSON 输入目录: {unified_dir}")
print(f"📂 Step3 原始数据目录: {step3_all_in_dir}")
print(f"📂 输出目录: {output_dir}")

# 1. 构建索引
print("\n📚 构建 step3 索引...")
print("   策略1: caption → {context, table_content} 索引（用于精确匹配）")
print("   策略2: 完整文件列表（用于基于 rows 的匹配）")

# 索引1：caption 精确匹配索引
caption_index: Dict[str, Dict[str, Any]] = {}
# 索引2：所有 step3 文件的完整信息列表（用于基于 rows 的匹配）
step3_files: List[Dict[str, Any]] = []

if not step3_all_in_dir.exists():
    print(f"⚠️ Step3 目录不存在: {step3_all_in_dir}")
    print("   将跳过匹配，所有文件的 table_content 和 context 将为空")
    step3_files = []
else:
    files = sorted(step3_all_in_dir.glob("*.json"), key=lambda p: p.stem)
    print(f"   扫描 {len(files)} 个 step3 JSON 文件")

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            parsed = data.get("parsed_content", {}) or {}
            
            caption = parsed.get("caption", "").strip()
            table_content = parsed.get("table_content", "").strip()
            
            # 从 references 里提取所有 content，拼接成上下文
            references = parsed.get("references", []) or []
            context_parts = []
            for ref in references:
                if isinstance(ref, dict):
                    content = ref.get("content", "").strip()
                    if content:
                        context_parts.append(content)
            
            context = "\n\n".join(context_parts)
            
            # 构建 caption 索引（用于精确匹配）
            if caption and caption not in caption_index:
                caption_index[caption] = {
                    "context": context,
                    "table_content": table_content,
                    "file_id": path.stem
                }
            
            # 构建完整文件列表（用于基于 rows 的匹配）
            if table_content:
                step3_files.append({
                    "file_id": path.stem,
                    "caption": caption,
                    "context": context,
                    "table_content": table_content,
                    "table_content_lower": table_content.lower()
                })
        except Exception as e:
            logger.warning(f"读取 step3 文件失败 {path.name}: {e}")
            pass

print(f"✅ 索引构建完成")
print(f"   - caption 索引: {len(caption_index)} 个")
print(f"   - step3 文件列表: {len(step3_files)} 个")

# 2. 定义数据一致性验证函数
def validate_data_consistency(rows: List[Dict], table_content: str, threshold: float = 0.3) -> bool:
    """
    验证 rows 和 table_content 是否一致
    
    Args:
        rows: 统一 JSON 的 rows 数据
        table_content: step3 的 table_content
        threshold: 匹配阈值（默认 0.3，即至少30%的key匹配）
    
    Returns:
        True 如果一致，False 如果不一致
    """
    if not rows or not table_content:
        return True  # 如果缺少一个，认为一致（可能是无数值字段的情况）
    
    # 提取 rows 的 key
    rows_keys = set()
    for row in rows[:5]:  # 检查前5行
        if isinstance(row, dict):
            rows_keys.update(k.lower() for k in row.keys())
    
    if not rows_keys:
        return True
    
    # 检查 table_content 中是否包含这些 key
    table_lower = table_content.lower()
    matched = sum(1 for key in rows_keys if key in table_lower)
    
    match_ratio = matched / len(rows_keys) if rows_keys else 0
    
    # 如果匹配度低于阈值，认为不一致
    if match_ratio < threshold:
        return False
    
    return True

# 3. 定义基于 rows 的匹配函数
def extract_row_keys(rows: List[Dict]) -> List[str]:
    """从 rows 中提取所有的 key"""
    if not isinstance(rows, list) or not rows:
        return []
    
    keys = set()
    for row in rows[:5]:  # 只看前 5 行
        if isinstance(row, dict):
            keys.update(row.keys())
    
    return list(keys)


def match_by_rows(row_keys: List[str], row_values: List[str], rows: List[Dict]) -> Optional[Dict[str, Any]]:
    """
    使用 rows 的 key/value 信息匹配 step3 的原始表格
    策略：
    1. 先用 key 匹配（检查 table_content 是否包含这些 key）
    2. 如果 key 匹配失败，用 value 匹配
    3. 验证数据一致性
    """
    if not row_keys and not row_values:
        return None
    
    best_match = None
    best_score = 0
    
    # 策略 1：用 key 匹配（优先）
    for step3_data in step3_files:
        full_content = step3_data["table_content_lower"]
        
        # 检查有多少个 row_key 在 table_content 里
        matched_keys = 0
        for row_key in row_keys:
            if row_key.lower() in full_content:
                matched_keys += 1
        
        # 计算匹配分数
        if len(row_keys) > 0:
            score = matched_keys / len(row_keys)
            # 如果至少 80% 的 key 都匹配上了，且数据一致性验证通过
            if score >= 0.8:
                # 验证数据一致性
                if validate_data_consistency(rows, step3_data["table_content"], threshold=0.3):
                    return {
                        "context": step3_data["context"],
                        "table_content": step3_data["table_content"],
                        "file_id": step3_data["file_id"],
                        "match_method": "rows_key",
                        "match_score": score
                    }
                elif score > best_score:
                    best_match = {
                        "context": step3_data["context"],
                        "table_content": step3_data["table_content"],
                        "file_id": step3_data["file_id"],
                        "match_method": "rows_key",
                        "match_score": score
                    }
                    best_score = score
    
    # 策略 2：用 value 匹配（备选，但需要验证一致性）
    for row_value in row_values[:10]:  # 只用前 10 个 value
        if not row_value or len(str(row_value)) < 3:
            continue
        
        value_lower = str(row_value).lower()
        
        for step3_data in step3_files:
            full_content = step3_data["table_content_lower"]
            
            # 如果 value 在 table_content 里出现了
            if value_lower in full_content:
                # 验证数据一致性
                if validate_data_consistency(rows, step3_data["table_content"], threshold=0.3):
                    return {
                        "context": step3_data["context"],
                        "table_content": step3_data["table_content"],
                        "file_id": step3_data["file_id"],
                        "match_method": "rows_value"
                    }
    
    # 如果没有找到一致性验证通过的匹配，返回最佳匹配（但标记为不一致）
    if best_match and best_score >= 0.5:
        logger.warning(f"找到匹配但数据一致性较低 (score={best_score:.2f}): {best_match['file_id']}")
        return best_match
    
    return None


# 4. 处理统一 JSON
print("\n" + "=" * 80)
print("🔄 处理统一 JSON 文件")
print("=" * 80)
sys.stdout.flush()

if not unified_dir.exists():
    logger.error(f"❌ 统一 JSON 输入目录不存在: {unified_dir}")
    sys.exit(1)

unified_files = sorted(unified_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
print(f"📄 待处理文件数: {len(unified_files)}")
sys.stdout.flush()

success_count = 0
matched_by_title = 0
matched_by_rows = 0
not_matched = 0
mismatched = 0  # 数据不一致的匹配
failed_count = 0
start_time = time.time()


def log_progress(prefix: str = "进度快照"):
    """随机/阶段性打印进度信息，便于 nohup 查看"""
    elapsed = max(time.time() - start_time, 1e-6)
    speed = success_count / elapsed
    remaining = len(unified_files) - success_count
    eta = remaining / speed if speed > 0 else float("inf")
    logger.info(
        f"{prefix}: {success_count}/{len(unified_files)} | "
        f"title={matched_by_title}, rows={matched_by_rows}, none={not_matched}, mismatch={mismatched}, "
        f"speed={speed:.2f}/s, ETA≈{eta/60:.1f}min"
    )
    sys.stdout.flush()

for file_path in unified_files:
    file_id = file_path.stem
    
    try:
        # 读取统一 JSON
        with file_path.open("r", encoding="utf-8") as f:
            unified_data = json.load(f)
        
        table_title = (unified_data.get("table_title") or "").strip()
        rows = unified_data.get("rows") or unified_data.get("json_data")
        
        step3_data = None
        match_method = None
        is_mismatched = False
        
        # 策略1: 精确匹配（table_title → caption）
        if table_title and table_title in caption_index:
            step3_data = caption_index[table_title].copy()
            match_method = "title_exact"
            
            # 验证数据一致性
            if not validate_data_consistency(rows, step3_data.get("table_content", ""), threshold=0.3):
                logger.warning(f"文件 {file_id}: title 匹配但数据不一致，将标记为 mismatch")
                is_mismatched = True
                mismatched += 1
            else:
                matched_by_title += 1
        else:
            # 策略2: 基于 rows 的匹配
            if isinstance(rows, list) and rows:
                row_keys = extract_row_keys(rows)
                row_values = []
                for row in rows[:5]:
                    if isinstance(row, dict):
                        row_values.extend([str(v) for v in row.values() if v is not None])
                
                step3_data = match_by_rows(row_keys, row_values, rows)
                if step3_data:
                    match_method = step3_data.pop("match_method", "rows")
                    match_score = step3_data.pop("match_score", 1.0)
                    
                    # 检查是否是不一致匹配
                    if match_score < 0.8 or not validate_data_consistency(rows, step3_data.get("table_content", ""), threshold=0.3):
                        is_mismatched = True
                        mismatched += 1
                    else:
                        matched_by_rows += 1

        # 使用覆盖率做最终筛选：只接受列覆盖率 >= 0.8 的匹配
        if step3_data and isinstance(rows, list) and rows:
            try:
                normalized = normalize_table_content(step3_data.get("table_content", ""))
                coverage = compare_rows(rows, normalized)
                col_cov = coverage.get("column_coverage", 0.0)
                row_cov = coverage.get("row_coverage", 0.0)
                if col_cov < 0.8:
                    logger.info(
                        "文件 %s 与 step3 文件 %s 覆盖率过低 (col=%.3f,row=%.3f)，丢弃匹配",
                        file_id,
                        step3_data.get("file_id", ""),
                        col_cov,
                        row_cov,
                    )
                    step3_data = None
                    match_method = "none_low_coverage"
                    not_matched += 1
            except Exception as e:
                logger.warning("文件 %s 计算覆盖率失败，将视为未匹配: %s", file_id, e)
                step3_data = None
                match_method = "none_coverage_error"
                not_matched += 1

        # 添加匹配到的信息（通过覆盖率筛选后的最终结果）
        if step3_data:
            unified_data["context"] = step3_data.get("context", "")
            unified_data["table_content"] = step3_data.get("table_content", "")
            unified_data["matched_step3_file_id"] = step3_data.get("file_id", "")
            unified_data["match_method"] = match_method
            if is_mismatched:
                unified_data["match_method"] = f"{match_method}_mismatch"
                unified_data["data_consistency_warning"] = True
        else:
            # 如果没有匹配或被覆盖率筛掉，添加空字段
            unified_data["context"] = ""
            unified_data["table_content"] = ""
            unified_data["matched_step3_file_id"] = ""
            unified_data["match_method"] = match_method or "none"
        
        # 写出增强后的 JSON
        out_path = output_dir / file_path.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(unified_data, f, ensure_ascii=False, indent=2)
        
        success_count += 1
        
        # 每 1000 个文件打印一次进度，同时随机输出快照方便 nohup 观察
        if success_count % 1000 == 0:
            log_progress("阶段进度")
        elif random.random() < 0.01:
            log_progress("随机进度")
    
    except Exception as e:
        logger.error(f"❌ 处理文件 {file_path.name} 失败: {e}")
        failed_count += 1

# 4. 统计
print("\n" + "=" * 80)
print("📊 处理统计")
print("=" * 80)
print(f"✅ 成功处理: {success_count} 个文件")
print(f"❌ 处理失败: {failed_count} 个文件")
print(f"\n匹配统计:")
total_matched = matched_by_title + matched_by_rows
match_rate = (total_matched / success_count * 100) if success_count > 0 else 0
print(f"  🔗 总匹配成功: {total_matched} 个文件（{match_rate:.1f}%）")
print(f"    - 精确匹配（table_title）: {matched_by_title} 个")
print(f"    - rows匹配: {matched_by_rows} 个")
print(f"  ⚠️  数据不一致匹配: {mismatched} 个文件（已标记为 mismatch）")
print(f"  ❌ 未匹配: {not_matched} 个文件（{not_matched/success_count*100:.1f}%）")
print(f"\n📁 输出目录: {output_dir}")
print("=" * 80)