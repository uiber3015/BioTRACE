#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量逆向匹配脚本：
- 对 unified_json_enriched 里的每个 JSON：
  - 使用 rows 的值，在 all_in 里的所有 step3 表格中寻找“值覆盖率最高”的 table_content
  - 不再设置 80% 阈值，始终选取覆盖率最高的候选
  - 将该 step3 的 table_content 和 context 写回 unified_json_enriched 对应文件
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

BASE_DIR = Path(__file__).resolve().parent
ENRICHED_DIR = BASE_DIR / "unified_json_enriched"
STEP3_DIR = BASE_DIR / "all_in"


def _normalize_text(s: str) -> str:
    """简单文本归一化：小写、压缩空白。"""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def compute_row_value_coverage(rows: List[Dict[str, Any]], table_content: str, min_len: int = 3) -> Tuple[float, int, int]:
    """
    反向检查：rows 里的值是否都能在 table_content 中出现（不能被截断）。
    返回 (coverage, total_values, matched_values)
    """
    if not isinstance(rows, list) or not rows or not table_content:
        return 0.0, 0, 0

    norm_table = _normalize_text(table_content)

    values: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for v in row.values():
            if v is None:
                continue
            # 展开简单的嵌套 dict（如 { "White": "52 (96.3)", "Black": "2 (3.7)" }）
            if isinstance(v, dict):
                for vv in v.values():
                    if vv is None:
                        continue
                    vs = str(vv).strip()
                    if len(vs) >= min_len:
                        values.append(vs)
                continue

            vs = str(v).strip()
            if len(vs) < min_len:
                continue
            values.append(vs)

    if not values:
        return 0.0, 0, 0

    total = len(values)
    matched = 0

    for vs in values:
        n_vs = _normalize_text(vs)
        if n_vs and n_vs in norm_table:
            matched += 1

    coverage = matched / total if total > 0 else 0.0
    return coverage, total, matched


def build_step3_index():
    """预加载 all_in 里的 step3 表格内容与上下文，供匹配使用。"""
    index = []
    for path in sorted(
        STEP3_DIR.glob("*.json"),
        key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem),
    ):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        parsed = data.get("parsed_content") or {}
        table_content = (parsed.get("table_content") or "").strip()
        if not table_content:
            continue

        # 与 enrich_unified_json.py 保持一致：
        # 仅从 references 里提取 content，拼接成 context（不包含 caption / mentioned_in_text）
        refs = parsed.get("references", []) or []
        context_parts = []
        for r in refs:
            if isinstance(r, dict):
                c = (r.get("content") or "").strip()
                if c:
                    context_parts.append(c)
        context = "\n\n".join(context_parts)
        index.append(
            {
                "file_id": path.stem,
                "table_content": table_content,
                "context": context,
            }
        )
    return index


def find_best_step3_for_rows(rows: List[Dict[str, Any]], step3_index) -> Dict[str, Any]:
    """在所有 step3 表中，找到对给定 rows 值覆盖率最高的那个（始终返回最高覆盖）。"""
    best = None
    best_cov = -1.0
    for item in step3_index:
        cov, total, matched = compute_row_value_coverage(rows, item["table_content"])
        if cov > best_cov:
            best_cov = cov
            best = {
                "step3_file_id": item["file_id"],
                "coverage": cov,
                "total_values": total,
                "matched_values": matched,
                "table_content": item["table_content"],
                "context": item["context"],
            }
    return best or {}


def main():
    print("📚 正在构建 step3/all_in 索引（可能需要几十秒）...")
    if not STEP3_DIR.exists():
        print(f"❌ step3/all_in 目录不存在: {STEP3_DIR}")
        return
    step3_index = build_step3_index()
    print(f"  共载入 {len(step3_index)} 个带 table_content 的 step3 表")

    if not ENRICHED_DIR.exists():
        print(f"❌ unified_json_enriched 目录不存在: {ENRICHED_DIR}")
        return

    enriched_files = sorted(
        ENRICHED_DIR.glob("*.json"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    total = len(enriched_files)
    matched = 0

    for i, path in enumerate(enriched_files, 1):
        fid = path.stem
        try:
            with path.open("r", encoding="utf-8") as f:
                enriched = json.load(f)
        except Exception as e:
            print(f"[{fid}] 读取 enriched 失败: {e}")
            continue

        rows = enriched.get("rows") or enriched.get("json_data")
        if not isinstance(rows, list) or not rows:
            # 没有有效 rows，清空相关字段
            enriched["context"] = ""
            enriched["table_content"] = ""
            enriched["matched_step3_file_id"] = ""
            enriched["match_method"] = "none_no_rows"
            with path.open("w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False, indent=2)
            continue

        best = find_best_step3_for_rows(rows, step3_index)
        cov = best.get("coverage", 0.0)
        enriched["context"] = best.get("context", "")
        enriched["table_content"] = best.get("table_content", "")
        enriched["matched_step3_file_id"] = best.get("step3_file_id", "")
        enriched["match_method"] = "reverse_rows_value"
        enriched["value_coverage"] = cov

        with path.open("w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        matched += 1

        if i % 500 == 0 or i == total:
            print(f"  进度: {i}/{total}，已更新 {matched} 个文件，当前覆盖率={cov:.3f} (fid={fid})")

    print("\n✅ 逆向匹配完成")
    print(f"  总文件数: {total}")
    print(f"  已尝试匹配: {matched} 个（有 rows 的文件）")
    print(f"  结果已就地写回目录: {ENRICHED_DIR}")


if __name__ == "__main__":
    main()


