"""
最终 R_rule 评测器 (Final v3.0)

最终算法：
1. 精确匹配（匹配数/总数），不是Jaccard
2. 归一化：大小写归一、多余空格删除、只保留字母和数字
3. 匈牙利算法进行双向最优匹配
4. 预测链多 → 按预测链数目平均（惩罚多预测的链）
5. 预测链少 → 按真实链数目平均（惩罚漏预测的链）
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Set, Tuple, Any

PARSE_OK = "OK"
PARSE_MULTI = "MULTI_RESULT"
PARSE_ERROR = "PARSE_ERROR"
PARSE_EMPTY = "EMPTY"


def _normalize_text(text: str) -> str:
    """温和归一化：删除空格 + 字母小写 + "-"变为"_"（其他符号保持不变）"""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', '', text)  # 删除所有空格
    text = text.replace('-', '_')  # "-"变为"_"
    return text


def _ensure_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return None


def _normalize_chain(chain: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize chain to flat fields: condition/result/entity_recognition."""
    if not isinstance(chain, dict):
        return {"condition": None, "result": None, "entity_recognition": None}
    base = chain.get("knowledge_chain") if isinstance(chain.get("knowledge_chain"), dict) else chain
    entity_recognition = chain.get("entity_recognition")
    if entity_recognition is None and isinstance(base, dict):
        entity_recognition = base.get("entity_recognition")
    return {
        "condition": base.get("condition") if isinstance(base, dict) else None,
        "result": base.get("result") if isinstance(base, dict) else None,
        "entity_recognition": entity_recognition,
    }


def split_condition_to_kv_pairs(condition_str: Optional[str]) -> Tuple[Optional[Set[Tuple[str, str]]], Optional[str]]:
    """
    Parse condition_str:
    - split by '|' into segments
    - for each non-empty seg: strip() then split by FIRST ':' into key,value
    - strip key/value only
    - return set of (key,value)
    """
    s = _ensure_str(condition_str)
    if s is None:
        return None, PARSE_ERROR
    parts = [seg.strip() for seg in s.split("|")]
    pairs: Set[Tuple[str, str]] = set()
    for seg in parts:
        if not seg:
            continue
        if ":" not in seg:
            return None, PARSE_ERROR
        key, val = seg.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key == "" and val == "":
            continue
        pairs.add((key, val))
    if len(pairs) == 0:
        return set(), PARSE_ERROR
    return pairs, PARSE_OK


def split_result_to_kv(result_str: Optional[str]) -> Tuple[str, Optional[Tuple[str, str]]]:
    """
    Return one of:
     - (PARSE_EMPTY, None)  -> missing/None/empty
     - (PARSE_OK, (k,v))    -> single kv parsed
     - (PARSE_MULTI, None)  -> multiple kvs detected
     - (PARSE_ERROR, None)  -> parse error
    """
    s = _ensure_str(result_str)
    if s is None:
        return PARSE_EMPTY, None
    if s.strip() == "":
        return PARSE_EMPTY, None
    if "|" in s:
        return PARSE_MULTI, None
    if ":" not in s:
        return PARSE_ERROR, None
    key, val = s.split(":", 1)
    if ";" in val:
        return PARSE_MULTI, None
    if ":" in val:
        possible_next_key = val.split(":", 1)[0].strip()
        if len(possible_next_key) <= 40 and " " not in possible_next_key:
            return PARSE_MULTI, None
    return PARSE_OK, (key.strip(), val.strip())


def extract_entity_pairs(entity_obj: Any) -> Tuple[Optional[Set[Tuple[str, str]]], Optional[str]]:
    """
    Accepts dict mapping type -> list of values.
    Returns (set((type,value)), PARSE_OK) or (None, PARSE_ERROR).
    Empty dict -> treated as EMPTY.
    """
    if entity_obj is None:
        return None, PARSE_EMPTY
    if not isinstance(entity_obj, dict):
        return None, PARSE_ERROR
    if len(entity_obj) == 0:
        return set(), PARSE_EMPTY
    pairs: Set[Tuple[str, str]] = set()
    for t, vals in entity_obj.items():
        if not isinstance(vals, list):
            return None, PARSE_ERROR
        type_str = _ensure_str(t)
        if type_str is None:
            return None, PARSE_ERROR
        for v in vals:
            vstr = _ensure_str(v)
            if vstr is None:
                continue
            pairs.add((type_str.strip(), vstr.strip()))
    return pairs, PARSE_OK


def exact_match_pairs(A: Optional[Set[Tuple[str, str]]], B: Optional[Set[Tuple[str, str]]]) -> float:
    """
    精确匹配分数：匹配数 / 总数（以B为基准）
    - 使用归一化后的key和value进行匹配
    """
    if A is None or B is None:
        return 0.0
    
    if len(A) == 0 and len(B) == 0:
        return 1.0  # 两个都为空算完全匹配
    
    if len(B) == 0:
        return 0.0
    
    # 归一化所有key和value
    A_norm = {(_normalize_text(k), _normalize_text(v)) for k, v in A}
    B_norm = {(_normalize_text(k), _normalize_text(v)) for k, v in B}
    
    # 计算交集
    intersection = A_norm & B_norm
    
    # 以B的数目为基准计算匹配比例
    return len(intersection) / len(B_norm)


def exact_match_result(pred_result: Optional[Tuple[str, str]], gold_result: Optional[Tuple[str, str]], 
                       pred_status: str, gold_status: str) -> float:
    """
    Result精确匹配分数：
    - 如果gold_result为空且pred_result也为空，得1分
    - 否则完全匹配（归一化后）得1分，否则得0分
    """
    # 两个都为空
    if gold_status == PARSE_EMPTY and pred_status == PARSE_EMPTY:
        return 1.0
    
    # 两个都成功解析
    if gold_status == PARSE_OK and pred_status == PARSE_OK and gold_result and pred_result:
        gold_norm = (_normalize_text(gold_result[0]), _normalize_text(gold_result[1]))
        pred_norm = (_normalize_text(pred_result[0]), _normalize_text(pred_result[1]))
        return 1.0 if gold_norm == pred_norm else 0.0
    
    return 0.0


def compute_chain_match_score(pred_chain: Dict[str, Any], gold_chain: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算预测链与真实链的匹配分数
    返回：cond_match, result_match, entity_match, total_match
    """
    pred_norm = _normalize_chain(pred_chain)
    gold_norm = _normalize_chain(gold_chain)
    
    # Condition匹配
    pred_cond_pairs, pred_cond_status = split_condition_to_kv_pairs(pred_norm.get("condition"))
    gold_cond_pairs, gold_cond_status = split_condition_to_kv_pairs(gold_norm.get("condition"))
    cond_match = exact_match_pairs(pred_cond_pairs, gold_cond_pairs)
    
    # Result匹配
    pred_result_status, pred_result_kv = split_result_to_kv(pred_norm.get("result"))
    gold_result_status, gold_result_kv = split_result_to_kv(gold_norm.get("result"))
    result_match = exact_match_result(pred_result_kv, gold_result_kv, pred_result_status, gold_result_status)
    
    # Entity匹配
    pred_entity_pairs, pred_entity_status = extract_entity_pairs(pred_norm.get("entity_recognition"))
    gold_entity_pairs, gold_entity_status = extract_entity_pairs(gold_norm.get("entity_recognition"))
    entity_match = exact_match_pairs(pred_entity_pairs, gold_entity_pairs)
    
    total_match = (cond_match + result_match + entity_match) / 3.0
    
    return {
        "cond_match": cond_match,
        "result_match": result_match,
        "entity_match": entity_match,
        "total_match": total_match,
    }


def hungarian_algorithm(scores: List[List[float]]) -> List[Tuple[int, int]]:
    """
    匈牙利算法进行最优匹配
    scores[i][j] = 预测链i与真实链j的匹配分数
    返回：匹配对列表 [(pred_idx, gold_idx), ...]
    """
    import numpy as np
    
    n_pred = len(scores)
    n_gold = len(scores[0]) if scores else 0
    
    if n_pred == 0 or n_gold == 0:
        return []
    
    # 创建成本矩阵（分数取负值，匈牙利算法最小化成本）
    max_n = max(n_pred, n_gold)
    cost_matrix = np.zeros((max_n, max_n))
    
    for i in range(n_pred):
        for j in range(n_gold):
            # 分数越高，成本越低
            cost_matrix[i][j] = -scores[i][j]
    
    # 未匹配的位置设为0成本
    cost_matrix[n_pred:, :] = 0
    cost_matrix[:, n_gold:] = 0
    
    # 使用scipy的线性分配算法
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        for i, j in zip(row_ind, col_ind):
            if i < n_pred and j < n_gold:
                matches.append((i, j))
        return matches
    except ImportError:
        # 如果没有scipy，使用贪婪算法
        matches = []
        used_pred = set()
        used_gold = set()
        
        # 按分数从高到低排序
        all_pairs = []
        for i in range(n_pred):
            for j in range(n_gold):
                all_pairs.append((scores[i][j], i, j))
        all_pairs.sort(reverse=True)
        
        for score, i, j in all_pairs:
            if i not in used_pred and j not in used_gold:
                matches.append((i, j))
                used_pred.add(i)
                used_gold.add(j)
        
        return matches


def compute_per_chain_structure(pred_chain: Dict[str, Any]) -> Dict[str, Any]:
    """
    计算预测链的结构分：
    - cond_format: 1 if condition格式正确（有正确的KV对）
    - result_format: 1 if result格式正确或为空
    - entity_format: 1 if entity格式正确或为空
    - chain_structure: 三个格式都对得1分，否则0分
    """
    pred_norm = _normalize_chain(pred_chain)
    
    # Condition
    cond_pairs, cond_status = split_condition_to_kv_pairs(pred_norm.get("condition"))
    cond_format = 1 if (cond_status == PARSE_OK and cond_pairs and len(cond_pairs) >= 1) else 0
    
    # Result
    result_status, result_kv = split_result_to_kv(pred_norm.get("result"))
    result_format = 1 if (result_status == PARSE_EMPTY or result_status == PARSE_OK) else 0
    
    # Entity
    ent_pairs, ent_status = extract_entity_pairs(pred_norm.get("entity_recognition"))
    entity_format = 1 if (ent_status in (PARSE_EMPTY, PARSE_OK)) else 0
    
    # 三个都对才算结构正确
    chain_structure = 1 if (cond_format == 1 and result_format == 1 and entity_format == 1) else 0
    
    return {
        "cond_format": cond_format,
        "result_format": result_format,
        "entity_format": entity_format,
        "chain_structure": chain_structure,
    }


def compute_r_rule_final(sample: Dict[str, Any], generated_key: str = "generated_json") -> Dict[str, Any]:
    """
    计算最终R_rule (v3.0)
    
    核心逻辑：
    - 预测链多 → 按预测链数目平均（惩罚多预测的链）
    - 预测链少 → 按真实链数目平均（惩罚漏预测的链）
    
    Args:
        sample: 包含 ground_truth 和 generated_json 的字典
        generated_key: generated_json的键名（默认为"generated_json"）
    
    Returns:
        包含 R_rule, S_structure, S_content, per_chain_debug 等字段的字典
    """
    out = {
        "R_rule": 0.0,
        "S_structure": 0.0,
        "S_content": 0.0,
        "per_chain_debug": [],
        "match_pairs": [],
        "strategy": "",
        "n_pred": 0,
        "n_gold": 0,
        "valid": False,
    }
    
    # 解析生成的JSON
    gen = sample.get(generated_key)
    if gen is None:
        return out
    if isinstance(gen, str):
        try:
            gen = json.loads(gen)
        except Exception:
            return out
    
    # 解析ground_truth
    gold = sample.get("ground_truth") or {}
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except Exception:
            gold = {}
    
    # 获取knowledge chains
    gen_chains = gen.get("knowledge_mining", []) if isinstance(gen, dict) else []
    gold_chains = gold.get("knowledge_mining", []) if isinstance(gold, dict) else []
    
    n_pred = len(gen_chains)
    n_gold = len(gold_chains)
    
    out["n_pred"] = n_pred
    out["n_gold"] = n_gold
    
    # 如果没有真实链，返回0
    if n_gold == 0:
        return out
    
    # 计算所有预测链与真实链的匹配分数矩阵
    if n_pred > 0:
        score_matrix = []
        for pred in gen_chains:
            row = []
            for gold_c in gold_chains:
                match_result = compute_chain_match_score(pred, gold_c)
                row.append(match_result["total_match"])
            score_matrix.append(row)
        
        # 使用匈牙利算法找最优匹配
        match_pairs = hungarian_algorithm(score_matrix)
    else:
        match_pairs = []
    
    # 标记哪些预测链和真实链已被匹配
    matched_pred = set(i for i, _ in match_pairs)
    matched_gold = set(j for _, j in match_pairs)
    
    # 计算每个预测链的分数
    pred_scores = []
    for i, pred in enumerate(gen_chains):
        # 结构分
        struct = compute_per_chain_structure(pred)
        
        # 内容分
        if i in matched_pred:
            # 找到匹配的gold链
            matched_gold_idx = None
            for pred_i, gold_j in match_pairs:
                if pred_i == i:
                    matched_gold_idx = gold_j
                    break
            
            if matched_gold_idx is not None:
                match_result = compute_chain_match_score(pred, gold_chains[matched_gold_idx])
                chain_content = (match_result["cond_match"] + 
                               match_result["result_match"] + 
                               match_result["entity_match"]) / 3.0
            else:
                chain_content = 0.0
        else:
            chain_content = 0.0
        
        pred_scores.append({
            "pred_idx": i,
            "structure": struct["chain_structure"],
            "content": chain_content,
            "cond_format": struct["cond_format"],
            "result_format": struct["result_format"],
            "entity_format": struct["entity_format"],
            "matched": i in matched_pred,
        })
    
    # 计算每个真实链的分数
    gold_scores = []
    for j, gold_c in enumerate(gold_chains):
        # 结构分（真实链的结构分基于它本身的格式是否正确）
        gold_norm = _normalize_chain(gold_c)
        cond_pairs, cond_status = split_condition_to_kv_pairs(gold_norm.get("condition"))
        result_status, _ = split_result_to_kv(gold_norm.get("result"))
        ent_status, _ = extract_entity_pairs(gold_norm.get("entity_recognition"))
        
        cond_format = 1 if (cond_status == PARSE_OK and cond_pairs) else 0
        result_format = 1 if (result_status == PARSE_EMPTY or result_status == PARSE_OK) else 0
        entity_format = 1 if (ent_status in (PARSE_EMPTY, PARSE_OK)) else 0
        chain_structure = 1 if (cond_format == 1 and result_format == 1 and entity_format == 1) else 0
        
        # 内容分（基于匹配的预测链）
        if j in matched_gold:
            matched_pred_idx = None
            for pred_i, gold_j in match_pairs:
                if gold_j == j:
                    matched_pred_idx = pred_i
                    break
            
            if matched_pred_idx is not None:
                match_result = compute_chain_match_score(gen_chains[matched_pred_idx], gold_c)
                chain_content = (match_result["cond_match"] + 
                               match_result["result_match"] + 
                               match_result["entity_match"]) / 3.0
            else:
                chain_content = 0.0
        else:
            chain_content = 0.0
        
        gold_scores.append({
            "gold_idx": j,
            "structure": chain_structure,
            "content": chain_content,
        })
    
    # 根据链的数量选择汇总策略
    if n_pred >= n_gold:
        # 预测链多，按预测链数目平均
        strategy = f"pred({n_pred}) >= gold({n_gold}), 按{n_pred}条预测链平均"
        S_structure = sum(s["structure"] for s in pred_scores) / n_pred if n_pred > 0 else 0.0
        S_content = sum(s["content"] for s in pred_scores) / n_pred if n_pred > 0 else 0.0
    else:
        # 预测链少，按真实链数目平均
        strategy = f"pred({n_pred}) < gold({n_gold}), 按{n_gold}条真实链平均"
        S_structure = sum(s["structure"] for s in gold_scores) / n_gold if n_gold > 0 else 0.0
        S_content = sum(s["content"] for s in gold_scores) / n_gold if n_gold > 0 else 0.0
    
    R_rule = (S_structure + S_content) / 2.0
    
    # 构建per_chain_debug
    for i, pred in enumerate(gen_chains):
        match_result = None
        if i in matched_pred:
            for pred_i, gold_j in match_pairs:
                if pred_i == i:
                    match_result = compute_chain_match_score(pred, gold_chains[gold_j])
                    break
        
        debug = {
            "pred_idx": i,
            "matched": i in matched_pred,
            "cond_format": pred_scores[i]["cond_format"],
            "result_format": pred_scores[i]["result_format"],
            "entity_format": pred_scores[i]["entity_format"],
            "structure": pred_scores[i]["structure"],
            "cond_match": match_result["cond_match"] if match_result else 0.0,
            "result_match": match_result["result_match"] if match_result else 0.0,
            "entity_match": match_result["entity_match"] if match_result else 0.0,
            "content": pred_scores[i]["content"],
        }
        out["per_chain_debug"].append(debug)
    
    out["R_rule"] = float(R_rule)
    out["S_structure"] = float(S_structure)
    out["S_content"] = float(S_content)
    out["match_pairs"] = match_pairs
    out["strategy"] = strategy
    out["valid"] = True
    
    return out


def compute_r_rule(sample: Dict[str, Any], generated_key: str = "generated_json") -> Dict[str, Any]:
    """兼容旧接口，默认使用最终算法"""
    return compute_r_rule_final(sample, generated_key)


def evaluate_file_final(file_path: str, generated_key: str = "generated_json", 
                        output_key: str = None) -> Dict[str, Any]:
    """
    评估整个文件的数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    results = []
    for sample in data:
        result = compute_r_rule_final(sample, generated_key)
        result["sample_id"] = sample.get("file_id") or sample.get("sample_id") or "unknown"
        results.append(result)
    
    # 汇总统计
    valid_results = [r for r in results if r.get("valid", False)]
    r_rules = [r["R_rule"] for r in valid_results]
    s_structs = [r["S_structure"] for r in valid_results]
    s_contents = [r["S_content"] for r in valid_results]
    
    # 统计pred和gold的分布
    pred_counts = [r["n_pred"] for r in valid_results]
    gold_counts = [r["n_gold"] for r in valid_results]
    
    stats = {
        "total_samples": int(len(results)),
        "valid_samples": int(len(valid_results)),
        "avg_R_rule": float(sum(r_rules) / len(r_rules)) if r_rules else 0.0,
        "avg_S_structure": float(sum(s_structs) / len(s_structs)) if s_structs else 0.0,
        "avg_S_content": float(sum(s_contents) / len(s_contents)) if s_contents else 0.0,
        "min_R_rule": float(min(r_rules)) if r_rules else 0.0,
        "max_R_rule": float(max(r_rules)) if r_rules else 0.0,
        "std_R_rule": float(_std(r_rules)) if len(r_rules) > 1 else 0.0,
        "avg_pred_chains": float(sum(pred_counts) / len(pred_counts)) if pred_counts else 0.0,
        "avg_gold_chains": float(sum(gold_counts) / len(gold_counts)) if gold_counts else 0.0,
        "pred_gt_gold_count": int(sum(1 for p, g in zip(pred_counts, gold_counts) if p >= g)),
        "pred_lt_gold_count": int(sum(1 for p, g in zip(pred_counts, gold_counts) if p < g)),
        "results": [_convert_to_native_types(r) for r in results],
    }
    
    return stats


def _std(values: List[float]) -> float:
    """计算标准差"""
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _convert_to_native_types(obj: Any) -> Any:
    """将numpy类型转换为Python原生类型"""
    import numpy as np
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _convert_to_native_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_native_types(item) for item in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def print_evaluation_report(stats: Dict[str, Any], title: str = "Evaluation Report"):
    """打印评估报告"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  Total Samples:      {stats['total_samples']}")
    print(f"  Valid Samples:      {stats['valid_samples']}")
    print("-" * 70)
    print(f"  Avg R_rule:         {stats['avg_R_rule']:.4f}")
    print(f"  Avg S_structure:    {stats['avg_S_structure']:.4f}")
    print(f"  Avg S_content:      {stats['avg_S_content']:.4f}")
    print("-" * 70)
    print(f"  Min R_rule:         {stats['min_R_rule']:.4f}")
    print(f"  Max R_rule:         {stats['max_R_rule']:.4f}")
    print(f"  Std R_rule:         {stats['std_R_rule']:.4f}")
    print("-" * 70)
    print(f"  Avg Pred Chains:    {stats['avg_pred_chains']:.2f}")
    print(f"  Avg Gold Chains:    {stats['avg_gold_chains']:.2f}")
    print(f"  Pred >= Gold:       {stats['pred_gt_gold_count']} samples")
    print(f"  Pred < Gold:        {stats['pred_lt_gold_count']} samples")
    print("=" * 70)
    return stats


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Final R-rule Evaluator v3.0")
    parser.add_argument("file", help="JSON file to evaluate")
    parser.add_argument("--key", default="generated_json", help="Key for generated JSON in sample")
    parser.add_argument("--output", help="Output file for results (JSON)")
    args = parser.parse_args()
    
    stats = evaluate_file_final(args.file, args.key)
    print_evaluation_report(stats, "Final R-rule Evaluation Results (v3.0)")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")

