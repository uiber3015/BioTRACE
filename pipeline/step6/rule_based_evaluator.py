#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based evaluator implementing two modes for chain matching:
 - strict: exact (key,value) equality, result must be single key:value pair
 - tolerant: allows minor formatting differences and numeric tolerance

Usage:
python step6/rule_based_evaluator.py --results_file step6/test_results/results_trained.json --mode strict --output step6/test_results/eval_strict.json
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional


KV_RE = re.compile(r'\s*\|\s*')


def split_condition_to_kv_pairs(cond: str) -> Set[Tuple[str, str]]:
    parts = [p.strip() for p in cond.split('|') if p.strip()]
    pairs = set()
    for part in parts:
        if ':' in part:
            k, v = part.split(':', 1)
            pairs.add((k.strip(), v.strip()))
        else:
            pairs.add((part.strip(), ""))  # key-only
    return pairs


def split_result_to_kv(result: str) -> List[Tuple[str, str]]:
    # result may be like "Indicator:value" or other; split on first colon only
    parts = [p.strip() for p in result.split('|') if p.strip()]
    kvs = []
    for part in parts:
        if ':' in part:
            k, v = part.split(':', 1)
            kvs.append((k.strip(), v.strip()))
        else:
            kvs.append((part.strip(), ""))
    return kvs


def extract_entity_pairs(entity_dict: Dict) -> Set[Tuple[str, str]]:
    pairs = set()
    if not isinstance(entity_dict, dict):
        return pairs
    for etype, values in entity_dict.items():
        if isinstance(values, list):
            for v in values:
                pairs.add((str(etype).strip(), str(v).strip()))
        elif isinstance(values, str):
            pairs.add((str(etype).strip(), values.strip()))
    return pairs


def normalize_text_tol(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def try_parse_number(s: str) -> Optional[float]:
    s_clean = s.strip()
    # remove commas
    s_clean = s_clean.replace(',', '')
    # find first number-like substring
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s_clean)
    if m:
        try:
            return float(m.group(0))
        except:
            return None
    return None


def compare_kv_strict(kv_gen: Tuple[str, str], kv_gt: Tuple[str, str]) -> bool:
    return kv_gen == kv_gt


def compare_kv_tolerant(kv_gen: Tuple[str, str], kv_gt: Tuple[str, str], num_tol: float = 1e-2) -> bool:
    k1, v1 = kv_gen
    k2, v2 = kv_gt
    if normalize_text_tol(k1) != normalize_text_tol(k2):
        return False
    # try numeric compare
    n1 = try_parse_number(v1)
    n2 = try_parse_number(v2)
    if n1 is not None and n2 is not None:
        # relative tolerance
        denom = max(abs(n1), abs(n2), 1.0)
        return abs(n1 - n2) / denom <= num_tol
    # fallback to normalized exact match
    return normalize_text_tol(v1) == normalize_text_tol(v2)


def jaccard_pairs(a: Set[Tuple[str, str]], b: Set[Tuple[str, str]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def best_chain_match(gen_chain: Dict, gt_chains: List[Dict], mode: str) -> Dict:
    best_score = 0.0
    best_idx = -1
    best_detail = {}

    gen_cond_pairs = split_condition_to_kv_pairs(gen_chain.get('knowledge_chain', {}).get('condition', ''))
    gen_result_kvs = split_result_to_kv(gen_chain.get('knowledge_chain', {}).get('result', ''))
    gen_entity_pairs = extract_entity_pairs(gen_chain.get('entity_recognition', {}))

    # if more than one result kv, result score forced 0 per your rule
    gen_result_count = len(gen_result_kvs)

    for i, gt in enumerate(gt_chains):
        gt_cond_pairs = split_condition_to_kv_pairs(gt.get('knowledge_chain', {}).get('condition', ''))
        gt_result_kvs = split_result_to_kv(gt.get('knowledge_chain', {}).get('result', ''))
        gt_entity_pairs = extract_entity_pairs(gt.get('entity_recognition', {}))

        # condition score: Jaccard over exact (key,value) pairs under mode rules
        if mode == 'strict':
            cond_score = jaccard_pairs(gen_cond_pairs, gt_cond_pairs)
        else:
            # tolerant: compare pairwise using tolerant compare and compute jaccard-like
            matched = 0
            gen_list = list(gen_cond_pairs)
            gt_list = list(gt_cond_pairs)
            used_gt = set()
            for g in gen_list:
                for j, h in enumerate(gt_list):
                    if j in used_gt:
                        continue
                    if compare_kv_tolerant(g, h):
                        matched += 1
                        used_gt.add(j)
                        break
            union = len(gen_list) + len(gt_list) - matched
            cond_score = matched / union if union > 0 else 1.0

        # result score
        if gen_result_count != 1 or len(gt_result_kvs) != 1:
            result_score = 0.0
        else:
            gk = gen_result_kvs[0]
            gtk = gt_result_kvs[0]
            if mode == 'strict':
                result_score = 1.0 if compare_kv_strict(gk, gtk) else 0.0
            else:
                result_score = 1.0 if compare_kv_tolerant(gk, gtk) else 0.0

        # entity score as Jaccard on (type,value) pairs; tolerant mode normalizes text
        if mode == 'strict':
            entity_score = jaccard_pairs(gen_entity_pairs, gt_entity_pairs)
        else:
            # tolerant: normalize and match
            gen_norm = set((t.lower().strip(), normalize_text_tol(v)) for (t, v) in gen_entity_pairs)
            gt_norm = set((t.lower().strip(), normalize_text_tol(v)) for (t, v) in gt_entity_pairs)
            entity_score = jaccard_pairs(gen_norm, gt_norm)

        chain_score = (cond_score + result_score + entity_score) / 3.0

        if chain_score > best_score:
            best_score = chain_score
            best_idx = i
            best_detail = {
                'cond_score': cond_score,
                'result_score': result_score,
                'entity_score': entity_score,
                'chain_score': chain_score
            }

    return {'best_idx': best_idx, 'best_score': best_score, 'detail': best_detail}


def evaluate_results_file(results_file: Path, mode: str, output_file: Path):
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    per_sample = []
    scores = []
    for sample in data:
        gen_json = sample.get('generated_json') or {}
        gt = sample.get('ground_truth') or {}
        gen_km = gen_json.get('knowledge_mining', [])
        gt_km = gt.get('knowledge_mining', [])

        if not gen_km:
            per_sample.append({'file_id': sample.get('file_id'), 'score': 0.0, 'detail': 'no_generated_chains'})
            scores.append(0.0)
            continue

        # for each generated chain find best gt chain
        chain_scores = []
        for g in gen_km:
            bm = best_chain_match(g, gt_km, mode)
            chain_scores.append(bm['best_score'])

        sample_score = sum(chain_scores) / len(chain_scores) if chain_scores else 0.0
        per_sample.append({'file_id': sample.get('file_id'), 'score': sample_score, 'chain_scores': chain_scores})
        scores.append(sample_score)

    summary = {
        'num_samples': len(per_sample),
        'avg_score': sum(scores) / len(scores) if scores else 0.0,
        'min_score': min(scores) if scores else 0.0,
        'max_score': max(scores) if scores else 0.0
    }

    out = {'summary': summary, 'per_sample': per_sample}
    with open(output_file, 'w', encoding='utf-8') as wf:
        json.dump(out, wf, ensure_ascii=False, indent=2)
    print(f"Saved {mode} evaluation to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['strict', 'tolerant'], default='strict')
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    evaluate_results_file(Path(args.results_file), args.mode, Path(args.output_file))


if __name__ == "__main__":
    main()


