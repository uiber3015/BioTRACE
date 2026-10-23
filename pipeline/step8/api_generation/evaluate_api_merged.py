#!/usr/bin/env python3
"""
API 模型评测脚本 - 合并正反向评测
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "step7"))

from step7.r_rule_evaluator_final import compute_r_rule


def load_merged_model_results(results_dir: Path):
    """加载并合并每个模型的正反向结果"""
    models = {}
    
    # 加载 DeepSeek 正向和反向
    ds_forward_path = results_dir / "api_kc_deepseek-v3.2.json"
    ds_reverse_path = results_dir / "api_kc_deepseek-v3.2_reverse.json"
    
    if ds_forward_path.exists():
        with open(ds_forward_path, 'r') as f:
            ds_fwd = json.load(f)
    else:
        ds_fwd = {"samples": []}
        
    if ds_reverse_path.exists():
        with open(ds_reverse_path, 'r') as f:
            ds_rev = json.load(f)
    else:
        ds_rev = {"samples": []}
    
    # 合并 DeepSeek
    ds_samples = ds_fwd.get("samples", []) + ds_rev.get("samples", [])
    ds_ids_seen = set()
    ds_samples_merged = []
    for s in ds_samples:
        sid = s.get("sample_id")
        if sid not in ds_ids_seen:
            ds_ids_seen.add(sid)
            ds_samples_merged.append(s)
    
    models["DeepSeek-V3.2"] = {
        "model_full_name": "DeepSeek-V3.2",
        "samples": ds_samples_merged,
        "forward_only": len(ds_fwd.get("samples", [])),
        "reverse_only": len(ds_rev.get("samples", [])),
    }
    
    # 加载 Gemini 正向
    gm_forward_path = results_dir / "api_kc_gemini-3-flash-preview.json"
    if gm_forward_path.exists():
        with open(gm_forward_path, 'r') as f:
            gm_fwd = json.load(f)
        models["Gemini-3-Flash"] = {
            "model_full_name": "Gemini-3-Flash",
            "samples": gm_fwd.get("samples", []),
            "forward_only": len(gm_fwd.get("samples", [])),
            "reverse_only": 0,
        }
    
    # 加载其他已完成的模型
    for fname in results_dir.glob("api_kc_*.json"):
        name = fname.stem.replace("api_kc_", "")
        if name in ["deepseek-v3.2", "gemini-3-flash-preview"]:
            continue
        try:
            with open(fname, 'r') as f:
                data = json.load(f)
            models[name] = {
                "model_full_name": data.get("model_full_name", name),
                "samples": data.get("samples", []),
            }
        except:
            pass
    
    return models


def evaluate_sample(sample):
    """评估单个样本"""
    ground_truth = sample.get("ground_truth", [])
    if isinstance(ground_truth, list):
        ground_truth = {"knowledge_mining": ground_truth}
    
    generated_kc = sample.get("generated_kc", {"knowledge_mining": []})
    
    eval_sample = {
        "sample_id": sample.get("sample_id", ""),
        "ground_truth": ground_truth,
        "generated_json": generated_kc
    }
    
    return compute_r_rule(eval_sample)


def evaluate_model(model_name, model_data):
    """评测单个模型"""
    samples = model_data["samples"]
    total = len(samples)
    
    r_scores = []
    parse_errors = 0
    empty_outputs = 0
    
    for i, sample in enumerate(samples):
        if not sample.get("success", False):
            parse_errors += 1
            r_scores.append(0.0)
            continue
        
        generated_kc = sample.get("generated_kc", {})
        if not generated_kc or not generated_kc.get("knowledge_mining"):
            empty_outputs += 1
            r_scores.append(0.0)
            continue
        
        result = evaluate_sample(sample)
        r_scores.append(result.get("R_rule", 0.0))
        
        if (i + 1) % 500 == 0:
            print(f"  [{model_name}] {i+1}/{total}")
    
    valid_scores = [s for s in r_scores if s > 0]
    
    return {
        "model": model_name,
        "total_samples": total,
        "avg_r_rule": sum(r_scores) / len(r_scores) if r_scores else 0.0,
        "avg_r_rule_valid": sum(valid_scores) / len(valid_scores) if valid_scores else 0.0,
        "valid_count": len(valid_scores),
        "parse_errors": parse_errors,
        "empty_outputs": empty_outputs,
    }


def main():
    results_dir = Path(__file__).parent
    
    print("=" * 80)
    print("  API 模型评测 (合并正反向)")
    print("=" * 80)
    
    models = load_merged_model_results(results_dir)
    
    all_results = []
    for name, data in sorted(models.items()):
        print(f"\n📊 评测模型: {name}")
        print(f"   样本数: {len(data['samples'])}")
        result = evaluate_model(name, data)
        all_results.append(result)
    
    # 打印结果
    print("\n" + "=" * 80)
    print("  评测结果")
    print("=" * 80)
    
    sorted_results = sorted(all_results, key=lambda x: x["avg_r_rule"], reverse=True)
    
    print(f"\n{'模型':<25} {'R-rule':>8} {'有效样本':>10} {'总样本':>8}")
    print("-" * 60)
    
    for r in sorted_results:
        total = r["total_samples"]
        valid = r["valid_count"]
        pct = 100 * valid / total if total > 0 else 0
        print(f"{r['model']:<25} {r['avg_r_rule']:>8.4f} {valid:>6} ({pct:.1f}%) {total:>8}")
    
    from datetime import datetime
    output_file = results_dir / f"api_evaluation_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
