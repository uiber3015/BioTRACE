#!/usr/bin/env python3
"""
详细评估闭环修正结果：
1. 评估修正前后的分数
2. 统计分数提高的样本个数
3. 实际检查修正后的知识链是否真正有效
"""
import json
import sys
import os
from collections import Counter

sys.path.insert(0, '/home/weijie_yu/yjw/ECLO-RL/step7')
from r_rule_evaluator_final import evaluate_file_final, _normalize_text

def load_ground_truth():
    """加载测试集ground truth"""
    test_dir = '/home/weijie_yu/yjw/ECLO-RL/step6/Test'
    ground_truths = {}
    
    for filename in os.listdir(test_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_id = data.get('file_id', filename.replace('.json', ''))
                ground_truths[file_id] = data
    
    print(f"加载了 {len(ground_truths)} 个测试样本的ground truth")
    return ground_truths

def evaluate_closed_loop_results():
    """评估闭环修正结果"""
    
    # 加载ground truth
    ground_truths = load_ground_truth()
    
    # 加载闭环结果
    result_path = '/home/weijie_yu/yjw/ECLO-RL/step7/closed_loop_results/closed_loop_raw_deepseek_20260211_005130.json'
    with open(result_path, 'r', encoding='utf-8') as f:
        closed_data = json.load(f)
    
    print(f"\n加载了 {len(closed_data)} 个闭环结果")
    
    # 统计变量
    improved_count = 0          # 分数提高的样本数
    decreased_count = 0          # 分数下降的样本数
    same_count = 0               # 分数不变的样本数
    error_count = 0              # 失败的样本数
    
    detailed_results = []
    
    for sample in closed_data:
        sample_id = sample.get('sample_id', 'unknown')
        status = sample.get('status', 'unknown')
        
        # 获取ground truth
        ground_truth = sample.get('ground_truth', {})
        
        # 获取初始生成和最终生成
        initial_generated = sample.get('initial_generated', {})
        final_generated = sample.get('final_generated', {})
        
        # 构建评估数据
        initial_data = [{
            'sample_id': sample_id,
            'ground_truth': ground_truth,
            'generated_json': initial_generated
        }]
        
        final_data = [{
            'sample_id': sample_id,
            'ground_truth': ground_truth,
            'generated_json': final_generated
        }]
        
        # 临时保存并评估
        import tempfile
        
        # 评估初始生成
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
            initial_temp = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
            final_temp = f.name
        
        try:
            initial_stats = evaluate_file_final(initial_temp)
            final_stats = evaluate_file_final(final_temp)
            
            # 提取R-rule分数
            initial_r_rule = initial_stats.get('avg_R_rule', 0)
            final_r_rule = final_stats.get('avg_R_rule', 0)
            initial_s_content = initial_stats.get('avg_S_content', 0)
            final_s_content = final_stats.get('avg_S_content', 0)
            
            # 计算变化
            r_rule_change = final_r_rule - initial_r_rule
            s_content_change = final_s_content - initial_s_content
            
            # 判断是否提高
            if r_rule_change > 0.001:
                improved_count += 1
                improvement_type = "⬆️ 提高"
            elif r_rule_change < -0.001:
                decreased_count += 1
                improvement_type = "⬇️ 下降"
            else:
                same_count += 1
                improvement_type = "➡️ 不变"
            
            # 收集详细信息
            sample_result = {
                'sample_id': sample_id,
                'status': status,
                'initial_r_rule': initial_r_rule,
                'final_r_rule': final_r_rule,
                'r_rule_change': r_rule_change,
                'initial_s_content': initial_s_content,
                'final_s_content': final_s_content,
                's_content_change': s_content_change,
                'improvement_type': improvement_type,
                'initial_chains': len(initial_generated.get('knowledge_mining', [])),
                'final_chains': len(final_generated.get('knowledge_mining', [])),
                'ground_truth_chains': len(ground_truth.get('knowledge_mining', [])),
            }
            detailed_results.append(sample_result)
            
            print(f"[{improvement_type}] 样本 {sample_id}: R-rule {initial_r_rule:.4f} → {final_r_rule:.4f} ({r_rule_change:+.4f}) | "
                  f"S_content {initial_s_content:.4f} → {final_s_content:.4f} ({s_content_change:+.4f}) | "
                  f"链数: {sample_result['initial_chains']} → {sample_result['final_chains']} | "
                  f"GT: {sample_result['ground_truth_chains']}")
            
        except Exception as e:
            error_count += 1
            print(f"[❌ 错误] 样本 {sample_id}: {e}")
            sample_result = {
                'sample_id': sample_id,
                'status': status,
                'error': str(e),
                'improvement_type': "❌ 错误"
            }
            detailed_results.append(sample_result)
        
        finally:
            os.unlink(initial_temp)
            os.unlink(final_temp)
    
    # 打印统计摘要
    print("\n" + "=" * 80)
    print("📊 闭环修正效果统计摘要")
    print("=" * 80)
    print(f"总样本数: {len(closed_data)}")
    print(f"成功完成: {len(closed_data) - error_count}")
    print(f"失败: {error_count}")
    print()
    print(f"分数提高: {improved_count} ({improved_count/len(closed_data)*100:.1f}%)")
    print(f"分数下降: {decreased_count} ({decreased_count/len(closed_data)*100:.1f}%)")
    print(f"分数不变: {same_count} ({same_count/len(closed_data)*100:.1f}%)")
    print()
    
    # 详细检查修正后的知识链
    print("\n" + "=" * 80)
    print("🔍 修正后知识链有效性详细检查")
    print("=" * 80)
    
    for result in detailed_results:
        if 'error' in result:
            continue
            
        sample_id = result['sample_id']
        final_r_rule = result['final_r_rule']
        final_s_content = result['final_s_content']
        initial_chains = result['initial_chains']
        final_chains = result['final_chains']
        gt_chains = result['ground_truth_chains']
        improvement_type = result['improvement_type']
        
        print(f"\n[{improvement_type}] 样本 {sample_id}:")
        print(f"  初始链数: {initial_chains} → 最终链数: {final_chains} | GT链数: {gt_chains}")
        print(f"  最终分数: R-rule={final_r_rule:.4f}, S_content={final_s_content:.4f}")
        
        # 评估知识链质量
        if final_r_rule >= 0.9 and final_s_content >= 0.8:
            print(f"  ✅ 质量优秀: R-rule≥0.9 且 S_content≥0.8")
        elif final_r_rule >= 0.7 and final_s_content >= 0.5:
            print(f"  ⚠️ 质量一般: 0.7≤R-rule<0.9 或 0.5≤S_content<0.8")
        elif final_r_rule >= 0.5:
            print(f"  ❌ 质量较差: R-rule<0.7 或 S_content<0.5")
        else:
            print(f"  ⛔ 质量很差: R-rule<0.5 且 S_content<0.5")
        
        # 检查链数合理性
        if final_chains == gt_chains:
            print(f"  ✅ 链数匹配: 与GT数量一致 ({gt_chains})")
        elif final_chains > gt_chains:
            over_percent = (final_chains - gt_chains) / gt_chains * 100
            print(f"  ⚠️ 链数过多: 比GT多 {over_percent:.1f}%")
        else:
            under_percent = (gt_chains - final_chains) / gt_chains * 100
            print(f"  ⚠️ 链数过少: 比GT少 {under_percent:.1f}%")
    
    # 保存详细结果
    output_path = '/home/weijie_yu/yjw/ECLO-RL/step7/closed_loop_evaluation_detailed.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_samples': len(closed_data),
                'success_count': len(closed_data) - error_count,
                'error_count': error_count,
                'improved_count': improved_count,
                'decreased_count': decreased_count,
                'same_count': same_count,
                'improved_percent': improved_count / len(closed_data) * 100 if len(closed_data) > 0 else 0,
            },
            'detailed_results': detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 详细结果已保存到: {output_path}")
    
    return {
        'total': len(closed_data),
        'improved': improved_count,
        'decreased': decreased_count,
        'same': same_count,
        'error': error_count
    }

if __name__ == '__main__':
    stats = evaluate_closed_loop_results()

