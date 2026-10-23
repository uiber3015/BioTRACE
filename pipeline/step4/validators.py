"""JSON验证和清洗函数"""
import json
import re
from entity_types import ENTITY_TYPE_LIST


def clean_json_text(text):
    """清理JSON文本，移除常见格式问题"""
    # 1. 移除 Qwen3 的 <think> 标签及其内容
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 提取 markdown 代码块中的内容（如果存在）
    # 查找 ```json ... ``` 或 ``` ... ```
    code_block_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)
    else:
        # 如果没有代码块，移除可能的 markdown 标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text)
        text = re.sub(r'^```', '', text)
    
    # 3. 移除前后空白
    text = text.strip()
    
    # 4. 修复常见JSON错误
    # 移除尾部多余逗号
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # 移除注释（// 和 /* */）
    text = re.sub(r'//.*?$', '', text, flags=re.MULTILINE)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    return text


def validate_chain(chain):
    """
    验证单条知识链的合法性
    
    Returns:
        (is_valid: bool, errors: list)
    """
    errors = []
    
    # 1. 检查必需字段
    required_fields = ['condition', 'result', 'entity']
    for field in required_fields:
        if field not in chain:
            errors.append(f"缺少必需字段: {field}")
    
    if errors:
        return False, errors
    
    # 2. 检查 condition
    condition = chain.get('condition', '')
    if not condition or not isinstance(condition, str) or len(condition.strip()) == 0:
        errors.append("condition 必须是非空字符串")
    
    # 3. 检查 result
    result = chain.get('result', '')
    if not result or not isinstance(result, str):
        errors.append("result 必须是字符串")
    elif ':' not in result:
        errors.append("result 必须包含至少一个 '指标:数值' 对（用冒号分隔）")
    
    # 4. 检查 entity
    entity = chain.get('entity', {})
    if not isinstance(entity, dict):
        errors.append("entity 必须是字典类型")
    elif len(entity) == 0:
        errors.append("entity 必须包含至少一个实体类型")
    else:
        # 4.1 检查实体类型是否在白名单中
        for entity_type in entity.keys():
            if entity_type not in ENTITY_TYPE_LIST:
                errors.append(f"未知的实体类型: '{entity_type}'（必须使用英文实体类型，如 Gene Symbol, Disease 等）")
        
        # 4.2 检查实体值类型
        for entity_type, values in entity.items():
            if not isinstance(values, list):
                errors.append(f"实体类型 '{entity_type}' 的值必须是数组")
                continue
            
            if len(values) == 0:
                errors.append(f"实体类型 '{entity_type}' 的值数组不能为空")
        
        # 4.3 检查实体值是否在 condition/result 中出现（放宽规则）
        combined_text = str(condition) + "|" + str(result)
        for entity_type, values in entity.items():
            if isinstance(values, list):
                for value in values:
                    value_str = str(value)
                    
                    # 清理和规范化
                    # 1. 移除括号及其内容（处理 "disease (abbr)" 格式）
                    cleaned_value = re.sub(r'\s*\([^)]*\)', '', value_str).strip()
                    # 2. 规范化标点符号空格（处理 "P. ananatis" vs "P.ananatis"）
                    cleaned_value = re.sub(r'\s*\.\s*', '.', cleaned_value)  # 点号
                    cleaned_value = re.sub(r'\s*,\s*', ',', cleaned_value)   # 逗号
                    cleaned_value = re.sub(r'\s*:\s*', ':', cleaned_value)   # 冒号
                    # 3. 规范化空格（移除多余空格）
                    normalized_value = ' '.join(cleaned_value.split())
                    # 4. 转为小写（不区分大小写匹配）
                    normalized_value_lower = normalized_value.lower()
                    
                    # 同样处理combined_text（与实体值清理逻辑保持一致）
                    # 1. 移除括号及其内容
                    normalized_text = re.sub(r'\s*\([^)]*\)', '', combined_text).strip()
                    # 2. 规范化标点符号空格
                    normalized_text = re.sub(r'\s*\.\s*', '.', normalized_text)
                    normalized_text = re.sub(r'\s*,\s*', ',', normalized_text)
                    normalized_text = re.sub(r'\s*:\s*', ':', normalized_text)
                    # 3. 规范化空格
                    normalized_text = ' '.join(normalized_text.split())
                    # 4. 转为小写
                    normalized_text_lower = normalized_text.lower()
                    
                    # 检查是否出现（部分匹配，不区分大小写）
                    if normalized_value_lower not in normalized_text_lower:
                        errors.append(f"实体值 '{value_str}' (类型:{entity_type}) 未在 condition 或 result 中出现")
    
    return len(errors) == 0, errors


def parse_and_validate(raw_output):
    """
    解析并验证JSON输出
    
    Returns:
        (valid_chains: list or None, message: str)
    """
    # 1. 清理文本
    cleaned = clean_json_text(raw_output)
    
    if not cleaned:
        return None, "输出为空"
    
    # 2. 尝试解析JSON
    try:
        chains = json.loads(cleaned)
    except json.JSONDecodeError as e:
        # 尝试修复常见问题后再解析一次
        try:
            # 强制修复：移除所有注释和多余逗号
            cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
            chains = json.loads(cleaned)
        except:
            return None, f"JSON 解析失败: {str(e)}"
    
    # 3. 必须是数组
    if not isinstance(chains, list):
        return None, "输出必须是 JSON 数组"
    
    # 4. 空数组是合法的（表示无法抽取）
    if len(chains) == 0:
        return [], "空数组（表格无有效知识链）"
    
    # 5. 验证每条链
    valid_chains = []
    validation_errors = []
    
    for i, chain in enumerate(chains):
        if not isinstance(chain, dict):
            validation_errors.append(f"链 {i+1}: 必须是对象类型")
            continue
        
        is_valid, errors = validate_chain(chain)
        if is_valid:
            valid_chains.append(chain)
        else:
            validation_errors.append(f"链 {i+1}: {'; '.join(errors)}")
    
    # 6. 如果没有有效链，返回失败
    if len(valid_chains) == 0:
        error_summary = '\n  - '.join(validation_errors[:5])  # 只显示前5个错误
        return None, f"所有知识链验证失败:\n  - {error_summary}"
    
    # 7. 返回有效的链
    success_rate = len(valid_chains) / len(chains) * 100
    message = f"成功验证 {len(valid_chains)}/{len(chains)} 条链 ({success_rate:.1f}%)"
    
    if len(validation_errors) > 0:
        message += f"（{len(validation_errors)} 条被过滤）"
    
    return valid_chains, message
