"""知识链提示词模板 - 适配结构化JSON输入（任务2输出格式）"""
from entity_types import get_entity_types_description


SYSTEM_PROMPT = """你是一个专业的生物医学知识抽取专家。你的任务是从结构化表格JSON中抽取知识链，输出严格合法的 JSON 格式。

**核心要求**：
1. 仅输出 JSON 数组，不得有任何其他文字
2. 严格遵循 JSON 语法，不得有重复键、多余逗号
3. 如果无法抽取有效知识链，输出空数组 []
"""


def build_user_prompt(table_json):
    """
    构建用户提示词
    
    Args:
        table_json: dict格式的表格数据
            {
                "table_title": "...",
                "rows": [
                    {"列1": "值1", "列2": "值2"},
                    {"列1": "值3", "列2": "值4"}
                ]
            }
    """
    
    entity_desc = get_entity_types_description()
    
    # 提取表格信息
    table_title = table_json.get('table_title') or '无标题'
    rows = table_json.get('rows', [])
    
    # 格式化rows为易读格式
    import json
    rows_display = json.dumps(rows, ensure_ascii=False, indent=2)
    
    # 提取列名
    column_names = list(rows[0].keys()) if rows else []
    
    prompt = f"""请从以下结构化表格JSON中抽取知识链。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 输入数据
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 表格标题
{table_title}

### 表格结构
- 列数: {len(column_names)}
- 行数: {len(rows)}
- 列名: {', '.join(column_names)}

### 表格数据（JSON格式）
```json
{rows_display}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 抽取规则
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 一、实体类型定义（14种）

{entity_desc}

### 二、知识链抽取规则

**知识链 = condition（条件） + result（结果） + entity（实体）**

#### 2.1 JSON表格解析策略

**输入格式说明**：
- 表格已经是结构化JSON，每个对象是一行数据
- 对象的键（key）是列名，值（value）是单元格内容
- 不需要解析markdown，直接读取JSON结构

**知识抽取策略**：

1. **识别表格类型**
   
   a) **横向表格**（最常见）：
      - 每个row对象代表一条独立知识
      - **第一个键通常是实体名称列**（如基因名、化合物名、物种名）
      - 其他键是属性列（描述、分类）或数值列（测量结果）
      
      示例：
      ```json
      {{
        "Gene_Symbol": "BRCA1",        // ← 实体名称
        "Function": "DNA_repair",       // ← 描述属性
        "Expression_Level": "2.5"       // ← 数值结果
      }}
      ```
      
      → condition: "BRCA1|DNA_repair"
      → result: "Expression_Level: 2.5"
      → entity: {{"Gene Symbol": ["BRCA1"]}}
   
   b) **矩阵型表格**（行列都是实体）：
      - 第一个键是行标识
      - 其他键是列标识
      - 值是交叉关系的结果
      
      示例：
      ```json
      {{
        "Compound": "Aspirin",          // ← 行实体
        "Species_A": "0.5",             // ← 列实体的测量值
        "Species_B": "0.8"
      }}
      ```
      
      → 抽取2条知识链：
         1. condition: "Aspirin|Species_A", result: "value: 0.5"
         2. condition: "Aspirin|Species_B", result: "value: 0.8"

2. **构建condition（条件）**
   
   **必须包含**：
   - ✅ 第一个键的值（通常是实体名称）
   - ✅ 所有描述性键的值（非数值的属性）
   - ✅ 表格标题的研究背景
   
   **格式**：多个条件用 `|` 分隔
   
   **关键规则**：
   - 实体名称必须在condition中出现
   - 确保entity中的值能在condition中找到
   
   **示例**：
   ```json
   // 输入行：
   {{"Gene": "TP53", "Pathway": "Apoptosis", "Fold_Change": "3.2"}}
   
   // 正确：
   "condition": "TP53|Apoptosis"  // ✅ 包含实体名
   "entity": {{"Gene Symbol": ["TP53"]}}  // ✅ TP53在condition中
   
   // 错误：
   "condition": "Gene_expression|Apoptosis"  // ❌ 没有实体名
   "entity": {{"Gene Symbol": ["TP53"]}}     // ❌ TP53不在condition中
   ```

3. **构建result（结果）**
   
   **严格格式**：`"指标: 数值"`
   
   **识别数值列**：
   - 值是数字、百分比、科学计数法
   - 列名通常包含：value, level, score, rate, count, fold, p-value等
   
   **格式规则**：
   - 使用列名作为指标名
   - 多个结果用 `|` 分隔
   - 如果列名含下划线，保持原样或转为空格
   
   **示例**：
   ```json
   // 输入：
   {{"Protein": "P53", "Abundance": "150", "p_value": "0.001"}}
   
   // 输出：
   "result": "Abundance: 150|p_value: 0.001"
   ```

4. **构建entity（实体）**
   
   **抽取规则**：
   - 只从**已包含在condition中的值**提取实体
   - 根据列名和值的语义判断实体类型
   - 使用14种预定义类型，严禁自创
   
   **列名到实体类型的映射提示**：
   ```
   Gene/Gene_Symbol/Gene_Name → "Gene Symbol"
   Protein/Protein_Name → "Protein"
   Species/Organism → "Species"
   Disease/Disorder → "Disease"
   Drug/Compound/Chemical → "Drug" 或 "Compound"
   Cell_Type/Cell_Line → "Cell"
   Tissue/Organ → "Tissue" 或 "Organ"
   Method/Technique → "Experimental Method"
   Phenotype/Trait → "Phenotype"
   Regulator/Regulator_Gene → "Regulator"
   Target/Target_Gene → "Target"
   Mechanism/Function → "Functional Mechanism"
   ```
   
   **验证规则**：
   - entity中的每个值必须在condition或result中出现
   - 空列表的实体类型必须删除
   - 对每个实体列表去重

#### 2.2 完整抽取流程

**Step 1**: 读取JSON的每个row对象

**Step 2**: 识别列的类型
   - 实体列：第一列，通常包含名称
   - 属性列：描述性文本
   - 数值列：测量结果

**Step 3**: 为每行构建知识链
   ```python
   condition = 实体列的值 + "|" + 属性列的值
   result = "数值列名: 数值列值"
   entity = {{实体类型: [从condition中提取的实体]}}
   ```

**Step 4**: 验证
   - 检查JSON格式
   - 检查entity对齐（每个值必须在condition中）
   - 去重并清理空列表

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 示例1：基因表达表格

**输入JSON**：
```json
[
  {{
    "Gene_Symbol": "BRCA1",
    "Function": "DNA_repair",
    "Expression_Fold_Change": "2.5",
    "p_value": "0.001"
  }},
  {{
    "Gene_Symbol": "TP53",
    "Function": "Apoptosis",
    "Expression_Fold_Change": "3.2",
    "p_value": "0.0001"
  }}
]
```

**输出知识链**：
```json
[
  {{
    "condition": "BRCA1|DNA_repair",
    "result": "Expression_Fold_Change: 2.5|p_value: 0.001",
    "entity": {{
      "Gene Symbol": ["BRCA1"],
      "Functional Mechanism": ["DNA_repair"]
    }}
  }},
  {{
    "condition": "TP53|Apoptosis",
    "result": "Expression_Fold_Change: 3.2|p_value: 0.0001",
    "entity": {{
      "Gene Symbol": ["TP53"],
      "Functional Mechanism": ["Apoptosis"]
    }}
  }}
]
```

### 示例2：药物敏感性表格

**输入JSON**：
```json
[
  {{
    "Compound": "Aspirin",
    "Disease": "Cardiovascular_disease",
    "IC50": "50",
    "Efficacy": "High"
  }}
]
```

**输出知识链**：
```json
[
  {{
    "condition": "Aspirin|Cardiovascular_disease|Efficacy: High",
    "result": "IC50: 50",
    "entity": {{
      "Drug": ["Aspirin"],
      "Disease": ["Cardiovascular_disease"]
    }}
  }}
]
```

### 示例3：无法抽取（缺少生物实体）

**输入JSON**：
```json
[
  {{
    "Sample_ID": "S001",
    "Temperature": "37",
    "pH": "7.4"
  }}
]
```

**输出知识链**：
```json
[]
```
原因：缺少生物医学实体，仅有实验条件参数

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 质量要求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ **必须做到**：
1. JSON 语法完全正确，可直接解析
2. 每条链有 condition、result、entity 三个字段
3. result 严格遵循 "指标:数值" 格式
4. entity 的值必须在 condition 的文本部分中实际出现
5. 实体类型键必须且只能从14种中选择
6. 对实体列表去重，删除空列表
7. 实体值使用JSON中的原始值（不改写、不标准化）

❌ **严禁**：
1. 输出非 JSON 内容（如注释、说明、`<think>`标签）
2. 编造JSON中不存在的数据
3. 自创实体类型（如 Temperature, Method, Sample_ID, Score 等）
4. 输出空的实体类型列表
5. 完全重复的知识链
6. 改写或标准化实体值

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 输出要求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. **直接输出JSON数组**，不要包含任何推理过程
2. **只输出纯JSON**，格式：`[{{"condition": ..., "result": ..., "entity": ...}}, ...]`
3. 如果无法抽取有效知识链，输出空数组：`[]`
4. 如果某个表格无生物医学实体，entity可以为空对象：`{{}}`

**现在请开始抽取，直接输出 JSON 数组：**
"""
    
    return prompt
