from typing import Dict, Any, List
import json
import re
# 兼容直接脚本加载：优先绝对导入，失败再尝试相对导入
try:
    from entity_type import EntityTypeAgent
except ImportError:
    try:
        from .entity_type import EntityTypeAgent
    except ImportError:
        import sys
        from pathlib import Path

        step5_dir = Path(__file__).resolve().parent
        if str(step5_dir) not in sys.path:
            sys.path.insert(0, str(step5_dir))
        from entity_type import EntityTypeAgent


class EntityTypeAgentV1(EntityTypeAgent):
    """
    精简提示版实体识别（基因领域），逻辑沿用基类，只替换提示词。
    """

    _GENE_ITEMS_PROMPT = """你是一个基因领域实体标注器。

【绝对输出要求】
你接收到的输入是一个 JSON，对其中的每条 knowledge_chain 做实体识别。
你的最终输出必须是且仅是下面这种结构的 JSON：
{
  "knowledge_chains": [
    {
      "chain_index": <整数>,
      "entity": {
        "<实体类型1>": ["实体1", "实体2"],
        "<实体类型2>": ["实体3"]
      }
    },
    ...
  ]
}
要求：
- 顶层只能有一个字段：knowledge_chains。
- knowledge_chains 是数组。
- 数组中每个元素必须同时包含：
  - chain_index：整数，对应原始链的索引。
  - entity：对象；若该链没有任何实体，则为 {}。
- 不能输出任何多余内容：不能有额外顶层字段，不能有文本说明、注释、代码块标记、示例说明等。

【输入说明】
- 输入 JSON 中每条 knowledge_chain 至少包含：
  - chain_index：该链的索引（整数）。
  - items：数组，元素形如 {"text": "...", "source": "condition_value"}。
- text：已经是需要判断的完整字符串，即使包含顿号、逗号、分号等也不得拆分，必须整体判断。
- source：仅作参考，表示该字符串来自 condition_key / condition_value / result_key / result_value，但不强制决定实体类型。

【需要识别的 14 类实体】（此段定义内容不得修改）
1. gene_symbol：基因的官方符号或常用名（如 MALAT1、NEAT2、PF01715 等命名基因）。
2. regulator：调控因子，处于调控关系上游的分子实体（如 p53、ER、TMZ）。
3. target：靶点，被特定分子调控的下游分子或区域（如 TP53、DSCAM-AS1 promoter）。
4. functional_mechanism：功能机制，描述分子发挥作用的途径或过程（如 转录调控、pathogenesis、stem cell marker expression）。
5. species：物种名称（如 human、mouse、P. ananatis）。
6. disease：疾病或异常生理状态（如 colorectal cancer、ulcerative colitis、CRC）。
7. organ：器官层级（如 heart、colorectal、renal）。
8. tissue：组织层级（如 skeletal muscle、neocortical tissues）。
9. cell：细胞或细胞系（如 macrophages、GMB cell lines、HeLa）。
10. protein：蛋白质产物（如 IPP transferase、GMP synthase C terminal domain）。
11. compound：化合物/物质（如 glucose、PF00958、Phenolic acid decarboxylase）。
12. drug：药物或具有治疗作用的制剂（如 TMZ、Aspirin）。
13. experimental_method：实验技术或平台（如 LC-MS/MS、RNA-seq、2D-PAGE）。
14. phenotype：可观察的表型/性状（如 antibiotc resistance、biogas production 表型描述）。
（描述指标、测量值或纯数值的词语不属于任何实体）

【标注规则】
1. 遍历每条 knowledge_chain 的 items：
   - 针对每个 items.text，判断其整体是否属于上述 14 类实体中的某一类。
   - 若属于，则将该 text 原文加入对应实体类型数组。
   - 可参考 source 辅助判断（如很多 result_key 是标签，不是实体），但最终以 text 内容为准。
2. 不拆分 text：
   - text 内即便包含多个以逗号、顿号、分号分隔的项目，也必须视为“一个整体短语”整体判断。
   - 只有当整个短语整体明显对应“单一实体”时才打标；若短语包含多个并列实体（用逗号/and/or/加号/下划线/斜杠/管道符等连接）或混合概念，整体不标。
   - 绝不拆分子短语单独打标。
3. 去重与顺序：
   - 同一 knowledge_chain 中，同一实体类型下的字符串去重，按首次出现顺序保留。
   - 若某条链未识别出任何实体，则该链的 entity 字段为 {}（空对象，不省略）。
4. 禁止的推断与拆分（非常重要）：
   - 只能标注 items.text 中显式出现的内容：
     - 不根据背景知识补全全称或新增别名。
     - 不从代码/编号推断出额外的蛋白/基因/域名称。
   - “数字 + 名词”的计数字段一律视为数量，整体不拆分、不打实体，例如：
     - "33 proteins"
     - "6070 genes"
     - "884,000 reads"
     - "0.01%"
     - "1.2e5 copies/mL"
     - "n=20"
     - "Average age:50"
   - 上述字符串整体不打任何实体标签。
   - 仅有笼统词的短语（如 "proteins", "housekeeping proteins", "identified proteins:33 proteins"）无具体名称，不打实体。
   - 包含多个疾病/蛋白并列的长串（如 "cardiovascular disease cancer diabetes chronic respiratory disease" 或 "enzymes of methanogenesis + housekeeping proteins"）整体不标。
   - 等位/突变/基因型字符串（如 "F508del/G542X", "p.V600E"）若未出现明确疾病名，视为基因变异，不标 disease；可不标或仅在明确为基因符号/变异时标 gene_symbol。
5. 特别注意的非实体：
   - 国家或地区名称（如 Italy、USA、China、Japan、France、Germany 等）：
     - 不标记为 species，也不标其他实体，直接忽略。
   - 仅表示分组/批次/编号的词语（如 Group A、Cohort 1、Sample 01、Patient 1、Case、Control 等）：
     - 不标记为任何实体。
   - Pfam/域编号等标识符：
     - 如果只是一串编号，不将其视为 protein/compound/gene_symbol。
     - 只有当 text 中包含清晰的蛋白/域名称时，才标为 protein 等。
6. 列表/限定短语整体判断：
   - 含多个条目的列表短语，整体作为一个短语来判断：
     - 若整体明显是某类实体（如 “RNA-seq, LC-MS/MS and 2D-PAGE” 是实验方法列表），可整体标为 experimental_method。
     - 若整体语义混杂或不清晰，可不标。
   - 限定短语如 “Only in P. ananatis strains”：
     - 作为整体判断；若整体不清晰可不标。
     - 不要拆出其中的物种/蛋白/域编号单独打标。

【输出格式要求（再次强调）】
- 最终输出必须是一个 JSON 对象，格式固定为：
{
  "knowledge_chains": [
    {
      "chain_index": <整数>,
      "entity": { ... }
    },
    ...
  ]
}
- 不允许有任何其他顶层字段或结构。
- 不允许输出任何解释性文字、注释、Markdown 代码块标记等。
"""

    def _build_gene_batch_system_prompt(self) -> str:
        return self._GENE_ITEMS_PROMPT

    _GENE_ITEMS_PROMPT_STRICT = """你是一个基因领域实体标注器。

【绝对输出要求】
你接收到的输入是一个 JSON，对其中的每条 knowledge_chain 做实体识别。
你的最终输出必须是且仅是下面这种结构的 JSON：
{
  "knowledge_chains": [
    {
      "chain_index": <整数>,
      "entity": {
        "<实体类型1>": ["实体1", "实体2"],
        "<实体类型2>": ["实体3"]
      }
    },
    ...
  ]
}
要求：
- 顶层只能有一个字段：knowledge_chains。
- knowledge_chains 是数组。
- 数组中每个元素必须同时包含：
  - chain_index：整数，对应原始链的索引。
  - entity：对象；若该链没有任何实体，则为 {}。
- 不能输出任何多余内容：不能有额外顶层字段，不能有文本说明、注释、代码块标记、示例说明等。

【核心约束：实体必须完全匹配原始文本】
**这是最重要的规则**：你标注的每个实体值，必须是在对应 knowledge_chain 的 condition 或 result 字段中**完全匹配出现**的文本片段。
- 不允许标注任何在 condition/result 中找不到的实体。
- 不允许从 condition/result 中提取部分文本后重新组合成新实体。
- 不允许根据背景知识推断或补全实体名称。
- 实体值必须与 condition/result 中的某个完整文本片段完全一致（大小写、标点、空格都要一致）。
- **绝对禁止拆分原始 items.text**：如果 items.text 是 "↓ Kv4.2 [A-type K+ currents] [32]"，你不能只标注其中的 "Kv4.2"。要么标注整个文本（如果整体是实体），要么不标注。绝不能从原始文本中提取子字符串单独标注。
- 每个实体值必须与某个 items.text 完全匹配，不能是 items.text 的子字符串拆分。

【输入说明】
- 输入 JSON 中每条 knowledge_chain 包含：
  - chain_index：该链的索引（整数）。
  - condition：条件字符串，格式为 "key1:value1 | key2:value2 | ..."
  - result：结果字符串，格式为 "指标名:数值" 或空字符串。
  - items：数组，元素形如 {"text": "...", "source": "condition_value"}。
- text：已经是需要判断的完整字符串，即使包含顿号、逗号、分号等也不得拆分，必须整体判断。
- source：仅作参考，表示该字符串来自 condition_key / condition_value / result_key / result_value。

【需要识别的 14 类实体】（此段定义内容不得修改）
1. gene_symbol：基因的官方符号或常用名（如 MALAT1、NEAT2、PF01715 等命名基因）。
2. regulator：调控因子，处于调控关系上游的分子实体（如 p53、ER、TMZ）。
3. target：靶点，被特定分子调控的下游分子或区域（如 TP53、DSCAM-AS1 promoter）。
4. functional_mechanism：功能机制，描述分子发挥作用的途径或过程（如 转录调控、pathogenesis、stem cell marker expression）。
5. species：物种名称（如 human、mouse、P. ananatis）。
6. disease：疾病或异常生理状态（如 colorectal cancer、ulcerative colitis、CRC）。
7. organ：器官层级（如 heart、colorectal、renal）。
8. tissue：组织层级（如 skeletal muscle、neocortical tissues）。
9. cell：细胞或细胞系（如 macrophages、GMB cell lines、HeLa）。
10. protein：蛋白质产物（如 IPP transferase、GMP synthase C terminal domain）。
11. compound：化合物/物质（如 glucose、PF00958、Phenolic acid decarboxylase）。
12. drug：药物或具有治疗作用的制剂（如 TMZ、Aspirin）。
13. experimental_method：实验技术或平台（如 LC-MS/MS、RNA-seq、2D-PAGE）。
14. phenotype：可观察的表型/性状（如 antibiotc resistance、biogas production 表型描述）。
（描述指标、测量值或纯数值的词语不属于任何实体）

【标注规则】
1. **严格匹配检查**：在标注实体前，必须确认该实体值在对应 knowledge_chain 的 condition 或 result 中完全匹配出现。
   - 例如：如果 condition 是 "Locusa:Cj0151_00022 | Putative_functionb:Hypothetical | BLAST_identity_results:Low identity across Campylobacter genus"
   - 你可以标注 "Cj0151_00022" 为 gene_symbol（因为它在 condition 中完全匹配）
   - 你可以标注 "Campylobacter genus" 为 species（因为它在 condition 中完全匹配）
   - 但**不能**标注 "Campylobacter" 单独为 species（因为 condition 中是 "Campylobacter genus"，不是 "Campylobacter"）
2. 遍历每条 knowledge_chain 的 items：
   - 针对每个 items.text，判断其整体是否属于上述 14 类实体中的某一类。
   - **但前提是**：该 text 必须在 condition 或 result 中完全匹配出现。
   - 若属于且完全匹配，则将该 text 原文加入对应实体类型数组。
3. 不拆分 text：
   - text 内即便包含多个以逗号、顿号、分号分隔的项目，也必须视为"一个整体短语"整体判断。
   - 只有当整个短语整体明显对应"单一实体"且完全匹配时才打标。
   - 绝不拆分子短语单独打标。
4. 去重与顺序：
   - 同一 knowledge_chain 中，同一实体类型下的字符串去重，按首次出现顺序保留。
   - 若某条链未识别出任何实体，则该链的 entity 字段为 {}（空对象，不省略）。
5. 禁止的推断与拆分（非常重要）：
   - 只能标注 items.text 中显式出现且完全匹配的内容。
   - "数字 + 名词"的计数字段一律视为数量，整体不拆分、不打实体。
   - 仅有笼统词的短语（如 "proteins", "housekeeping proteins"）无具体名称，不打实体。
   - 包含多个疾病/蛋白并列的长串整体不标。
   - 等位/突变/基因型字符串若未出现明确疾病名，视为基因变异，不标 disease。
6. 特别注意的非实体：
   - 国家或地区名称不标记为 species，也不标其他实体，直接忽略。
   - 仅表示分组/批次/编号的词语不标记为任何实体。
   - Pfam/域编号等标识符如果只是一串编号，不将其视为 protein/compound/gene_symbol。

【输出格式要求（再次强调）】
- 最终输出必须是一个 JSON 对象，格式固定为：
{
  "knowledge_chains": [
    {
      "chain_index": <整数>,
      "entity": { ... }
    },
    ...
  ]
}
- 不允许有任何其他顶层字段或结构。
- 不允许输出任何解释性文字、注释、Markdown 代码块标记等。
"""

    def _extract_key_value_pairs(self, condition: str, result: str) -> set:
        """
        从 condition 和 result 中提取所有的 key:value 片段（包括 key 和 value）
        返回所有可能的完全匹配文本集合
        """
        pairs = set()
        
        # 处理 condition: "key1:value1 | key2:value2 | ..."
        if condition:
            for part in condition.split('|'):
                part = part.strip()
                if ':' in part:
                    key, value = part.split(':', 1)
                    pairs.add(key.strip())
                    pairs.add(value.strip())
                else:
                    pairs.add(part)
        
        # 处理 result: "指标名:数值"
        if result:
            for part in result.split('|'):
                part = part.strip()
                if ':' in part:
                    key, value = part.split(':', 1)
                    pairs.add(key.strip())
                    pairs.add(value.strip())
                else:
                    pairs.add(part)
        
        return pairs

    async def validate_and_regenerate_entities(
        self,
        knowledge_chains: List[Dict[str, Any]],
        entity_recognitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        验证实体识别结果：检查每个实体值是否完全匹配 condition/result 中的 key 或 value。
        如果发现不匹配（是子字符串拆分），只对不匹配的实体值对应的 items.text 进行重新识别。
        
        Args:
            knowledge_chains: 知识链列表
            entity_recognitions: 实体识别结果列表
            
        Returns:
            修正后的实体识别结果列表
        """
        updated_recognitions = list(entity_recognitions)
        
        # 对每条链进行检查
        for idx, (chain, entity_rec) in enumerate(zip(knowledge_chains, entity_recognitions)):
            condition = chain.get('condition', '')
            result = chain.get('result', '')
            entity = entity_rec.get('entity', {})
            
            # 提取 condition/result 中的所有 key:value 片段
            valid_texts = self._extract_key_value_pairs(condition, result)
            
            # 获取原始的 items 列表
            items = self._extract_gene_chain_items(chain)
            
            # 检查每个实体值，找出不匹配的实体及其对应的 items
            invalid_entities = []  # [(entity_type, entity_value, item_text, item_index)]
            valid_entity = {}  # 保留匹配的实体
            
            for entity_type, entity_values in entity.items():
                if not isinstance(entity_values, list):
                    continue
                
                valid_values = []
                for entity_value in entity_values:
                    if not isinstance(entity_value, str):
                        continue
                    
                    # 检查是否完全匹配 condition/result 中的某个 key 或 value
                    if entity_value in valid_texts:
                        # 完全匹配，保留
                        valid_values.append(entity_value)
                    else:
                        # 不匹配，需要找出对应的 items.text 进行重新识别
                        # 查找包含该实体值的 items
                        found_item = None
                        for item_idx, item in enumerate(items):
                            item_text = item.get('text', '')
                            if entity_value in item_text:
                                found_item = (item_text, item_idx)
                                break
                        
                        if found_item:
                            invalid_entities.append((entity_type, entity_value, found_item[0], found_item[1]))
                            self.log_activity(
                                f"链 {idx} 的实体 '{entity_value}' (类型: {entity_type}) 不完全匹配，需要重新识别对应的 items.text: '{found_item[0]}'"
                            )
                        else:
                            # 找不到对应的 items，直接移除
                            self.log_activity(
                                f"链 {idx} 的实体 '{entity_value}' (类型: {entity_type}) 不在 condition/result 中且找不到对应 items，移除"
                            )
                
                if valid_values:
                    valid_entity[entity_type] = valid_values
            
            # 如果有不匹配的实体，只对这些 items 进行重新识别
            if invalid_entities:
                # 收集需要重新识别的 items（去重）
                items_to_regen = {}
                for entity_type, entity_value, item_text, item_idx in invalid_entities:
                    if item_text not in items_to_regen:
                        items_to_regen[item_text] = {
                            'text': item_text,
                            'source': items[item_idx].get('source', 'condition_value'),
                            'item_index': item_idx
                        }
                
                # 只对这些 items 进行实体识别
                try:
                    payload = [{
                        "chain_index": 0,
                        "items": list(items_to_regen.values())
                    }]
                    
                    system_prompt = self._GENE_ITEMS_PROMPT_STRICT
                    payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
                    instructions = self._build_gene_batch_user_instructions()
                    user_text = f"""以下是需要重新识别的 items（共 {len(items_to_regen)} 个）：
{payload_json}

{instructions}

**重要提醒**：
1. 请仔细检查每个实体值是否在对应的 condition 或 result 字段中完全匹配出现。如果找不到完全匹配，请不要标注该实体。
2. **绝对禁止拆分原始文本**：要么标注整个文本（如果整体是实体），要么不标注。绝不能从原始文本中提取子字符串单独标注。
3. 每个实体值必须与某个 items.text 完全匹配，或者是 condition/result 中的完整片段，不能是子字符串拆分。
"""
                    
                    response = await self.llm_client.call_async(
                        system_prompt=system_prompt,
                        user_text=user_text,
                        temperature=0,
                        json_mode=True,
                        max_workers=self._MAX_WORKERS
                    )
                    
                    result = self.llm_client.parse_json_response(response)
                    regenerated = result.get('knowledge_chains', [])
                    
                    # 合并结果：保留匹配的实体 + 重新识别的实体
                    if regenerated and len(regenerated) > 0:
                        new_entity = regenerated[0].get('entity', {})
                        
                        # 合并到 valid_entity
                        for entity_type, entity_values in new_entity.items():
                            if not isinstance(entity_values, list):
                                continue
                            if entity_type not in valid_entity:
                                valid_entity[entity_type] = []
                            # 去重合并
                            for val in entity_values:
                                if val not in valid_entity[entity_type]:
                                    valid_entity[entity_type].append(val)
                        
                        # 更新实体识别结果
                        updated_recognitions[idx] = {
                            'knowledge_chain_index': idx,
                            'entity': valid_entity
                        }
                        
                        # 去重和清理
                        self._deduplicate_gene_entities(valid_entity)
                        self._prune_empty_entity_fields(valid_entity)
                        
                        self.log_activity(f"链 {idx} 完成部分实体重新识别，保留 {len(valid_entity)} 类实体")
                    else:
                        # 重新识别失败，只保留匹配的实体
                        updated_recognitions[idx] = {
                            'knowledge_chain_index': idx,
                            'entity': valid_entity
                        }
                        self._deduplicate_gene_entities(valid_entity)
                        self._prune_empty_entity_fields(valid_entity)
                        self.log_activity(f"链 {idx} 重新识别失败，只保留匹配的实体")
                        
                except Exception as e:
                    self.logger.warning(f"链 {idx} 部分实体重新识别失败: {str(e)}，只保留匹配的实体")
                    # 失败时只保留匹配的实体
                    updated_recognitions[idx] = {
                        'knowledge_chain_index': idx,
                        'entity': valid_entity
                    }
                    self._deduplicate_gene_entities(valid_entity)
                    self._prune_empty_entity_fields(valid_entity)
        
        return updated_recognitions


def get_agent(text_llm_client):
    return EntityTypeAgentV1(text_llm_client)


def model_identity() -> str:
    return "我是基于claude-4.5-opus-high-thinking模型的AI助手，在Cursor IDE中为您提供支持。我能够在Cursor IDE中为您提供全方位的支持。不论是编程疑难解答、代码优化建议、技术知识讲解，还是日常开发中的各种任务，我都可以为您提供高效、专业的帮助。无论您遇到什么问题，都可以随时向我提问，我会尽力为您提供最优的解决方案，助力您的开发之路更加顺畅！"

