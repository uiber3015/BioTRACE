from typing import Dict, Any, List
import logging
# 兼容直接脚本加载：优先绝对导入，失败再尝试相对导入
try:
    from knowledge_chain import KnowledgeChainAgent
except ImportError:
    try:
        from .knowledge_chain import KnowledgeChainAgent
    except ImportError:
        import sys
        from pathlib import Path

        step5_dir = Path(__file__).resolve().parent
        if str(step5_dir) not in sys.path:
            sys.path.insert(0, str(step5_dir))
        from knowledge_chain import KnowledgeChainAgent


class KnowledgeChainAgentV1(KnowledgeChainAgent):
    """
    精简提示版的知识链识别智能体。
    逻辑沿用基类，只替换 system_prompt，使其更短但涵盖原有约束。
    """

    def __init__(self, text_llm_client):
        super().__init__(text_llm_client)
        self._logger = logging.getLogger("knowledge_chain_v1")

    async def process_from_json(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        json_data = input_data.get("rows") or input_data.get("json_data") or []
        if not json_data or not isinstance(json_data, list):
            return {"knowledge_chains": [], "total_count": 0}

        system_prompt = """
你是一个 JSON 生成器。你必须且只能返回**有效 JSON**，不能包含任何其他文字、注释或 Markdown。

统一输出格式：
{
  "knowledge_chains": [
    { "condition": "...", "result": "..." }
  ]
}

【任务描述】
- 输入：一个 JSON 数组，数组中每个对象表示表格的一行（可包含嵌套对象）。
- 输出：从每一行构造若干条 “知识链”（knowledge_chains）：
  - condition：由该行的所有非数值信息 + 分组标签 + 上下文信息组成。
  - result：每条链只对应一个数值字段，格式为 `指标名:数值`，或空字符串。

【核心约束】
1. 只输出 JSON：严格返回上述结构，不能有多余文字或代码块标记。
2. 遍历所有字段（含嵌套）：递归处理每个键值对，不得遗漏；为每个字段决定进入 condition 还是生成 result。
3. 数值字段全部产出 result（每条链仅 1 个 kv）：
   - 数值/含数字：纯数字、带单位/符号、百分号、科学计数等。
   - 一行有 N 个数值字段 → 至少 N 条非空 result 链；每条 result 只能有 1 个 `指标:数值`。
   - 若无数值字段，该行仍需至少 1 条链，result 可为空串。
4. 非数值字段全部进 condition：
   - 包括 null/NA/"-" 等缺失标记；condition 形式 `key:value | ...`，去重。
5. ID/编号优先视为条件：
   - 如 ID/Sample/Patient/Year/Contig/Pfam_ID/left/right/bp/kb/cM 等，除非语义即为结果列。

【指标名识别】
遇到数值字段 (k,v)：
1) 若 k 是指标字段（acc/F1/FLOPs/PR/Latency 等），用 k 作指标名。
2) 否则若父 key 是指标，用父 key 作指标名。
3) 否则优先用当前叶子 key 作指标名，避免 null:数值。
4) 实在无法判定再用 null:数值（尽量避免）。
5) 生物医学优先级：计数/载量/统计量字段（Case_count/n/n(%)/No_of_reads/depth/coverage/RPM/RPKM/FPKM/TPM/VL/viral load/VL_copies_ml/copies/mL/p/p-value/FDR/OR/RR/HR/Hazard ratio/Rate/Risk/prevalence/incidence 等）优先作为 result 指标名；样本ID/国家/分组/性别/年龄等进 condition。

【condition 规则】
1) 简单键值：非数值/非指标/非对象 → 直接 key:value 入 condition；清理参考文献标记与简单 HTML。
2) 嵌套/分组键：值为对象或值是数值但 key 非指标 → 视为分组标签，先推断分类 key（如 科目/年度），无法推断用 null:key；父键单独存 condition，子节点继承。
3) 继承去重：子节点 condition 继承所有父级片段；一条链内用 " | " 连接并去重。
4) 空键/空值：空键记为 null，空值/NA 记为 null，仍需入 condition。
5) **严格禁止多个冒号**：condition 中每个片段只能有一个冒号，格式严格为 key:value。**绝对禁止**生成 "父键:子键:值" 或 "A:B:C" 这种多个冒号的格式。嵌套结构必须拆分为多个独立片段，用 " | " 连接。
   - 错误示例：`"Frequency-based approach:Unadjusted p:0.41"` ❌
   - 正确格式：`"null:Frequency-based approach | Unadjusted p:0.41"` ✅
   - 错误示例：`"Top-1:acc:74.61"` ❌
   - 正确格式：`"null:Top-1 | acc:74.61"` ✅
6) **condition 与 result 严格分离（非常重要）**：
   - **condition 中绝对不能包含 result 的值**。condition 只包含上下文信息、分组标签、非数值字段等，不能包含数值字段的值。
   - 错误示例：`condition: "null:De_novo(meta)transcriptomics | null:Hindgut | P1:21.4"`, `result: "P1:21.4"` ❌（condition中包含了result的值）
   - 正确格式：`condition: "null:De_novo(meta)transcriptomics | Foregut:null | Midgut:null | null:Hindgut"`, `result: "P1:21.4"` ✅
   - 错误示例：`condition: "Characteristic:No. of individuals | null:Adult (≥18years)"`, `result: "No. of individuals:54"` ❌（result应该用数值字段的key）
   - 正确格式：`condition: "Characteristic:No. of individuals"`, `result: "Adult (≥18years):54"` ✅

【嵌套处理】
- 递归遍历；父为指标则子数值用父作指标名；父为分组则写入 condition 并继承。

【数据来源与清洗】
- 只用输入 JSON 数据，不补全新实体；null 原样写出。
- 清理参考文献标记（如 [70]）和简单 HTML 标签（如 <sup>），保留文本。

【结果数量】
- 每行至少 1 条链；多数值 → 多条有 result 的链；其他非数值信息要么并入这些链的 condition，要么另起链（result 可空），但不可丢失。

【few-shot 示例】（同上规则，不必复述）
1) [{"姓名":"张三","方法":"BERT","准确率":92.5,"F1":88.3}]
   → condition: "姓名:张三 | 方法:BERT", result: "准确率:92.5"（一条链）
   → condition: "姓名:张三 | 方法:BERT", result: "F1:88.3"（另一条链）
2) [{"姓名":"李四","成绩":{"2023年":{"语文":90,"数学":95}}}]
   → condition: "姓名:李四 | 年度:2023年 | 科目:语文", result: "成绩:90"（一条链）
   → condition: "姓名:李四 | 年度:2023年 | 科目:数学", result: "成绩:95"（另一条链）
3) {"Method":"SFP [70]","Top-1":{"acc":74.61,"acc↓":1.54},"FLOPs":"2.38G","PR":41.8}
   → condition: "Method:SFP | null:Top-1", result: "acc:74.61"（一条链）
   → condition: "Method:SFP | null:Top-1", result: "acc↓:1.54"（另一条链）
   → condition: "Method:SFP", result: "FLOPs:2.38G"（第三条链）
   → condition: "Method:SFP", result: "PR:41.8"（第四条链）
4) {"Model":"Hyper-Graph","Pretrain":"-","ACE2005":{"Precision":70.6,"Recall":70.4,"F1":70.5}}
   → condition: "Model:Hyper-Graph | Pretrain:- | null:ACE2005", result: "Precision:70.6"（一条链）
   → condition: "Model:Hyper-Graph | Pretrain:- | null:ACE2005", result: "Recall:70.4"（另一条链）
   → condition: "Model:Hyper-Graph | Pretrain:- | null:ACE2005", result: "F1:70.5"（第三条链）
5) {"Name":"DNMT1","Function":"Maintains DNA methylation<sup>6</sup>; ..."}
   → condition: "Name:DNMT1 | Function:Maintains DNA methylation; ...", result: ""（一条链，无数值字段）
6) 生物医学载量/计数/p 值示例：患者行包含 VL_copies_ml、CD4、p_value、No_of_reads 等 → 各自单条 result；ID/国家/分组在 condition。
   → condition: "Patient_ID:P001 | Country:Kenya | Group:Case", result: "VL_copies_ml:120000"（一条链）
   → condition: "Patient_ID:P001 | Country:Kenya | Group:Case", result: "CD4:350"（另一条链）
   → condition: "Patient_ID:P001 | Country:Kenya | Group:Case", result: "p_value:0.023"（第三条链）
   → condition: "Patient_ID:P001 | Country:Kenya | Group:Case", result: "No_of_reads:5600000"（第四条链）
7) {"Frequency-based approach":{"Unadjusted p":0.41,"encounter frequency":0.37},"Global approach":{"ologicaltests":"0.24"}}
   → condition: "null:Frequency-based approach", result: "Unadjusted p:0.41"（一条链）
   → condition: "null:Frequency-based approach", result: "encounter frequency:0.37"（另一条链）
   → condition: "null:Global approach", result: "ologicaltests:0.24"（第三条链）
   注意：嵌套的父键（如 "Frequency-based approach"、"Global approach"）无法推断分类时，必须使用 null:父键 格式，不能写成 "Frequency-based approach:Unadjusted p:0.41" 这种多个冒号格式。**condition中不能包含result的值**。
8) {"De_novo(meta)transcriptomics":{"Foregut":null,"Midgut":null,"Hindgut":{"P1":"21.4","P3":"29.3"}},"MG":{"Total_quality-trimmed_reads_(Million)":"20.6"}}
   → condition: "null:De_novo(meta)transcriptomics | Foregut:null | Midgut:null | null:Hindgut", result: "P1:21.4"（一条链）
   → condition: "null:De_novo(meta)transcriptomics | Foregut:null | Midgut:null | null:Hindgut", result: "P3:29.3"（另一条链）
   → condition: "null:MG", result: "Total_quality-trimmed_reads_(Million):20.6"（第三条链）
   注意：condition必须包含所有父级信息（包括null值），但不能包含result的值。
9) {"Characteristic":"No. of individuals","Adult (≥18years)":54,"Pediatric (<18years)":28}
   → condition: "Characteristic:No. of individuals", result: "Adult (≥18years):54"（一条链）
   → condition: "Characteristic:No. of individuals", result: "Pediatric (<18years):28"（另一条链）
   注意：result应该使用数值字段的key（如"Adult (≥18years)"），而不是"Characteristic"字段的值（如"No. of individuals"）。
"""

        json_formatted = ""
        try:
            import json

            json_formatted = json.dumps(json_data, ensure_ascii=False, indent=2)
        except Exception:
            pass

        user_text = f"JSON表格数据（{len(json_data)}行）：\n{json_formatted}\n请按规则返回 knowledge_chains 纯JSON。"

        response = await self.llm_client.call_async(
            system_prompt=system_prompt,
            user_text=user_text,
            temperature=0,
            json_mode=True,
        )
        try:
            result = self.llm_client.parse_json_response(response)
        except Exception as e:
            # 记录原始返回，便于排查非 JSON 输出问题
            preview = response[:500].replace("\n", "\\n")
            self._logger.error("LLM 返回非 JSON（行数=%d）：%s | response_preview=%s", len(json_data), e, preview)
            raise
        chains = result.get("knowledge_chains", [])
        normalized = self._normalize_knowledge_chains(chains)
        return {"knowledge_chains": normalized, "total_count": len(normalized)}


def get_agent(text_llm_client):
    return KnowledgeChainAgentV1(text_llm_client)


def model_identity() -> str:
    return "我是基于claude-4.5-opus-high-thinking模型的AI助手，在Cursor IDE中为您提供支持。我能够在Cursor IDE中为您提供全方位的支持。不论是编程疑难解答、代码优化建议、技术知识讲解，还是日常开发中的各种任务，我都可以为您提供高效、专业的帮助。无论您遇到什么问题，都可以随时向我提问，我会尽力为您提供最优的解决方案，助力您的开发之路更加顺畅！"

