#!/usr/bin/env python3
"""
从 all_in 中查找未出现在 unified_json_enriched 的文件，直接基于原始 table_content 生成
与 gene_entity_output_v1 相同格式的输出（知识链 + 实体识别），rows 为空。

流程：
1. 找出缺失的文件（all_in - unified_json_enriched）
2. 使用 DeepSeek 基于 caption + table_content 生成知识链（覆盖数值字段、严格1 kv/链）
3. 使用 EntityTypeAgentV1 进行实体识别与验证
4. 输出到指定目录（默认 unified_json_enriched），格式与 gene_entity_output_v1 一致

使用示例：
  python generate_missing_from_all_in.py --concurrency 10
  python generate_missing_from_all_in.py --output-dir gene_entity_output_missing
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp

# 项目路径
STEP5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STEP5_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# 确保 step3 在路径中，优先使用统一的 parse_json_response
STEP3_DIR = PROJECT_ROOT / "step3"
if str(STEP3_DIR) not in sys.path:
    sys.path.insert(0, str(STEP3_DIR))

# 尝试使用 step3.utils 的鲁棒解析器，若失败则回退简单解析
try:
    from step3.utils import parse_json_response as parse_llm_json  # type: ignore
except Exception:
    import json as _json
    def parse_llm_json(txt: str):
        trimmed = txt.strip()
        if trimmed.startswith("```"):
            trimmed = trimmed.lstrip("`")
            if trimmed.lower().startswith("json"):
                trimmed = trimmed[4:].lstrip()
            end_idx = trimmed.find("```")
            if end_idx != -1:
                trimmed = trimmed[:end_idx].strip()
        return _json.loads(trimmed)

# 导入 V1 Agent（精简提示版实体识别，文件名以数字开头需用 importlib）
import importlib.util
_et_path = STEP5_DIR / "01_entity_type.py"
_spec = importlib.util.spec_from_file_location("et_v1", _et_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载模块 et_v1 from {_et_path}")
_et_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_et_module)  # type: ignore
EntityTypeAgentV1 = _et_module.EntityTypeAgentV1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_missing")


class DeepseekTextClient:
    def __init__(
        self,
        api_url: str = "https://api.chatanywhere.tech/v1/chat/completions",
        api_key: str = "",  # set via env / CLI; do not hardcode
        model: str = "deepseek-v3.2",
        timeout: int = 240,
        max_retries: int = 3,
        retry_backoff: int = 5,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    async def call_async(
        self,
        system_prompt: str,
        user_text: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": float(temperature),
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(self.api_url, headers=headers, json=payload) as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            raise RuntimeError(
                                f"ChatAnywhere API 错误: status={resp.status}, body={text[:300]}"
                            )
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
            except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError) as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_backoff * (2 ** attempt))
                    continue
                raise
            except Exception as e:
                last_err = e
                raise
        if last_err:
            raise last_err
        raise RuntimeError("未知错误")

    @staticmethod
    def parse_json_response(response: str) -> Dict[str, Any]:
        trimmed = response.strip()
        # 优先用更鲁棒的解析器（可剥离围栏/额外文本）
        try:
            return parse_llm_json(trimmed)
        except Exception:
            pass
        # 回退：手动剥离代码围栏
        if trimmed.startswith("```"):
            trimmed = trimmed.lstrip("`")
            if trimmed.lower().startswith("json"):
                trimmed = trimmed[4:].lstrip()
            end_idx = trimmed.find("```")
            if end_idx != -1:
                trimmed = trimmed[:end_idx].strip()
        try:
            return json.loads(trimmed)
        except Exception as e:
            raise RuntimeError(f"解析 JSON 失败: {e}; content preview: {trimmed[:200]}")


# ---- 知识链生成（基于 table_content） ----

# 表格 Markdown 提示词（对齐 01_knowledge_chain 规则，输入仅 table_content）
SYSTEM_PROMPT = """
你是一个 JSON 生成器。你必须且只能返回**合法 JSON**，不能包含任何其他文字、注释或 Markdown 代码块。
输入：表格的 Markdown 文本（仅 Table Content）。请遍历表格的每一行，生成知识链。

输出格式（唯一合法结构）：
{
  "knowledge_chains": [
    { "condition": "...", "result": "..." }
  ]
}

核心约束（与 01_knowledge_chain 保持一致）：
1) 遍历所有字段（含嵌套/分组）：每个字段要么进 condition，要么生成 result。
2) condition：仅含非数值/分组/上下文信息；使用 key:value 片段，用 " | " 连接；片段内只能有 1 个冒号；禁止多个冒号（如 A:B:C）；禁止把 result 的值放入 condition。
3) result：每条链只对应 1 个数值字段，格式 "指标:数值"。一行若有 N 个数值字段 → 至少 N 条非空 result 链；行无数值时，允许 result 为空串。
4) 必须覆盖所有数值字段（含带单位/百分号/科学计数等）；ID/编号字段优先进 condition，除非语义即为结果列。
5) condition 与 result 分离：condition 不含数值；result 不含多个 kv（禁止 "A:1 | B:2"）。
6) 若无法抽取，返回 {"knowledge_chains": []}。

few-shot 示例（仅供格式参考，不要复述示例文本）：

示例 1（多基因行，score 为数值）：
Table Content
Genelocus| Genename| Reference| Predictedscore
---|---|---|---
AT1G49860| AtGSTF14,glutathione s-transferase(classphi)14| N.A.| 0.64
AT2G46370| JAR1,JasmonateResistant1| [40, 41]| 0.56
AT1G47480| Alpha/beta-hydrolases superfamily protein| N.A.| 0.56
输出：
{
  "knowledge_chains": [
    {"condition": "Genelocus:AT1G49860 | Genename:AtGSTF14,glutathione s-transferase(classphi)14 | Reference:N.A.", "result": "Predictedscore:0.64"},
    {"condition": "Genelocus:AT2G46370 | Genename:JAR1,JasmonateResistant1 | Reference:[40, 41]", "result": "Predictedscore:0.56"},
    {"condition": "Genelocus:AT1G47480 | Genename:Alpha/beta-hydrolases superfamily protein | Reference:N.A.", "result": "Predictedscore:0.56"}
  ]
}

示例 2（多数值列，一行拆多链）：
Table Content
Cadmium (μg/l)| Length (mm)| Lipid-Ovary Indexa| NumberofClutchesa| Totalneonatesa| Per Capita Birth Rate (b)a
---|---|---|---|---|---
0| 2.90±0.11a| 3.83±1.50a| 5.50±0.55a| 118.86±29.94a| 5.37±0.89a
0.25| 2.54±0.10b| 1.57±1.13b| 4.63±0.52a| 65.13±16.60b| 2.88±0.66b
输出：
{
  "knowledge_chains": [
    {"condition": "Cadmium (μg/l):0", "result": "Length (mm):2.90±0.11a"},
    {"condition": "Cadmium (μg/l):0", "result": "Lipid-Ovary Indexa:3.83±1.50a"},
    {"condition": "Cadmium (μg/l):0", "result": "NumberofClutchesa:5.50±0.55a"},
    {"condition": "Cadmium (μg/l):0", "result": "Totalneonatesa:118.86±29.94a"},
    {"condition": "Cadmium (μg/l):0", "result": "Per Capita Birth Rate (b)a:5.37±0.89a"},
    {"condition": "Cadmium (μg/l):0.25", "result": "Length (mm):2.54±0.10b"},
    {"condition": "Cadmium (μg/l):0.25", "result": "Lipid-Ovary Indexa:1.57±1.13b"},
    {"condition": "Cadmium (μg/l):0.25", "result": "NumberofClutchesa:4.63±0.52a"},
    {"condition": "Cadmium (μg/l):0.25", "result": "Totalneonatesa:65.13±16.60b"},
    {"condition": "Cadmium (μg/l):0.25", "result": "Per Capita Birth Rate (b)a:2.88±0.66b"}
  ]
}

请按上述约束与格式返回 JSON，勿输出任何其他文本。
"""


def build_user_text(caption: str, table_content: str) -> str:
    return f"""## Caption\n{caption}\n\n## Table Content\n{table_content}"""


async def generate_knowledge_chains(text_client: DeepseekTextClient, caption: str, table_content: str) -> List[Dict[str, Any]]:
    user_text = build_user_text(caption, table_content)
    resp = await text_client.call_async(
        system_prompt=SYSTEM_PROMPT,
        user_text=user_text,
        temperature=0.0,
    )
    parsed = text_client.parse_json_response(resp)
    chains = parsed.get("knowledge_chains", [])
    if not isinstance(chains, list):
        return []
    return chains


async def process_one_file(
    file_path: Path,
    output_dir: Path,
    text_client: DeepseekTextClient,
    et_agent: EntityTypeAgentV1,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        file_id = file_path.stem
        # 输出文件名：保持与 unified_json_enriched 一致（不加后缀）
        out_path = output_dir / f"{file_id}.json"
        if out_path.exists():
            logger.info("已存在输出，跳过 %s", out_path.name)
            return
        try:
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error("读取 all_in 失败 (%s): %s", file_path.name, e)
            return

        parsed = data.get("parsed_content") or {}
        caption = parsed.get("caption") or ""
        table_content = parsed.get("table_content") or ""
        if not table_content.strip():
            logger.warning("%s 无 table_content，跳过", file_path.name)
            return

        # 1) 生成知识链
        try:
            knowledge_chains = await generate_knowledge_chains(text_client, caption, table_content)
        except Exception as e:
            logger.error("知识链生成失败 %s: %s", file_path.name, e)
            return
        if not knowledge_chains:
            logger.warning("%s 未生成任何知识链，跳过", file_path.name)
            return

        # 2) 实体识别
        try:
            et_result = await et_agent.process(
                {
                    "knowledge_chains": knowledge_chains,
                    "description": parsed.get("context") or "",
                    "table_title": parsed.get("caption") or "",
                    "table_content": table_content,
                }
            )
            entity_recognitions = et_result.get("entity_recognitions", [])
            # 验证实体匹配
            entity_recognitions = await et_agent.validate_and_regenerate_entities(
                knowledge_chains,
                entity_recognitions
            )
        except Exception as e:
            logger.error("实体识别失败 %s: %s", file_path.name, e)
            return

        # 3) 合并为与 gene_entity_output_v1 一致的格式（rows 留空）
        knowledge_mining: List[Dict[str, Any]] = []
        for idx, chain in enumerate(knowledge_chains):
            er = {}
            for rec in entity_recognitions or []:
                if rec.get("knowledge_chain_index") == idx:
                    er = rec.get("entity", {}) if isinstance(rec.get("entity", {}), dict) else {}
                    break
            knowledge_mining.append({
                "knowledge_chain": {
                    "condition": chain.get("condition", ""),
                    "result": chain.get("result", ""),
                },
                "entity_recognition": er,
            })

        output = {
            "file_id": file_id,
            "table_title": parsed.get("caption") or "",
            "description": parsed.get("context") or "",
            "rows": [],  # rows 缺失，置为空
            "table_content": table_content,
            "knowledge_mining": knowledge_mining,
        }

        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("写出结果失败 (%s): %s", out_path, e)
            return

        logger.info("✅ 处理完成 %s → %s", file_path.name, out_path.name)


def find_missing_files(all_in_dir: Path, enriched_dir: Path) -> List[Path]:
    def _to_key(stem: str):
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    all_ids = {p.stem for p in all_in_dir.glob("*.json")}
    enriched_ids = {p.stem for p in enriched_dir.glob("*.json")}
    missing_ids = all_ids - enriched_ids
    missing_files = [all_in_dir / f"{mid}.json" for mid in sorted(missing_ids, key=_to_key)]
    return missing_files


async def main(concurrency: int = 10, output_dir: Optional[Path] = None) -> None:
    all_in_dir = STEP5_DIR / "all_in"
    enriched_dir = STEP5_DIR / "unified_json_enriched"
    out_dir = output_dir or enriched_dir  # 默认直接写回 unified_json_enriched
    out_dir.mkdir(parents=True, exist_ok=True)

    if not all_in_dir.exists():
        logger.error("all_in 目录不存在: %s", all_in_dir)
        return
    if not enriched_dir.exists():
        logger.error("unified_json_enriched 目录不存在: %s", enriched_dir)
        return

    missing_files = find_missing_files(all_in_dir, enriched_dir)
    logger.info("在 all_in 中共 %d 个文件，缺失 %d 个需要处理", len(list(all_in_dir.glob('*.json'))), len(missing_files))
    if not missing_files:
        logger.info("没有缺失的文件需要处理")
        return

    text_client = DeepseekTextClient()
    et_agent = EntityTypeAgentV1(text_client)

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        process_one_file(f, out_dir, text_client, et_agent, semaphore)
        for f in missing_files
    ]
    await asyncio.gather(*tasks)
    logger.info("全部缺失文件处理完成，输出目录: %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理 all_in 中缺失的文件，直接用 table_content 生成知识链")
    parser.add_argument("--concurrency", "-c", type=int, default=10, help="并发数 (默认 10)")
    parser.add_argument("--output-dir", "-o", type=str, help="输出目录（默认写回 unified_json_enriched）")
    args = parser.parse_args()

    if args.concurrency < 1:
        print("并发数必须 >=1")
        sys.exit(1)

    out_dir = Path(args.output_dir) if args.output_dir else None
    asyncio.run(main(concurrency=args.concurrency, output_dir=out_dir))
