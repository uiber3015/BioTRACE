"""
Step5 管线（V1 精简提示版）：
- 使用 01_knowledge_chain.py / 01_entity_type.py 中的精简提示 Agent
- 流程：统一 JSON -> 知识链 -> 实体识别 -> 合并输出 knowledge_mining
用法示例：
  python run_step5_pipeline_v1.py --num 25 --concurrency 20
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import aiohttp
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 通过 importlib 加载文件名以数字开头的模块
def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块 {mod_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

# 精简提示版 Agent（01_knowledge_chain.py / 01_entity_type.py）
step5_dir = Path(__file__).resolve().parent
kc_module = _load_module("kc_v1", step5_dir / "01_knowledge_chain.py")
et_module = _load_module("et_v1", step5_dir / "01_entity_type.py")
KnowledgeChainAgentV1 = kc_module.KnowledgeChainAgentV1
EntityTypeAgentV1 = et_module.EntityTypeAgentV1

from step3.utils import parse_json_response as parse_llm_json  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("step5_pipeline_v1")


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
        json_mode: bool = True,
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

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        trimmed = response.strip()
        # 处理 ```json ... ``` 或 ``` ... ``` 围栏，剥离前后围栏再解析
        if trimmed.startswith("```"):
            # 去掉起始 ```json / ``` 头
            trimmed = trimmed.lstrip("`")
            # lstrip 后可能以 "json" 开头，去掉可选的语言标签
            if trimmed.lower().startswith("json"):
                trimmed = trimmed[4:].lstrip()
            # 如果还包含结束围栏，则截断到结束围栏
            end_idx = trimmed.find("```")
            if end_idx != -1:
                trimmed = trimmed[:end_idx].strip()
        return parse_llm_json(trimmed)


def merge_kc_and_entities(
    knowledge_chains: List[Dict[str, Any]],
    entity_recognitions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entity_index: Dict[int, Dict[str, Any]] = {}
    for rec in entity_recognitions or []:
        idx = rec.get("knowledge_chain_index")
        if idx is None:
            continue
        entity_payload = rec.get("entity", {})
        if not isinstance(entity_payload, dict):
            entity_payload = {}
        entity_index[idx] = entity_payload

    knowledge_mining: List[Dict[str, Any]] = []
    for idx, chain in enumerate(knowledge_chains or []):
        kc_obj = {
            "condition": chain.get("condition", ""),
            "result": chain.get("result", ""),
        }
        entity_obj = entity_index.get(idx, {})
        knowledge_mining.append(
            {"knowledge_chain": kc_obj, "entity_recognition": entity_obj}
        )
    return knowledge_mining


async def process_one_file(
    file_path: Path,
    kc_agent: KnowledgeChainAgentV1,
    et_agent: EntityTypeAgentV1,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        file_id = file_path.stem
        out_path = output_dir / f"{file_id}_gene_entities.json"
        if out_path.exists():
            logger.info("已存在输出，跳过 %s", out_path.name)
            return
        try:
            with file_path.open("r", encoding="utf-8") as f:
                unified = json.load(f)
        except Exception as e:
            logger.error("读取统一 JSON 失败 (%s): %s", file_path.name, e)
            return

        rows = unified.get("rows") or unified.get("json_data")
        table_title = (unified.get("table_title") or "").strip()
        description = unified.get("context") or ""
        table_content = unified.get("table_content") or ""

        if not isinstance(rows, list) or not rows:
            logger.warning("%s 没有有效的 rows/json_data，跳过", file_path.name)
            return

        try:
            kc_result = await kc_agent.process_from_json({"rows": rows})
        except Exception as e:
            logger.error("知识链生成/解析失败，跳过 %s: %s", file_path.name, e)
            return

        knowledge_chains: List[Dict[str, Any]] = kc_result.get("knowledge_chains", [])
        if not knowledge_chains:
            logger.warning("%s 未生成任何知识链，跳过", file_path.name)
            return

        try:
            et_result = await et_agent.process(
                {
                    "knowledge_chains": knowledge_chains,
                    "description": description,
                    "table_title": table_title,
                    "table_content": table_content,
                }
            )
            entity_recognitions = et_result.get("entity_recognitions", [])

            # 验证实体是否完全匹配 condition/result，不匹配则重新生成
            entity_recognitions = await et_agent.validate_and_regenerate_entities(
                knowledge_chains,
                entity_recognitions
            )
        except Exception as e:
            logger.error("实体识别/验证失败，跳过 %s: %s", file_path.name, e)
            return

        knowledge_mining = merge_kc_and_entities(knowledge_chains, entity_recognitions)

        output = {
            "file_id": file_id,
            "table_title": table_title,
            "description": description,
            "rows": rows,
            "table_content": table_content,
            "knowledge_mining": knowledge_mining,
        }

        out_path = output_dir / f"{file_id}_gene_entities.json"
        try:
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("写出结果失败 (%s): %s", out_path, e)
            return

        logger.info("✅ 处理完成 %s → %s", file_path.name, out_path.name)


async def main(num_files: Optional[int] = None, concurrency: int = 20) -> None:
    step5_dir = Path(__file__).resolve().parent
    unified_dir = step5_dir / "unified_json_enriched"
    output_dir = step5_dir / "gene_entity_output_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    text_client = DeepseekTextClient()
    kc_agent = KnowledgeChainAgentV1(text_client)
    et_agent = EntityTypeAgentV1(text_client)

    all_files = sorted(unified_dir.glob("*.json"), key=lambda p: int(p.stem))
    if not all_files:
        logger.warning("在 %s 下未找到任何 JSON 文件", unified_dir)
        return
    files = all_files[: num_files or len(all_files)]

    logger.info("输入目录: %s, 输出目录: %s, 并发: %d, 文件数: %d", unified_dir, output_dir, concurrency, len(files))

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        process_one_file(f, kc_agent, et_agent, output_dir, semaphore) for f in files
    ]
    await asyncio.gather(*tasks)
    logger.info("全部完成，输出目录: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step5 V1 管线：统一 JSON → 知识链(精简提示) → 基因实体(精简提示)"
    )
    parser.add_argument("--num", "-n", type=int, default=None, help="处理的文件数量")
    parser.add_argument("--concurrency", "-c", type=int, default=20, help="并发数")
    args = parser.parse_args()

    if args.concurrency < 1:
        logger.error("并发数必须 >=1")
        sys.exit(1)
    if args.num is not None and args.num < 1:
        logger.error("文件数必须 >=1")
        sys.exit(1)

    asyncio.run(main(num_files=args.num, concurrency=args.concurrency))

