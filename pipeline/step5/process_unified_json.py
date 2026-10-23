# -*- coding: utf-8 -*-
"""
统一JSON处理脚本 - 知识链生成 + 实体识别
使用 knowledge_chain.py 和 entity_type.py 的函数
适配统一JSON格式、14类实体类型、DeepSeek API
"""

import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

# ============================================================================
# 日志配置
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# API 配置
# ============================================================================

API_CONFIG = {
    "url": "https://api.chatanywhere.tech/v1/chat/completions",
    "model": "deepseek-v3",
    "api_key": "",  # set via env / config; do not hardcode
    "temperature": 0.0,
    "max_tokens": 4096,
    "timeout": 120,
    "max_workers": 20
}

# 14类实体类型定义（基因组领域）
ENTITY_TYPES_GENOMICS = {
    'gene_symbol': '基因名',
    'regulator': '调控因子',
    'target': '靶点',
    'functional_mechanism': '功能机制',
    'species': '物种',
    'disease': '疾病',
    'organ': '器官',
    'tissue': '组织',
    'cell': '细胞',
    'protein': '蛋白质',
    'compound': '化合物',
    'drug': '药物',
    'experimental_method': '实验方法',
    'phenotype': '表型'
}

# ============================================================================
# 数据处理器
# ============================================================================

class UnifiedJsonProcessor:
    """统一JSON处理器 - 知识链生成 + 实体识别"""
    
    def __init__(self, input_dir: str, output_dir: str, step3_dir: str):
        """
        初始化处理器
        
        Args:
            input_dir: 统一格式JSON输入目录
            output_dir: 输出目录
            step3_dir: Step3原始数据目录（用于获取上下文）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.step3_dir = Path(step3_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.api_config = API_CONFIG
        self.max_workers = self.api_config.get("max_workers", 20)
        self.semaphore = asyncio.Semaphore(self.max_workers)
        
        # 统计信息
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "skipped": 0
        }
    
    async def process_all(self):
        """处理所有JSON文件"""
        logger.info("=" * 80)
        logger.info("📊 统一JSON处理程序 - 知识链生成 + 实体识别")
        logger.info("=" * 80)
        
        # 获取所有JSON文件
        json_files = sorted(self.input_dir.glob("*.json"), 
                           key=lambda x: int(x.stem))
        
        logger.info(f"\n📂 输入目录: {self.input_dir}")
        logger.info(f"📄 找到 {len(json_files)} 个JSON文件")
        logger.info(f"🔄 并发数: {self.max_workers}")
        
        if not json_files:
            logger.warning("⚠️ 未找到JSON文件")
            return
        
        self.stats["total"] = len(json_files)
        
        # 并发处理
        logger.info("\n【开始处理】")
        tasks = [self._process_file(file_path) for file_path in json_files]
        
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            if i % 100 == 0:
                logger.info(f"  ✅ 已处理 {i}/{len(json_files)} 个文件")
        
        # 输出统计
        self._print_stats()
    
    async def _process_file(self, file_path: Path):
        """处理单个文件"""
        async with self.semaphore:
            try:
                # 读取统一格式JSON
                with open(file_path, 'r', encoding='utf-8') as f:
                    unified_data = json.load(f)
                
                table_title = unified_data.get('table_title', '')
                rows = unified_data.get('rows', [])
                
                if not rows:
                    self.stats["skipped"] += 1
                    return
                
                # 获取上下文（从Step3原始数据）
                description = self._get_context_from_step3(file_path.stem)
                
                # 第1步：生成知识链
                knowledge_chains = await self._generate_knowledge_chains(
                    table_title=table_title,
                    rows=rows,
                    description=description,
                    file_id=file_path.stem
                )
                
                if not knowledge_chains:
                    self.stats["skipped"] += 1
                    return
                
                # 第2步：识别实体
                entity_recognitions = await self._recognize_entities(
                    knowledge_chains=knowledge_chains,
                    table_title=table_title,
                    description=description,
                    file_id=file_path.stem
                )
                
                # 保存结果
                output_file = self.output_dir / f"{file_path.stem}_result.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "file_id": file_path.stem,
                        "table_title": table_title,
                        "knowledge_chains": knowledge_chains,
                        "entity_recognitions": entity_recognitions,
                        "processed_at": datetime.now().isoformat()
                    }, f, ensure_ascii=False, indent=2)
                
                self.stats["success"] += 1
            
            except Exception as e:
                logger.error(f"❌ 处理文件 {file_path.name} 失败: {str(e)}")
                self.stats["failed"] += 1
    
    def _get_context_from_step3(self, file_id: str) -> str:
        """
        从Step3原始数据获取上下文
        
        Args:
            file_id: 文件ID（如 "1", "2" 等）
        
        Returns:
            上下文描述文本
        """
        try:
            step3_file = self.step3_dir / f"{file_id}.json"
            if not step3_file.exists():
                return ""
            
            with open(step3_file, 'r', encoding='utf-8') as f:
                step3_data = json.load(f)
            
            # 提取caption作为主要上下文
            parsed_content = step3_data.get('parsed_content', {})
            caption = parsed_content.get('caption', '')
            
            # 如果有caption，返回caption；否则返回table_identifier
            if caption:
                return caption
            
            table_identifier = parsed_content.get('table_identifier', '')
            if table_identifier:
                return table_identifier
            
            return ""
        
        except Exception as e:
            logger.warning(f"⚠️ 获取Step3上下文失败 (file_id={file_id}): {str(e)}")
            return ""
    
    async def _generate_knowledge_chains(
        self,
        table_title: str,
        rows: List[Dict],
        description: str,
        file_id: str
    ) -> List[Dict]:
        """
        生成知识链
        
        Args:
            table_title: 表格标题
            rows: 表格行数据
            description: 表格描述
            file_id: 文件ID
        
        Returns:
            知识链列表
        """
        # 构建提示词
        system_prompt = self._build_knowledge_chain_system_prompt()
        user_prompt = self._build_knowledge_chain_user_prompt(
            table_title=table_title,
            rows=rows,
            description=description
        )
        
        # 调用API
        try:
            response = await self._call_api(system_prompt, user_prompt)
            chains = self._parse_knowledge_chains_response(response)
            return chains
        except Exception as e:
            logger.warning(f"⚠️ 文件 {file_id} 知识链生成失败: {str(e)}")
            return []
    
    async def _recognize_entities(
        self,
        knowledge_chains: List[Dict],
        table_title: str,
        description: str,
        file_id: str
    ) -> List[Dict]:
        """
        识别实体
        
        Args:
            knowledge_chains: 知识链列表
            table_title: 表格标题
            description: 表格描述
            file_id: 文件ID
        
        Returns:
            实体识别结果列表
        """
        # 构建提示词
        system_prompt = self._build_entity_system_prompt()
        user_prompt = self._build_entity_user_prompt(
            knowledge_chains=knowledge_chains,
            table_title=table_title,
            description=description
        )
        
        # 调用API
        try:
            response = await self._call_api(system_prompt, user_prompt)
            recognitions = self._parse_entity_response(response, knowledge_chains)
            return recognitions
        except Exception as e:
            logger.warning(f"⚠️ 文件 {file_id} 实体识别失败: {str(e)}")
            # 返回空实体结果
            return [{"chain_index": i, "entity": {}} for i in range(len(knowledge_chains))]
    
    def _build_knowledge_chain_system_prompt(self) -> str:
        """构建知识链生成系统提示词"""
        return """你是一个知识链生成专家。你必须且只能返回有效的JSON格式，不要有任何其他文字。

输出JSON结构：
{
  "knowledge_chains": [
    {"condition": "...", "result": "..."},
    {"condition": "...", "result": "..."}
  ]
}

任务：
- 从表格的每一行提取有意义的condition和result
- condition: 表格的列名和对应值（格式：键1:值1 | 键2:值2）
- result: 表格中的关键结果指标（格式：指标名:数值）
- 每行生成一条知识链
- 只返回JSON，不要任何其他文字"""
    
    def _build_knowledge_chain_user_prompt(
        self,
        table_title: str,
        rows: List[Dict],
        description: str
    ) -> str:
        """构建知识链生成用户提示词"""
        # 构建表格信息
        table_info = f"表格标题: {table_title}\n\n"
        
        # 显示前5行作为示例
        table_info += "表格数据（前5行）:\n"
        for i, row in enumerate(rows[:5], 1):
            table_info += f"行{i}: {json.dumps(row, ensure_ascii=False)}\n"
        
        if len(rows) > 5:
            table_info += f"... (共{len(rows)}行)\n"
        
        # 构建上下文
        context_info = ""
        if description:
            context_info = f"\n表格描述: {description}\n"
        
        prompt = f"""{table_info}{context_info}
请为表格中的每一行提取知识链（condition和result）。
返回JSON数组格式的知识链列表。"""
        
        return prompt
    
    def _build_entity_system_prompt(self) -> str:
        """构建实体识别系统提示词"""
        entity_types_str = "\n".join([
            f"{i+1}. {name_cn} ({name_en})"
            for i, (name_en, name_cn) in enumerate(ENTITY_TYPES_GENOMICS.items())
        ])
        
        return f"""你是一个生物医学领域的实体识别专家。你必须且只能返回有效的JSON格式，不要有任何其他文字。

**14类实体类型（基因组领域）**：
{entity_types_str}

输出JSON结构：
{{
  "knowledge_chains": [
    {{
      "chain_index": 0,
      "entity": {{
        "gene_symbol": ["..."],
        "regulator": ["..."],
        "target": ["..."],
        ...
      }}
    }}
  ]
}}

任务：
- 从知识链的condition和result中识别14类实体
- 每个实体类型对应一个字符串列表
- 去除重复实体
- 只返回JSON，不要任何其他文字"""
    
    def _build_entity_user_prompt(
        self,
        knowledge_chains: List[Dict],
        table_title: str,
        description: str
    ) -> str:
        """构建实体识别用户提示词"""
        # 构建知识链信息
        chains_list = []
        for idx, chain in enumerate(knowledge_chains):
            chains_list.append({
                "chain_index": idx,
                "condition": chain.get('condition', ''),
                "result": chain.get('result', '')
            })
        
        chains_json = json.dumps(chains_list, ensure_ascii=False, indent=2)
        
        # 构建上下文
        context_info = f"表格标题: {table_title}\n"
        if description:
            context_info += f"表格描述: {description}\n"
        
        prompt = f"""{context_info}
**知识链数据** ({len(knowledge_chains)}个):
{chains_json}

请为每个知识链识别14类实体类型。
关键规则：
1. 从condition和result中提取实体
2. condition按 | 分割，然后按 : 分割
3. 每个完整项作为一个整体识别，不要拆分
4. 去除重复，保持原有顺序
5. 不要包含空数组

返回JSON格式的识别结果。"""
        
        return prompt
    
    async def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """调用DeepSeek API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_config['api_key']}"
        }
        
        payload = {
            "model": self.api_config["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.api_config["temperature"],
            "max_tokens": self.api_config["max_tokens"]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_config["url"],
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(seconds=self.api_config["timeout"])
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result['choices'][0]['message']['content']
                else:
                    raise Exception(f"API返回错误: {resp.status}")
    
    def _parse_knowledge_chains_response(self, response: str) -> List[Dict]:
        """解析知识链API响应"""
        try:
            # 尝试直接解析JSON
            result = json.loads(response)
            if isinstance(result, dict) and 'knowledge_chains' in result:
                chains = result['knowledge_chains']
            elif isinstance(result, list):
                chains = result
            else:
                chains = []
            
            # 验证格式
            validated_chains = []
            for chain in chains:
                if isinstance(chain, dict):
                    condition = chain.get('condition', '')
                    result_val = chain.get('result', '')
                    if condition or result_val:
                        validated_chains.append({
                            'condition': condition,
                            'result': result_val
                        })
            
            return validated_chains
        
        except json.JSONDecodeError:
            # 尝试从响应中提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return self._parse_knowledge_chains_response(json_match.group())
                except:
                    pass
            return []
    
    def _parse_entity_response(self, response: str, knowledge_chains: List[Dict]) -> List[Dict]:
        """解析实体识别API响应"""
        try:
            # 尝试直接解析JSON
            result = json.loads(response)
            
            if isinstance(result, dict) and 'knowledge_chains' in result:
                recognitions = result['knowledge_chains']
            elif isinstance(result, list):
                recognitions = result
            else:
                recognitions = []
            
            # 合并原始知识链和识别结果
            merged = []
            for i, original in enumerate(knowledge_chains):
                merged_item = {
                    'chain_index': i,
                    'condition': original.get('condition', ''),
                    'result': original.get('result', ''),
                    'entity': {}
                }
                
                # 查找对应的识别结果
                if i < len(recognitions):
                    rec = recognitions[i]
                    if isinstance(rec, dict) and 'entity' in rec:
                        merged_item['entity'] = rec['entity']
                
                merged.append(merged_item)
            
            return merged
        
        except json.JSONDecodeError:
            # 返回空实体结果
            return [
                {
                    'chain_index': i,
                    'condition': chain.get('condition', ''),
                    'result': chain.get('result', ''),
                    'entity': {}
                }
                for i, chain in enumerate(knowledge_chains)
            ]
    
    def _print_stats(self):
        """打印统计信息"""
        logger.info("\n" + "=" * 80)
        logger.info("📈 处理统计")
        logger.info("=" * 80)
        logger.info(f"总文件数: {self.stats['total']}")
        logger.info(f"✅ 成功: {self.stats['success']}")
        logger.info(f"❌ 失败: {self.stats['failed']}")
        logger.info(f"⏭️  跳过: {self.stats['skipped']}")
        logger.info(f"\n📁 输出目录: {self.output_dir}")
        logger.info("=" * 80)

# ============================================================================
# 主程序
# ============================================================================

async def main():
    """主函数"""
    input_dir = "data/step5/unified_json_input"
    output_dir = "data/step5/processing_output"
    step3_dir = "data/step3/output/all_in"
    
    processor = UnifiedJsonProcessor(input_dir, output_dir, step3_dir)
    await processor.process_all()

if __name__ == "__main__":
    asyncio.run(main())
