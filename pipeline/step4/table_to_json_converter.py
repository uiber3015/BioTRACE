"""使用Qwen3-32B将表格内容转换为结构化JSON，结合14种生物医学实体类型约束"""
import json
import os
import asyncio
import aiohttp
import time
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from entity_types import ENTITY_TYPE_LIST, get_entity_types_description

class TableToJSONConverter:
    """表格转JSON转换器 - 使用Qwen3-32B模型"""
    
    def __init__(self, api_url=os.environ.get("QWEN_API_URL", ""), 
                 model_name="qwen3-32b", concurrency=5):
        self.api_url = api_url
        self.model_name = model_name
        self.concurrency = concurrency
        self.entity_types = ENTITY_TYPE_LIST
        self.entity_desc = get_entity_types_description()
        
    def build_system_prompt(self) -> str:
        """构建系统提示词"""
        return f"""你是一个专业的生物医学表格数据结构化专家。你的任务是将表格内容转换为结构化JSON，并识别其中的生物医学实体。

## 核心任务
将用户提供的表格内容转换为特定格式的JSON，同时识别并标注生物医学实体。

## 14种允许的生物医学实体类型
{self.entity_desc}

## 关键指令与约束

### 1. 表格结构识别
- 识别表头层级：表头可能跨越多行或多列
- 父子关系：如果表头跨越子表头，识别为层级关系
- 叶子节点：最底层表头，直接对应数据值

### 2. 数据映射规则
- 每一行数据转换为JSON数组中的一个独立对象
- JSON对象的key必须是表头文本
- JSON对象的value必须是该行对应的数据

### 3. 行完整性约束
- **严格一行一对象原则**：每个JSON对象只对应一行数据
- **绝不跨行合并**：不同行的内容不能放入同一个对象
- 如果数据位于下一行，创建新的JSON对象

### 4. 数据类型推断
- 数字值输出为Number（不带引号）
- 文本值输出为String（带引号）
- 空值使用null

### 5. 生物医学实体识别
- **只能使用上述14种实体类型**，严禁自创类型
- 实体值必须是表格中实际出现的文本
- 不要推断或补全表格中没有的内容
- 实体值保持原文形式（大小写、空格、标点）

## 输出格式
```json
{{
  "table_title": "表格标题或null",
  "rows": [
    {{
      "列名1": "值1",
      "列名2": 123,
      "列名3": "值3"
    }}
  ],j
  "entities": {{
    "实体类型1": ["实体值1", "实体值2"],
    "实体类型2": ["实体值3"]
  }}
}}
```

## 重要约束
- 只返回纯JSON，不要添加解释、注释或markdown标记
- table_title必须存在，无标题则设为null
- rows必须是数组，每个元素对应一行数据
- entities只包含在表格中实际出现的生物医学实体
- 实体类型必须从14种中选择，不能自创

现在请处理用户提供的表格内容。"""

    def build_user_prompt(self, table_content: str, caption: str = "") -> str:
        """构建用户提示词"""
        prompt = "请将以下表格内容转换为结构化JSON：\n\n"
        
        if caption and caption.strip():
            prompt += f"## 表格标题\n{caption}\n\n"
        
        prompt += f"## 表格内容\n{table_content}\n\n"
        prompt += "请严格按照要求输出JSON格式，识别其中的生物医学实体。"
        
        return prompt

    async def call_qwen_api(self, session: aiohttp.ClientSession, 
                           table_content: str, caption: str = "") -> Dict[str, Any]:
        """调用Qwen API"""
        system_prompt = self.build_system_prompt()
        user_prompt = self.build_user_prompt(table_content, caption)
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.1,
            "top_p": 0.9
        }
        
        try:
            async with session.post(self.api_url, json=payload, timeout=300) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return {"success": True, "content": content}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def clean_and_parse_json(self, content: str) -> Dict[str, Any]:
        """清理并解析JSON"""
        try:
            # 移除可能的markdown代码块
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # 解析JSON
            parsed = json.loads(content)
            return {"success": True, "data": parsed}
            
        except json.JSONDecodeError as e:
            return {"success": False, "error": f"JSON解析失败: {str(e)}", "content": content}

    async def process_single_table(self, session: aiohttp.ClientSession, 
                                 table_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单个表格"""
        table_id = table_data.get('number', 'unknown')
        parsed_content = table_data.get('parsed_content', {})
        table_content = parsed_content.get('table_content', '')
        caption = parsed_content.get('caption', '')
        
        if not table_content:
            return {
                'table_id': table_id,
                'status': 'failed',
                'error': '表格内容为空'
            }
        
        print(f"🔄 处理表格 {table_id}...")
        
        # 调用API
        api_result = await self.call_qwen_api(session, table_content, caption)
        
        if not api_result["success"]:
            return {
                'table_id': table_id,
                'status': 'failed',
                'error': f"API调用失败: {api_result['error']}"
            }
        
        # 解析JSON
        parse_result = self.clean_and_parse_json(api_result["content"])
        
        if not parse_result["success"]:
            return {
                'table_id': table_id,
                'status': 'failed',
                'error': f"JSON解析失败: {parse_result['error']}",
                'raw_content': api_result["content"][:500]
            }
        
        structured_data = parse_result["data"]
        
        # 验证结果
        validation_result = self.validate_result(structured_data)
        
        return {
            'table_id': table_id,
            'status': 'success' if validation_result["valid"] else 'validation_failed',
            'structured_data': structured_data,
            'validation_errors': validation_result.get("errors", []),
            'metadata': {
                'table_content_length': len(table_content),
                'caption_length': len(caption),
                'rows_count': len(structured_data.get('rows', [])),
                'entities_count': len(structured_data.get('entities', {}))
            }
        }

    def validate_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """验证结果格式"""
        errors = []
        
        # 检查必需字段
        if 'table_title' not in data:
            errors.append("缺少table_title字段")
        if 'rows' not in data:
            errors.append("缺少rows字段")
        elif not isinstance(data['rows'], list):
            errors.append("rows必须是数组")
        
        # 检查实体类型
        if 'entities' in data:
            entities = data['entities']
            if isinstance(entities, dict):
                for entity_type in entities.keys():
                    if entity_type not in self.entity_types:
                        errors.append(f"未知实体类型: {entity_type}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    async def process_all_tables(self, all_in_dir: str, output_dir: str) -> Dict[str, Any]:
        """处理所有表格"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载所有表格数据
        table_files = list(Path(all_in_dir).glob("*.json"))
        table_data_list = []
        
        for json_file in table_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    table_data_list.append(data)
            except Exception as e:
                print(f"❌ 加载文件失败 {json_file}: {e}")
        
        print(f"📊 总共找到 {len(table_data_list)} 个表格")
        
        # 创建异步会话
        connector = aiohttp.TCPConnector(limit=self.concurrency)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # 创建信号量控制并发
            semaphore = asyncio.Semaphore(self.concurrency)
            
            async def process_with_semaphore(table_data):
                async with semaphore:
                    return await self.process_single_table(session, table_data)
            
            # 并发处理所有表格
            start_time = time.time()
            results = await asyncio.gather(
                *[process_with_semaphore(data) for data in table_data_list],
                return_exceptions=True
            )
            end_time = time.time()
        
        # 统计结果
        successful = []
        failed = []
        validation_failed = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({
                    'table_id': table_data_list[i].get('number', f'table_{i}'),
                    'error': str(result)
                })
            elif result['status'] == 'success':
                successful.append(result)
                # 保存成功的结果
                output_file = os.path.join(output_dir, f"structured_{result['table_id']}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            elif result['status'] == 'validation_failed':
                validation_failed.append(result)
            else:
                failed.append(result)
        
        # 生成报告
        report = {
            'total_tables': len(table_data_list),
            'successful': len(successful),
            'validation_failed': len(validation_failed),
            'failed': len(failed),
            'success_rate': len(successful) / len(table_data_list) * 100 if table_data_list else 0,
            'processing_time': end_time - start_time,
            'avg_time_per_table': (end_time - start_time) / len(table_data_list) if table_data_list else 0,
            'successful_results': successful,
            'validation_failed_results': validation_failed,
            'failed_results': failed
        }
        
        # 保存报告
        report_file = os.path.join(output_dir, 'conversion_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 处理完成:")
        print(f"  总表格数: {report['total_tables']}")
        print(f"  成功: {report['successful']} ({report['success_rate']:.1f}%)")
        print(f"  验证失败: {report['validation_failed']}")
        print(f"  失败: {report['failed']}")
        print(f"  处理时间: {report['processing_time']:.1f}秒")
        print(f"  平均每表: {report['avg_time_per_table']:.1f}秒")
        print(f"  报告已保存: {report_file}")
        
        return report

async def main():
    """主函数"""
    # 配置参数
    all_in_dir = "data/step3/output/all_in"
    output_dir = "output/structured_json_tables"
    api_url = os.environ.get("QWEN_API_URL", "")
    model_name = "qwen3-32b"
    concurrency = 5
    
    print("🚀 开始使用Qwen3-32B转换表格为结构化JSON...")
    print(f"📡 API: {api_url}")
    print(f"🤖 模型: {model_name}")
    print(f"⚙️ 并发数: {concurrency}")
    
    # 创建转换器
    converter = TableToJSONConverter(
        api_url=api_url,
        model_name=model_name,
        concurrency=concurrency
    )
    
    # 处理所有表格
    report = await converter.process_all_tables(all_in_dir, output_dir)
    
    print("\n🎉 转换完成!")
    return report

if __name__ == "__main__":
    asyncio.run(main())
