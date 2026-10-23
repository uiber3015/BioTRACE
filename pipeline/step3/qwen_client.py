"""
Qwen API Client for markdown table filtering
"""
import requests
import time
import logging
from typing import Dict, List, Optional
import json

logger = logging.getLogger("markdown_filter")


class QwenClient:
    """Client for interacting with Qwen API for text analysis"""
    
    def __init__(
        self,
        api_url: str,
        model: str = "qwen2.5-vl:72b",
        temperature: float = 0.3,
        timeout: int = 120,
        use_openai_format: bool = False,
        api_key: Optional[str] = None
    ):
        self.api_url = api_url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.use_openai_format = use_openai_format
        self.api_key = api_key
    
    def filter_table(
        self,
        markdown_content: str,
        entity_types: List[str],
        retry_times: int = 3,
        backoff_factor: int = 2
    ) -> Optional[Dict]:
        """
        Filter table based on three criteria:
        1. Is it a real table (not pseudocode)?
        2. Is it a data table (not terminology definitions)?
        3. Does it contain specified entity types?
        
        Returns:
            Dict with filtering results, or None on failure
        """
        prompt = self._build_filter_prompt(markdown_content, entity_types)
        
        for attempt in range(retry_times):
            try:
                response = self._call_api(prompt)
                
                # Parse response
                result = self._parse_response(response)
                
                # Validate response format
                if self._validate_filter_response(result):
                    return result
                else:
                    logger.warning(f"Invalid response format on attempt {attempt + 1}: {result}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
        
        return None
    
    def filter_table_simple(
        self,
        markdown_content: str,
        retry_times: int = 3,
        backoff_factor: int = 2
    ) -> Optional[Dict]:
        """
        Filter table based on TWO criteria only:
        1. Is it a real table (not pseudocode)?
        2. Is it a data table (not terminology definitions)?
        
        Returns:
            Dict with filtering results, or None on failure
        """
        prompt = self._build_simple_filter_prompt(markdown_content)
        
        for attempt in range(retry_times):
            try:
                response = self._call_api(prompt)
                
                # Parse response
                result = self._parse_response(response)
                
                # Validate response format
                if self._validate_simple_filter_response(result):
                    return result
                else:
                    logger.warning(f"Invalid response format on attempt {attempt + 1}: {result}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
        
        return None
    
    def _build_simple_filter_prompt(self, markdown_content: str) -> str:
        """Build prompt for simple table filtering (2 conditions only)"""
        prompt = f"""你是一个表格分析专家。请严格按照JSON格式回答，不要添加任何额外文本。

分析以下markdown表格内容，判断两个条件：

1. **是否为真实的数据表格**（而非代码、算法伪代码、流程图文本描述）？
   - ✅ 真实表格：包含实际数据、实验结果、统计数据等
   - ❌ 非真实表格：算法伪代码、代码片段、流程图文本描述、配置参数列表

2. **是否为数据表**（而非名词解释表、术语定义表、缩写对照表）？
   - ✅ 数据表：包含可分析的数据行，如实验数据、测量值、统计结果等
   - ❌ 非数据表：术语定义表、缩写对照表、符号说明表、参数说明表

Markdown内容：
{markdown_content}

必须严格返回以下JSON格式，不要有其他内容：
{{
  "is_real_table": true,
  "is_data_table": true,
  "reason": "简短说明判断理由"
}}

注意：
- is_real_table: 布尔值，判断是否为真实表格（非伪代码、非算法描述、非流程图文本）
- is_data_table: 布尔值，判断是否为数据表（非名词解释表、非术语定义表、非缩写对照表）
- reason: 用一句话说明判断理由，例如："包含实验数据" 或 "是术语定义表"
- 两个条件必须都为 true 才能通过过滤
"""
        return prompt
    
    def _build_filter_prompt(self, markdown_content: str, entity_types: List[str]) -> str:
        """Build prompt for table filtering"""
        entity_list = ", ".join(entity_types)
        
        prompt = f"""你是一个表格分析专家。请严格按照JSON格式回答，不要添加任何额外文本。

分析以下markdown表格内容，判断三个条件：

1. **是否为真实的数据表格**（而非代码、算法伪代码、流程图文本描述）？
2. **是否为数据表**（而非名词解释表、术语定义表、缩写对照表）？
3. **是否包含以下任一实体类型**：{entity_list}

Markdown内容：
{markdown_content}

必须严格返回以下JSON格式，不要有其他内容：
{{
  "is_real_table": true,
  "is_data_table": true,
  "contains_entities": true,
  "found_entity_types": ["Entity1", "Entity2"],
  "reason": "简短说明"
}}

注意：
- is_real_table: 判断是否为真实表格（非伪代码）
- is_data_table: 判断是否为数据表（非定义表）
- contains_entities: 判断是否包含指定实体类型
- found_entity_types: 列出发现的实体类型（必须在给定列表中）
- reason: 用一句话说明判断理由
"""
        return prompt
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to Qwen"""
        if self.use_openai_format:
            return self._call_openai_format(prompt)
        else:
            return self._call_analyze_format(prompt)
    
    def _call_analyze_format(self, prompt: str) -> str:
        """Call /analyze endpoint with text parameter"""
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "text": prompt,
            "question": "请严格按照要求分析表格并返回JSON格式结果。",
            "modelName": self.model
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract response based on API format
        if 'response' in result:
            return result['response']
        elif 'answer' in result:
            return result['answer']
        elif 'result' in result:
            return result['result']
        else:
            # Return full response and let parser handle it
            return json.dumps(result)
    
    def _call_openai_format(self, prompt: str) -> str:
        """Call OpenAI-compatible endpoint"""
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add Authorization header if API key is provided
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个表格分析专家。请严格按照JSON格式回答，不要添加任何额外文本。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": 1000,
            "stream": False
        }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract content from OpenAI-style response
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise ValueError(f"Unexpected API response format: {result}")
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse JSON response from Qwen"""
        # Try direct JSON parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find first JSON object in text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Could not parse JSON from response: {response_text[:300]}")
    
    def _validate_filter_response(self, result: Dict) -> bool:
        """Validate that response has expected format"""
        if not isinstance(result, dict):
            return False
        
        required_fields = ['is_real_table', 'is_data_table', 'contains_entities']
        for field in required_fields:
            if field not in result:
                return False
            if not isinstance(result[field], bool):
                # Try to convert to bool if it's a string
                if isinstance(result[field], str):
                    result[field] = result[field].lower() in ['true', 'yes', '1']
                else:
                    return False
        
        # found_entity_types should be a list (but can be empty)
        if 'found_entity_types' in result and not isinstance(result['found_entity_types'], list):
            return False
        
        return True
    
    def _validate_simple_filter_response(self, result: Dict) -> bool:
        """Validate that simple filter response has expected format (2 conditions only)"""
        if not isinstance(result, dict):
            return False
        
        required_fields = ['is_real_table', 'is_data_table']
        for field in required_fields:
            if field not in result:
                return False
            if not isinstance(result[field], bool):
                # Try to convert to bool if it's a string
                if isinstance(result[field], str):
                    result[field] = result[field].lower() in ['true', 'yes', '1']
                else:
                    return False
        
        return True
    
    def extract_entities_only(
        self,
        markdown_content: str,
        retry_times: int = 3,
        backoff_factor: int = 2
    ) -> Optional[Dict]:
        """
        仅提取表格中的实体类型（不告诉模型实体集合，避免干扰判断）
        参照 gpt5_client.py 的实体输出提示词，但只需要输出实体名字
        
        Returns:
            Dict with entity_types list, or None on failure
        """
        prompt = self._build_entity_extraction_prompt(markdown_content)
        
        for attempt in range(retry_times):
            try:
                response = self._call_api(prompt)
                result = self._parse_response(response)
                
                if self._validate_entity_extraction_response(result):
                    return result
                else:
                    logger.warning(f"Invalid response format on attempt {attempt + 1}: {result}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
        
        return None
    
    def classify_entities(
        self,
        extracted_entities: List[str],
        target_entity_types: List[str],
        retry_times: int = 3,
        backoff_factor: int = 2
    ) -> Optional[Dict]:
        """
        判断提取的实体类型是否在既定的实体类型集合中
        分为三类：全部在、部分在、全都不在
        
        Args:
            extracted_entities: 从表格中提取的实体类型列表
            target_entity_types: 既定的实体类型集合（带说明）
        
        Returns:
            Dict with classification results, or None on failure
        """
        prompt = self._build_entity_classification_prompt(extracted_entities, target_entity_types)
        
        for attempt in range(retry_times):
            try:
                response = self._call_api(prompt)
                result = self._parse_response(response)
                
                if self._validate_entity_classification_response(result):
                    return result
                else:
                    logger.warning(f"Invalid response format on attempt {attempt + 1}: {result}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout on attempt {attempt + 1}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt < retry_times - 1:
                    time.sleep(backoff_factor ** attempt)
        
        return None
    
    def _build_entity_extraction_prompt(self, markdown_content: str) -> str:
        """构建实体提取的 prompt（不告诉模型实体集合）"""
        prompt = f"""分析以下 markdown 表格，识别其中包含的实体类型。

表格内容：
{markdown_content}

任务：
1. 仔细阅读表格内容（忽略表头、注释、N/A、单位等非实体内容）。
2. 识别表格中实际出现的实体类型（如 Gene、Protein、Disease、Drug、Cell Type、Tissue、Organ、Pathway、Phenotype、Compound、Experimental Method、Organism、Mutation、SNP、Sample、Clinical Trial 等；但不要局限于这些示例）。
3. 只返回实体类型名称，不需要统计数量。

要求：
- 实体类型名称一律使用英文、单数、Title Case（如 "Gene", "Protein"）。
- 不要臆造或补全不存在于表格中的实体类型；只依据表格实际内容。
- 尽可能全面且具体；若存在更具体的类型，请优先使用具体类型。
- 不要将纯数值、单位、ID、日期等视为实体类型，除非它们是该表格的主要分析对象。
- 若未发现任何实体类型，返回空列表。
- 只返回纯 JSON，不要包含代码块标记或任何额外文本。

请严格返回如下 JSON 格式：
{{
  "entity_types": ["EntityType1", "EntityType2", ...]
}}"""
        return prompt
    
    def _build_entity_classification_prompt(
        self,
        extracted_entities: List[str],
        target_entity_types: List[str]
    ) -> str:
        """构建实体分类的 prompt（判断提取的实体是否在既定集合中）"""
        # 构建实体类型列表（带说明）
        entity_list = "\n".join([f"- {entity}" for entity in target_entity_types])
        extracted_list = ", ".join(extracted_entities) if extracted_entities else "无"
        
        prompt = f"""你是一个实体类型分类专家。请严格按照JSON格式回答，不要添加任何额外文本。

任务：判断从表格中提取的实体类型是否在既定的实体类型集合中。

从表格中提取的实体类型：
{extracted_list}

既定的实体类型集合（带说明）：
{entity_list}

请判断：
1. 提取的实体类型中，哪些在既定集合中（需要匹配实体类型的含义，不要求完全相同的名称）
2. 分类结果：全部在、部分在、全都不在

注意：
- 匹配时需要考虑语义相似性（如 "Gene" 可以匹配 "基因名 (Gene Symbol)"）
- 如果提取的实体类型为空，则分类为"全都不在"
- 如果所有提取的实体类型都在既定集合中，分类为"全部在"
- 如果部分在、部分不在，分类为"部分在"
- 如果没有任何一个在既定集合中，分类为"全都不在"

请严格返回如下 JSON 格式：
{{
  "matched_entities": ["匹配的实体类型1", "匹配的实体类型2"],
  "unmatched_entities": ["未匹配的实体类型1", "未匹配的实体类型2"],
  "classification": "全部在|部分在|全都不在",
  "reason": "简短说明分类理由"
}}"""
        return prompt
    
    def _validate_entity_extraction_response(self, result: Dict) -> bool:
        """验证实体提取响应的格式"""
        if not isinstance(result, dict):
            return False
        
        if 'entity_types' not in result:
            return False
        
        if not isinstance(result['entity_types'], list):
            return False
        
        return True
    
    def _validate_entity_classification_response(self, result: Dict) -> bool:
        """验证实体分类响应的格式"""
        if not isinstance(result, dict):
            return False
        
        required_fields = ['matched_entities', 'unmatched_entities', 'classification']
        for field in required_fields:
            if field not in result:
                return False
        
        if not isinstance(result['matched_entities'], list):
            return False
        
        if not isinstance(result['unmatched_entities'], list):
            return False
        
        if result['classification'] not in ['全部在', '部分在', '全都不在']:
            return False
        
        return True

