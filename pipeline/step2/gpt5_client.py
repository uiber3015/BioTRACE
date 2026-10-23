"""
GPT-5 API Client for entity type analysis
"""
import requests
import time
import logging
from typing import Dict, Optional
import json

logger = logging.getLogger("entity_sampling")


class GPT5Client:
    """Client for interacting with GPT-5 API"""
    
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "gpt-5",
        temperature: float = 0.7,
        max_tokens: int = 15000,  # Must accommodate reasoning (~6000) + output tokens
        timeout: int = 180  # Increased for gpt-5 reasoning time (was 120, now 180)
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    def analyze_table_entities(
        self,
        markdown_content: str,
        retry_times: int = 4,  # Increased from 3 to 4 for better resilience
        backoff_factor: int = 2
    ) -> Optional[Dict]:
        """
        Analyze table markdown content to identify entity types
        
        Returns:
            Dict with entity_types and entity_counts, or None on failure
        """
        prompt = self._build_entity_analysis_prompt(markdown_content)
        
        for attempt in range(retry_times):
            try:
                response = self._call_api(prompt)
                
                # Parse response
                result = self._parse_response(response)
                
                # Validate response format
                if self._validate_entity_response(result):
                    if attempt > 0:
                        logger.info(f"✓ Success on retry attempt {attempt + 1}")
                    return result
                else:
                    logger.warning(f"Invalid response format on attempt {attempt + 1}/{retry_times}")
                    
            except requests.exceptions.Timeout:
                wait_time = backoff_factor ** attempt
                logger.warning(f"⏱️ Request timeout on attempt {attempt + 1}/{retry_times} (waited {self.timeout}s)")
                if attempt < retry_times - 1:
                    logger.info(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
            except requests.exceptions.RequestException as e:
                wait_time = backoff_factor ** attempt
                logger.error(f"❌ Request error on attempt {attempt + 1}/{retry_times}: {e}")
                if attempt < retry_times - 1:
                    logger.info(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                wait_time = backoff_factor ** attempt
                logger.error(f"❌ Unexpected error on attempt {attempt + 1}/{retry_times}: {e}")
                if attempt < retry_times - 1:
                    logger.info(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        logger.error(f"💥 All {retry_times} attempts failed")
        return None
    
    def _build_entity_analysis_prompt(self, markdown_content: str) -> str:
        """Build prompt for entity type analysis"""
        prompt = f"""分析以下 markdown 表格，识别其中包含的实体类型。

表格内容：
{markdown_content}

任务：
1. 仔细阅读表格内容（忽略表头、注释、N/A、单位等非实体内容）。
2. 识别表格中实际出现的实体类型（如 Gene、Protein、Disease、Drug、Cell Type、Tissue、Organ、Pathway、Phenotype、Compound、Experimental Method、Organism、Mutation、SNP、Sample、Clinical Trial 等；但不要局限于这些示例）。
3. 统计每种实体类型在表格中出现的单元格数量（按单元格计数，而非去重后的实体项数量）。

要求：
- 实体类型名称一律使用英文、单数、Title Case（如 "Gene", "Protein"）。
- 不要臆造或补全不存在于表格中的实体类型；只依据表格实际内容。
- 尽可能全面且具体；若存在更具体的类型，请优先使用具体类型。
- 不要将纯数值、单位、ID、日期等视为实体类型，除非它们是该表格的主要分析对象。
- 若未发现任何实体类型，返回空列表和空对象。
- 只返回纯 JSON，不要包含代码块标记或任何额外文本。

请严格返回如下 JSON 格式：
{{
  "entity_types": ["EntityType1", "EntityType2", ...],
  "entity_counts": {{"EntityType1": 单元格数量, "EntityType2": 单元格数量, ...}}
}}"""
        return prompt
    
    def _call_api(self, prompt: str) -> str:
        """Make API call to GPT-5"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个实体识别专家，擅长从科学文献表格中识别各类实体类型。请严格按照JSON格式返回结果。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
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
            content = result['choices'][0]['message']['content']
            
            # Check if content is empty or None
            if not content:
                logger.warning(f"API returned empty content. Full response: {result}")
                raise ValueError("API returned empty content")
            
            return content
        else:
            raise ValueError(f"Unexpected API response format: {result}")
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse JSON response from GPT-5"""
        # Log the raw response for debugging
        logger.debug(f"Raw response text: {response_text[:500]}")
        
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
        
        # Try to find JSON object in text (greedy match)
        json_match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try to find any JSON-like structure
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Log the full response on failure
        logger.error(f"Failed to parse response. Full text: {response_text}")
        raise ValueError(f"Could not parse JSON from response: {response_text[:500]}")
    
    def _validate_entity_response(self, result: Dict) -> bool:
        """Validate that response has expected format"""
        if not isinstance(result, dict):
            return False
        
        if 'entity_types' not in result or 'entity_counts' not in result:
            return False
        
        if not isinstance(result['entity_types'], list):
            return False
        
        if not isinstance(result['entity_counts'], dict):
            return False
        
        return True

