"""
Utility functions for Step 2: Entity Sampling
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any
import random


def setup_logging(log_dir: str, log_file: str = "entity_sampling.log") -> logging.Logger:
    """Setup logging configuration"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("entity_sampling")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_path / log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def ensure_dir(directory: str) -> Path:
    """Create directory if it doesn't exist"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(file_path: str) -> Any:
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """Save data to JSON file"""
    ensure_dir(Path(file_path).parent)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_markdown(file_path: str) -> str:
    """Load markdown file content"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_table_content(markdown_content: str) -> str:
    """
    Extract key sections from markdown for analysis
    Returns Caption + Table Content sections
    """
    lines = markdown_content.split('\n')
    extracted = []
    capture = False
    
    for line in lines:
        # Start capturing at Caption or Table Content
        if line.strip().startswith('## Caption') or line.strip().startswith('## Table Content'):
            capture = True
        # Stop capturing at Footnotes or Mentioned in Text
        elif line.strip().startswith('## Footnotes') or line.strip().startswith('## Mentioned in Text'):
            capture = False
        
        if capture:
            extracted.append(line)
    
    return '\n'.join(extracted)


def sample_markdown_files(input_dir: str, sample_size: int, seed: int = 42) -> List[str]:
    """
    Randomly sample markdown files from input directory
    Returns list of file paths
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Get all markdown files
    all_files = list(input_path.glob("*.md"))
    
    if len(all_files) == 0:
        raise ValueError(f"No markdown files found in {input_dir}")
    
    # Sample files
    random.seed(seed)
    sample_size = min(sample_size, len(all_files))
    sampled_files = random.sample(all_files, sample_size)
    
    return [str(f) for f in sampled_files]


def extract_table_id_from_filename(file_path: str) -> str:
    """Extract table ID from markdown filename"""
    filename = Path(file_path).stem
    # Assuming format: table_markdown_<id>.md or similar
    match = re.search(r'(\d+)', filename)
    if match:
        return f"table_{match.group(1)}"
    return filename


def parse_json_response(response_text: str) -> Dict:
    """
    Parse JSON from GPT response with flexible extraction
    Handles cases where JSON is wrapped in markdown code blocks or has extra text
    """
    # Try direct JSON parsing first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object in text
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not parse JSON from response: {response_text[:200]}")


def calculate_percentile(
    entity_stats: Dict[str, Dict],
    total_tables: int,
    percentile: float = 80
) -> Dict:
    """
    Calculate percentile threshold for entity types
    
    Args:
        entity_stats: Dict with entity types as keys and stats as values
        total_tables: Total number of tables sampled
        percentile: Target percentile (default 80)
    
    Returns:
        Dict with percentile_entities and threshold info
    """
    # Sort entities by table_count descending
    sorted_entities = sorted(
        entity_stats.items(),
        key=lambda x: x[1]['table_count'],
        reverse=True
    )
    
    # Calculate cumulative coverage
    threshold_count = int(total_tables * (percentile / 100))
    selected_entities = []
    cumulative_tables = set()
    
    for entity_type, stats in sorted_entities:
        # Add entity if we haven't reached threshold
        # Note: This is a simplified version - actual implementation would need
        # to track which tables contain which entities
        if len(cumulative_tables) < threshold_count:
            selected_entities.append(entity_type)
            # Approximate cumulative count
            cumulative_tables.update(range(stats['table_count']))
    
    return {
        "percentile": percentile,
        "threshold_table_count": threshold_count,
        "selected_entities": selected_entities,
        "total_entity_types": len(entity_stats)
    }

