"""
Utility functions for Step 3: Markdown Filtering
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from html.parser import HTMLParser


def setup_logging(log_dir: str, log_file: str = "markdown_filter.log") -> logging.Logger:
    """
    Setup logging configuration with detailed output for nohup runs
    
    Ensures:
    - Logs are written to both file and stdout
    - Output is immediately flushed (unbuffered)
    - Detailed format with timestamps
    - Works properly with nohup redirection
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("markdown_filter")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all logs
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler - detailed logs to file
    fh = logging.FileHandler(log_path / log_file, encoding='utf-8', mode='a')
    fh.setLevel(logging.DEBUG)
    
    # Console handler - explicitly use stdout for nohup compatibility
    # Using sys.stdout ensures output is visible in nohup.out
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)  # Console shows INFO and above
    
    # Detailed formatter with timestamp
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Simpler formatter for console (less verbose)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh.setFormatter(detailed_formatter)
    ch.setFormatter(console_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    # Force immediate flush for both handlers
    fh.flush()
    ch.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    
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


def extract_table_id_from_filename(file_path: str) -> str:
    """Extract table ID from markdown filename"""
    filename = Path(file_path).stem
    # Return the full filename without extension as ID
    return filename


class SimpleHTMLTableParser(HTMLParser):
    """Simple HTML table parser to extract rows"""
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = []
        self.current_cell = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False
    
    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.in_table = True
        elif tag == 'tr' and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ['td', 'th'] and self.in_row:
            self.in_cell = True
            self.current_cell = []
    
    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_table = False
        elif tag == 'tr' and self.in_row:
            self.in_row = False
            if self.current_row:
                self.rows.append(self.current_row)
        elif tag in ['td', 'th'] and self.in_cell:
            self.in_cell = False
            self.current_row.append(''.join(self.current_cell).strip())
    
    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)


def smart_truncate_for_filtering(markdown_content: str, sample_rows: int = 10) -> str:
    """
    Smart truncation for filtering: only keep essential information for judgment
    - Caption (title/description)
    - Table header (header row + separator)
    - First N sample data rows (enough to judge table type)
    
    Args:
        markdown_content: Full markdown content
        sample_rows: Number of sample data rows to keep (default: 10)
    
    Returns:
        Truncated markdown content with only essential information
    """
    lines = markdown_content.split('\n')
    result_lines = []
    in_caption = False
    in_table_content = False
    table_lines = []
    caption_lines = []
    
    for line in lines:
        # Capture Caption section
        if line.strip().startswith('## Caption'):
            in_caption = True
            result_lines.append(line)
            continue
        
        # Stop capturing Caption at next section
        if in_caption and line.strip().startswith('##'):
            in_caption = False
            result_lines.append(line)
            continue
        
        # Capture Caption content
        if in_caption:
            caption_lines.append(line)
            result_lines.append(line)
            continue
        
        # Detect Table Content section
        if line.strip().startswith('## Table Content'):
            in_table_content = True
            result_lines.append(line)
            continue
        
        # Detect next section (end of table content)
        if in_table_content and line.strip().startswith('##'):
            in_table_content = False
            # Process accumulated table lines with smart truncation
            if table_lines:
                truncated = _smart_truncate_table_lines(table_lines, sample_rows)
                result_lines.extend(truncated)
                table_lines = []
            result_lines.append(line)
            continue
        
        # Accumulate table lines
        if in_table_content:
            table_lines.append(line)
        else:
            result_lines.append(line)
    
    # Handle case where table content is at the end
    if table_lines:
        truncated = _smart_truncate_table_lines(table_lines, sample_rows)
        result_lines.extend(truncated)
    
    return '\n'.join(result_lines)


def truncate_markdown_table(markdown_content: str, max_rows: int = 200, truncate_to: int = 100) -> str:
    """
    Truncate markdown table content to avoid overly long prompts
    
    Args:
        markdown_content: Full markdown content
        max_rows: Maximum number of rows before truncation
        truncate_to: Number of rows to keep if truncation is needed
    
    Returns:
        Truncated markdown content
    """
    lines = markdown_content.split('\n')
    result_lines = []
    in_table_content = False
    table_lines = []
    
    for line in lines:
        # Detect Table Content section
        if line.strip().startswith('## Table Content'):
            in_table_content = True
            result_lines.append(line)
            continue
        
        # Detect next section (end of table content)
        if in_table_content and line.strip().startswith('##'):
            in_table_content = False
            # Process accumulated table lines
            if table_lines:
                truncated = _truncate_table_lines(table_lines, max_rows, truncate_to)
                result_lines.extend(truncated)
                table_lines = []
            result_lines.append(line)
            continue
        
        # Accumulate table lines
        if in_table_content:
            table_lines.append(line)
        else:
            result_lines.append(line)
    
    # Handle case where table content is at the end
    if table_lines:
        truncated = _truncate_table_lines(table_lines, max_rows, truncate_to)
        result_lines.extend(truncated)
    
    return '\n'.join(result_lines)


def _smart_truncate_table_lines(table_lines: List[str], sample_rows: int = 10) -> List[str]:
    """
    Smart truncation: only keep header + first N sample rows for filtering judgment
    """
    # Check if it's an HTML code block
    if any('```html' in line for line in table_lines):
        return _smart_truncate_html_table(table_lines, sample_rows)
    else:
        return _smart_truncate_markdown_table(table_lines, sample_rows)


def _smart_truncate_html_table(table_lines: List[str], sample_rows: int) -> List[str]:
    """Smart truncate HTML table: header + first N sample rows"""
    html_content = '\n'.join(table_lines)
    
    # Extract HTML from code block
    html_match = re.search(r'```html\s*(.*?)\s*```', html_content, re.DOTALL)
    if not html_match:
        return table_lines  # Return as-is if can't parse
    
    html_str = html_match.group(1)
    
    # Parse HTML
    parser = SimpleHTMLTableParser()
    try:
        parser.feed(html_str)
    except:
        return table_lines  # Return as-is if parsing fails
    
    if len(parser.rows) <= sample_rows + 1:  # +1 for header
        return table_lines  # No truncation needed if small enough
    
    # Keep header + first N sample rows
    header_rows = parser.rows[:1]  # Keep first row as header
    data_rows = parser.rows[1:sample_rows + 1]  # First N sample rows
    
    # Reconstruct as simple markdown table
    truncated_lines = []
    truncated_lines.append("```")
    truncated_lines.append(f"Table sample: showing {len(data_rows)} sample rows of {len(parser.rows)-1} total rows")
    truncated_lines.append("")
    
    # Add header
    if header_rows:
        truncated_lines.append(" | ".join(header_rows[0]))
        truncated_lines.append(" | ".join(["---"] * len(header_rows[0])))
    
    # Add sample data rows
    for row in data_rows:
        truncated_lines.append(" | ".join(row))
    
    truncated_lines.append("")
    truncated_lines.append(f"*(Showing {len(data_rows)} of {len(parser.rows)-1} data rows for filtering judgment)*")
    truncated_lines.append("```")
    
    return truncated_lines


def _smart_truncate_markdown_table(table_lines: List[str], sample_rows: int) -> List[str]:
    """Smart truncate markdown table: header + first N sample rows"""
    # Count table rows (lines with |)
    table_rows = [line for line in table_lines if '|' in line and line.strip()]
    
    if len(table_rows) <= sample_rows + 2:  # +2 for header + separator
        return table_lines  # No truncation needed if small enough
    
    # Keep header (first 2 rows: header + separator)
    header_lines = table_rows[:2] if len(table_rows) >= 2 else table_rows[:1]
    # Keep first N sample data rows (skip header + separator, so start from index 2)
    data_rows = table_rows[2:sample_rows + 2] if len(table_rows) >= 2 else []
    
    result = []
    result.extend(header_lines)
    result.extend(data_rows)
    result.append("")
    result.append(f"*(Showing {len(data_rows)} of {len(table_rows)-2} data rows for filtering judgment)*")
    
    return result


def _truncate_table_lines(table_lines: List[str], max_rows: int, truncate_to: int) -> List[str]:
    """Helper to truncate table lines"""
    # Check if it's an HTML code block
    if any('```html' in line for line in table_lines):
        return _truncate_html_table(table_lines, max_rows, truncate_to)
    else:
        return _truncate_markdown_table_format(table_lines, max_rows, truncate_to)


def _truncate_html_table(table_lines: List[str], max_rows: int, truncate_to: int) -> List[str]:
    """Truncate HTML table within markdown"""
    html_content = '\n'.join(table_lines)
    
    # Extract HTML from code block
    html_match = re.search(r'```html\s*(.*?)\s*```', html_content, re.DOTALL)
    if not html_match:
        return table_lines  # Return as-is if can't parse
    
    html_str = html_match.group(1)
    
    # Parse HTML
    parser = SimpleHTMLTableParser()
    try:
        parser.feed(html_str)
    except:
        return table_lines  # Return as-is if parsing fails
    
    if len(parser.rows) <= max_rows:
        return table_lines  # No truncation needed
    
    # Truncate rows
    header_rows = parser.rows[:1]  # Keep first row as header
    data_rows = parser.rows[1:truncate_to]
    
    # Reconstruct as simple markdown table
    truncated_lines = []
    truncated_lines.append("```")
    truncated_lines.append(f"Table truncated: showing {len(data_rows)} of {len(parser.rows)-1} data rows")
    truncated_lines.append("")
    
    # Add header
    if header_rows:
        truncated_lines.append(" | ".join(header_rows[0]))
        truncated_lines.append(" | ".join(["---"] * len(header_rows[0])))
    
    # Add data rows
    for row in data_rows:
        truncated_lines.append(" | ".join(row))
    
    truncated_lines.append("")
    truncated_lines.append("...(remaining rows omitted)")
    truncated_lines.append("```")
    
    return truncated_lines


def _truncate_markdown_table_format(table_lines: List[str], max_rows: int, truncate_to: int) -> List[str]:
    """Truncate standard markdown table format"""
    # Count table rows (lines with |)
    table_rows = [line for line in table_lines if '|' in line and line.strip()]
    
    if len(table_rows) <= max_rows:
        return table_lines  # No truncation needed
    
    # Keep header (first 2 rows: header + separator)
    header_lines = table_rows[:2] if len(table_rows) >= 2 else table_rows[:1]
    data_rows = table_rows[2:truncate_to] if len(table_rows) >= 2 else []
    
    result = []
    result.extend(header_lines)
    result.extend(data_rows)
    result.append("")
    result.append(f"*(Table truncated: showing {len(data_rows)} of {len(table_rows)-2} data rows)*")
    
    return result


def extract_relevant_sections(markdown_content: str) -> str:
    """
    Extract only Caption and Table Content sections from markdown
    
    Returns:
        Truncated markdown with only relevant sections
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
        
        if capture or line.strip().startswith('# Table'):  # Keep title
            extracted.append(line)
    
    result = '\n'.join(extracted)
    
    # Apply length limit
    if len(result) > 6000:
        result = result[:6000] + "\n\n...(content truncated)"
    
    return result


def parse_json_response(response_text: str) -> Dict:
    """
    Parse JSON from Qwen response with flexible extraction
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
    
    # Try to find JSON object in text (non-greedy)
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    raise ValueError(f"Could not parse JSON from response: {response_text[:300]}")

