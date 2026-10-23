"""
Configuration file for Step 3: Markdown Table Filtering with Qwen72b
"""
import os

# Qwen API Configuration (OpenAI-compatible format)
QWEN_API_URL = os.environ.get("QWEN_API_URL", "")
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_MODEL = "qwen2.5:72b-instruct"
QWEN_TEMPERATURE = 0.3  # Lower temperature for more consistent filtering

# Use OpenAI-compatible format by default
USE_OPENAI_FORMAT = True

# Sampling Configuration
SAMPLE_SIZE = None  # None = process all tables; set to a number for testing (e.g., 20)

# Path Configuration
INPUT_DIR = "data/step1/output/table_markdown_1/"  # Same as step2
ENTITY_TYPES_FILE = "../step2/percentile_80_entities.json"  # Will use default if not found
OUTPUT_DIR = "./step3/output"
LOG_DIR = "./step3/logs"

# Markdown Truncation Configuration
# Smart truncation for filtering: only keep essential info (Caption + Header + Sample rows)
SAMPLE_ROWS_FOR_FILTERING = 10  # Number of sample data rows to keep for filtering judgment
MAX_CONTENT_LENGTH = 6000  # Maximum characters for markdown content (fallback limit)

# Legacy truncation parameters (for backward compatibility)
MAX_TABLE_ROWS = 200  # Maximum rows to keep in table
TRUNCATE_TO_ROWS = 100  # If exceeds max, truncate to this many rows

# API Call Configuration
REQUEST_DELAY = 1.5  # Seconds between requests
RETRY_TIMES = 3  # Number of retries on failure
TIMEOUT = 120  # Request timeout in seconds
BACKOFF_FACTOR = 2  # Exponential backoff multiplier

# Progress Saving
SAVE_PROGRESS_EVERY = 100  # Save results every N tables processed

# Concurrency Configuration
MAX_WORKERS = 7  # Number of concurrent threads for processing tables

# Filtering Criteria
REQUIRE_ALL_CONDITIONS = True  # All three conditions must be True to pass

