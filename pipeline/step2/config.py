"""
Configuration file for Step 2: Entity Sampling and Analysis
"""
import os

# GPT-5 API Configuration
GPT5_API_URL = os.environ.get("LLM_API_URL", "")
GPT5_API_KEY = os.environ.get("GPT5_API_KEY", "")
GPT5_MODEL = "gpt-5"
GPT5_TEMPERATURE = 0.7
GPT5_MAX_TOKENS = 15000  # Must include reasoning tokens (~6000) + output tokens (~2000)

# Sampling Configuration
SAMPLE_SIZE = 3500  # Production mode: 3500 samples (change to 10 for testing)
RANDOM_SEED = 42  # For reproducible sampling

# Path Configuration
INPUT_DIR = "data/step1/output/table_markdown_1/"
OUTPUT_DIR = "./step2/output"
LOG_DIR = "./step2/logs"

# API Call Configuration
REQUEST_DELAY = 2  # Seconds between requests (increased to reduce API pressure)
RETRY_TIMES = 4  # Number of retries on failure (increased for GPT-5 stability)
TIMEOUT = 240  # Request timeout in seconds (increased for GPT-5 reasoning time)
BACKOFF_FACTOR = 2  # Exponential backoff multiplier (increased: 2s, 4s, 8s)

# Parallel Processing Configuration
MAX_WORKERS = 3  # Number of parallel workers (reduced to 3 to avoid API overload)
USE_PARALLEL = True  # Enable parallel processing

# Progress Saving
SAVE_PROGRESS_EVERY = 50  # Save results every N tables processed

# Percentile Configuration
PERCENTILE_THRESHOLD = 80  # Calculate 80th percentile of entity types

