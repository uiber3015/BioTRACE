import os
"""Step 4 配置文件"""
from pathlib import Path

# ============= 路径配置 =============
# 输入路径
INPUT_BASE_DIR = Path("data/step3/output")
INPUT_CATEGORY = "all_in"  # 只处理 all_in 分类

# 输出路径
OUTPUT_BASE_DIR = Path("./output")
RAW_CHAINS_DIR = OUTPUT_BASE_DIR / "raw_chains"
CLEAN_CHAINS_DIR = OUTPUT_BASE_DIR / "clean_chains"
FAILED_DIR = OUTPUT_BASE_DIR / "failed"

# ============= API 配置（本地 Qwen API）=============
# 注意：vLLM的路径包含 /api 前缀（见qwen3-32b.md中的--root-path /api配置）
API_URL = os.environ.get("QWEN_API_URL", "")
API_TOKEN = "not-needed"  # vLLM不需要token
MODEL_NAME = "qwen3-32b"

# 请求参数
TEMPERATURE = 0.3  # 稍高保持创造性，但不失格式稳定性
MAX_TOKENS = 8000  # 增加token限制，支持生成更多知识链（目标~20条/表）
TIMEOUT = 300  # 本地API可能需要更长时间（5分钟，因为要生成更多知识链）

# ============= 并发配置 =============
MAX_CONCURRENCY = 10  # 并发数
RETRY_TIMES = 3  # 重试次数
RETRY_DELAY = 5  # 重试延迟（秒）

# ============= 处理配置 =============
# 限制处理数量（测试用，None 表示处理全部）
LIMIT_COUNT = 20  # 先测试100个，确认无误后改为 None

# Context 长度限制
MAX_CONTEXT_LENGTH = 500  # mentioned_in_text 最大长度

# ============= 日志配置 =============
LOG_FILE = OUTPUT_BASE_DIR / "processing.log"
LOG_LEVEL = "INFO"
