"""
全局配置 — .env 加载、常量定义、启动校验。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# .env 加载 (优先级最高)
# ---------------------------------------------------------------------------

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------

DIFY_DOCS_DIR = os.environ.get("DIFY_DOCS_DIR", "C:/Users/Administrator/Desktop/zh")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")

# ---------------------------------------------------------------------------
# Embedding (阿里云百炼, 原生 API)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v1")

# Chroma collection 名: 按模型隔离, 不同 embedding 模型使用不同 collection
# 旧 v1 数据保留在 "langchain" (langchain_chroma 默认名) 中不受影响
CHROMA_COLLECTION_NAME = os.environ.get(
    "CHROMA_COLLECTION_NAME",
    f"dify_docs_{EMBEDDING_MODEL}",
)
EMBEDDING_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_URL = os.environ.get(
    "EMBEDDING_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
)

# ---------------------------------------------------------------------------
# LLM (DeepSeek)
# ---------------------------------------------------------------------------

LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("LLM_API_KEY", "")

# ---------------------------------------------------------------------------
# 检索参数
# ---------------------------------------------------------------------------

TOP_K = int(os.environ.get("TOP_K", "5"))

# ---------------------------------------------------------------------------
# 高级检索开关
# ---------------------------------------------------------------------------

USE_HYDE = os.environ.get("USE_HYDE", "true").lower() not in ("false", "0", "no", "off")
USE_QE = os.environ.get("USE_QE", "true").lower() not in ("false", "0", "no", "off")
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() not in ("false", "0", "no", "off")


# ---------------------------------------------------------------------------
# 启动前校验
# ---------------------------------------------------------------------------

def check_config():
    """检查必要的环境变量, 缺失时给出明确提示。"""
    missing = []

    if not EMBEDDING_API_KEY:
        missing.append("DASHSCOPE_API_KEY  (阿里云百炼, 用于 Embedding, 免费)")

    if not LLM_API_KEY:
        missing.append("DEEPSEEK_API_KEY 或 LLM_API_KEY  (DeepSeek, 用于问答)")

    if not Path(DIFY_DOCS_DIR).exists():
        missing.append(f"DIFY_DOCS_DIR 指向的目录不存在: {DIFY_DOCS_DIR}")

    if missing:
        print("❌ 缺少必要的环境变量, 请先设置:\n")
        for m in missing:
            print(f"   export {m}")
        print("\n示例:")
        print('   export DASHSCOPE_API_KEY="sk-xxxxxx"')
        print('   export DEEPSEEK_API_KEY="sk-xxxxxx"')
        print(f'   export DIFY_DOCS_DIR="{DIFY_DOCS_DIR}"')
        sys.exit(1)
