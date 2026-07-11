"""
Dify RAG Agent — 轻量级 RAG 检索 + 评估工具包。
================================================

用法:
  from dify_rag import ask, retrieve_docs, ensure_index
  from dify_rag.eval.engine import run_evaluation, EvalReport
"""

# ---- 核心 API ----
from dify_rag.rag import ask
from dify_rag.retrieval import retrieve_docs
from dify_rag.vectorstore import ensure_index

# ---- 评估 ----
from dify_rag.eval.engine import run_evaluation, save_report, EvalReport
from dify_rag.eval.dataset import get_default_dataset, load_dataset_from_json, dataset_summary

# ---- 配置 ----
from dify_rag.config import (
    DIFY_DOCS_DIR,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_API_KEY,
    LLM_MODEL,
    TOP_K,
    USE_HYDE,
    USE_QE,
    USE_RERANKER,
)

# ---- 底层模块 (按需引用) ----
from dify_rag.embedding import DashScopeEmbeddings
from dify_rag.llm import get_llm
