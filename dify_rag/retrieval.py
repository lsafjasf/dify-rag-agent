"""
检索流水线 — HyDE + QE BM25 + RRF 融合 + BGE Reranker 精排。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

from dify_rag.config import (
    CHROMA_PERSIST_DIR,
    USE_HYDE,
    USE_QE,
    USE_RERANKER,
    TOP_K,
)

# ---- 缓存 ----

_embeddings_cache = None
_hyde_cache = None
_reranker_cache = None


def _get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        from dify_rag.embedding import DashScopeEmbeddings, CachedEmbeddings
        _embeddings_cache = CachedEmbeddings(DashScopeEmbeddings())
    return _embeddings_cache


def _get_hyde_gen():
    global _hyde_cache
    if not USE_HYDE:
        return None
    if _hyde_cache is None:
        try:
            from dify_rag.llm import get_llm
            from dify_rag.hyde_generator import HyDEGenerator

            _hyde_cache = HyDEGenerator(get_llm())
            print("✅ HyDE 生成器已就绪")
        except Exception as exc:
            print(f"⚠️ HyDE 生成器初始化失败: {exc}")
            return None
    return _hyde_cache


def _get_reranker():
    global _reranker_cache
    if not USE_RERANKER:
        return None
    if _reranker_cache is None:
        try:
            from dify_rag.llm import get_llm
            from dify_rag.bge_reranker import BGEReranker
            from dify_rag.config import EMBEDDING_API_KEY

            _reranker_cache = BGEReranker(
                dashscope_api_key=EMBEDDING_API_KEY,
                llm=get_llm(),
            )
        except Exception as exc:
            print(f"⚠️ BGE Reranker 初始化失败: {exc}")
            return None
    return _reranker_cache


# ---------------------------------------------------------------------------
# 主检索接口
# ---------------------------------------------------------------------------


def retrieve_docs(
    question: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
    top_k: int = TOP_K,
) -> list["Document"]:
    """HyDE 向量检索 + QE BM25 + RRF 融合 + BGE Reranker 精排。

    完整流水线:
      1. HyDE: LLM 生成假设文档 → Embedding → 向量搜索
      2. QE:   LLM 生成关键词查询 → 每个 BM25 检索 → 合并去重
      3. RRF:  Reciprocal Rank Fusion 融合两组候选
      4. Reranker: BGE 跨编码器 / LLM 对候选精排 → Top-K

    Args:
        question: 用户问题（中文口语化）。
        persist_dir: Chroma 持久化目录。
        top_k: 返回结果数量。

    Returns:
        LangChain Document 列表。
    """
    from dify_rag.vectorstore import get_vectorstore
    from dify_rag.llm import get_llm
    from dify_rag.hybrid_retriever import get_hybrid_retriever

    vectorstore = get_vectorstore(persist_dir)
    embeddings = _get_embeddings()
    llm = get_llm()
    hyde_gen = _get_hyde_gen()
    reranker = _get_reranker()

    hybrid = get_hybrid_retriever(
        vectorstore,
        embeddings=embeddings,
        llm=llm,
        hyde_generator=hyde_gen,
        bge_reranker=reranker,
    )

    return hybrid.hybrid_search(question, top_k, use_qe=USE_QE)


def _retrieve_fn_for_eval(question: str, k: int = TOP_K):
    """适配 eval 检索接口: 返回带 source/page_content 的 dict 列表。"""
    docs = retrieve_docs(question, top_k=k)
    return [
        {
            "source": d.metadata.get("source", ""),
            "page_content": d.page_content,
        }
        for d in docs
    ]
