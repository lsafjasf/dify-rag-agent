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


# ---------------------------------------------------------------------------
# 查询意图 → 文档类别加权
# ---------------------------------------------------------------------------

# 技术参考类查询的关键词模式 —— 用户想了解"有哪些/是什么/区别"
_REF_QUERY_PATTERNS = [
    "支持哪些", "有哪些功能", "有什么功能", "有什么区别", "有什么不同",
    "有哪些类型", "支持什么", "限制", "版本控制", "参数", "配置选项",
    "是什么", "做什么", "有什么用", "用途", "功能包括",
    "排查", "什么问题", "检查什么",
    "需要哪些", "前置条件", "系统要求",
]

# 操作指南类查询 —— 用户想了解"怎么做"
_HOWTO_QUERY_PATTERNS = [
    "怎么用", "如何使用", "如何配置", "如何设置", "怎么配置",
    "怎么设置", "如何创建", "如何安装", "如何部署",
    "如何在", "如何管理", "如何通过", "如何将", "如何为",
    "如何从", "导入", "接入",
]


def _analyze_query_intent(question: str) -> str:
    """分析查询意图: reference / howto / mixed / neutral。

    - reference: 仅匹配技术参考类模式
    - howto: 仅匹配操作指南类模式
    - mixed: 同时匹配两类模式 (如 "支持哪些类型？如何安装？")
    - neutral: 都不匹配
    """
    ref_score = sum(1 for p in _REF_QUERY_PATTERNS if p in question)
    howto_score = sum(1 for p in _HOWTO_QUERY_PATTERNS if p in question)
    if ref_score > 0 and howto_score > 0:
        return "mixed"
    if ref_score > howto_score:
        return "reference"
    elif howto_score > ref_score:
        return "howto"
    return "neutral"


def _boost_reference_docs(
    question: str, docs: list["Document"], top_k: int
) -> list["Document"]:
    """对技术参考类查询, 将参考文档排名提升 1-2 位。

    仅当 top-3 中没有 reference 类文档, 且 top_k 内存在 reference 文档时触发。
    将一个排名最高的 reference 文档交换到第 2 位 (不替换第 1 位, 保守策略)。
    """
    intent = _analyze_query_intent(question)
    if intent != "reference":
        return docs
    if len(docs) <= 1:
        return docs

    # 检查 top-3 中是否已有 reference 文档
    top_ref = any(
        d.metadata.get("doc_category") == "reference"
        for d in docs[:3]
    )
    if top_ref:
        return docs  # 已有参考文档, 无需调整

    # 找排名最高的 reference 文档
    best_ref_idx = None
    for i, d in enumerate(docs):
        if d.metadata.get("doc_category") == "reference":
            best_ref_idx = i
            break

    if best_ref_idx is None or best_ref_idx < 3:
        return docs  # 没有可提升的 reference 文档, 或已在 top-3

    # 将 reference 文档提升到第 2 位 (保留第 1 位不变)
    ref_doc = docs.pop(best_ref_idx)
    docs.insert(1, ref_doc)

    return docs[:top_k]


def _boost_guide_docs(
    question: str, docs: list["Document"], top_k: int
) -> list["Document"]:
    """对操作指南类查询, 将 guide/tutorial 文档排名提升 1-2 位。

    仅当 top-3 中没有 guide/tutorial 类文档, 且 top_k 内存在时触发。
    将一个排名最高的 guide/tutorial 文档交换到第 2 位 (不替换第 1 位)。
    """
    intent = _analyze_query_intent(question)
    if intent != "howto":
        return docs
    if len(docs) <= 1:
        return docs

    # 检查 top-3 中是否已有 guide/tutorial 文档
    top_guide = any(
        d.metadata.get("doc_category") in ("guide", "tutorial")
        for d in docs[:3]
    )
    if top_guide:
        return docs  # 已有操作指南文档, 无需调整

    # 找排名最高的 guide/tutorial 文档
    best_idx = None
    for i, d in enumerate(docs):
        if d.metadata.get("doc_category") in ("guide", "tutorial"):
            best_idx = i
            break

    if best_idx is None or best_idx < 3:
        return docs  # 没有可提升的文档, 或已在 top-3

    # 将 guide/tutorial 文档提升到第 2 位 (保留第 1 位不变)
    guide_doc = docs.pop(best_idx)
    docs.insert(1, guide_doc)

    return docs[:top_k]


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

    # Phase 3: 分析查询意图, 触发元数据预过滤检索
    intent = _analyze_query_intent(question)
    docs = hybrid.hybrid_search(question, top_k, use_qe=USE_QE, intent=intent)

    # Phase 3: 后置 boost (作为预过滤的补充安全网)
    docs = _boost_reference_docs(question, docs, top_k)
    docs = _boost_guide_docs(question, docs, top_k)

    return docs


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
