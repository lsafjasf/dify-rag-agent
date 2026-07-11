"""
混合检索: HyDE 向量检索 + Query Expansion BM25 + RRF 融合 + BGE Reranker
======================================================================
完整的 RAG 检索流水线，集成以下技术：

  1. HyDE (Hypothetical Document Embeddings)
     └─ LLM 生成假设文档 → Embedding → 向量检索
  2. Query Expansion
     └─ LLM 生成 3-5 个关键词查询 → BM25 多路检索 → 结果去重合并
  3. RRF (Reciprocal Rank Fusion)
     └─ 融合向量检索和 BM25 检索的两组结果
  4. BGE Reranker
     └─ 跨编码器精排，筛选最相关文档

用法:
  from hybrid_retriever import get_hybrid_retriever
  retriever = get_hybrid_retriever(vectorstore, embeddings, llm)
  docs = retriever.hybrid_search("如何创建知识库?", top_k=5)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# BM25 检索器 (零依赖, 纯 Python)
# ---------------------------------------------------------------------------


class BM25Retriever:
    """纯 Python BM25 实现，针对中文文本使用字符级 unigram + bigram 分词。

    对于中文这种无空格分隔的语言，字符级 n-gram 是零依赖下最稳定的方案：
    - unigram 捕获单字匹配（如 "模型" 匹配 "模" + "型"）
    - bigram 捕获双字组合（如 "节点"、"工作流"），提高精确度
    """

    def __init__(
        self,
        documents: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """
        Args:
            documents: [{"content": str, "source": str, "chunk_index": int, ...}, ...]
            k1: 词频饱和参数 (典型值 1.2-2.0)
            b:  文档长度归一化参数 (典型值 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents = documents
        self.corpus = [self._tokenize(d.get("content", "")) for d in documents]
        self.N = len(self.corpus)
        self.avgdl = sum(len(doc) for doc in self.corpus) / max(self.N, 1)
        self._doc_lens = [len(doc) for doc in self.corpus]
        self.idf = self._compute_idf()

    def _tokenize(self, text: str) -> List[str]:
        """字符级 unigram + bigram 分词（中文友好）。"""
        # 提取所有非空白字符
        chars = [c for c in text if not c.isspace()]
        tokens = list(chars)  # unigrams
        for i in range(len(chars) - 1):
            tokens.append(chars[i] + chars[i + 1])  # bigrams
        return tokens

    def _compute_idf(self) -> Dict[str, float]:
        """计算 IDF: log((N - df + 0.5) / (df + 0.5) + 1)"""
        df: Dict[str, int] = {}
        for doc in self.corpus:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        N = max(self.N, 1)
        return {
            term: math.log((N - cnt + 0.5) / max(cnt, 0.5) + 1)
            for term, cnt in df.items()
        }

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """BM25 检索，返回 top_k 文档。"""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.documents[:top_k]

        scores: List[tuple[int, float]] = []
        for i, doc_tokens in enumerate(self.corpus):
            score = 0.0
            doc_len = self._doc_lens[i]
            for term in query_tokens:
                idf_val = self.idf.get(term, 0)
                if idf_val == 0:
                    continue
                tf = doc_tokens.count(term)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self.avgdl, 1)
                )
                score += idf_val * (numerator / denominator)
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.documents[i] for i, _ in scores[:top_k]]


# ---------------------------------------------------------------------------
# 混合检索器 (HyDE + QE + BM25 + Vector + RRF + Reranker)
# ---------------------------------------------------------------------------


class HybridRetriever:
    """HyDE 向量检索 + Query Expansion BM25 + RRF 融合 + BGE Reranker。

    完整流水线:
      1. HyDE: LLM 生成假设文档 → Embedding → 向量相似度搜索
      2. QE:   LLM 生成 3-5 个关键词查询 → 每个查询 BM25 检索 → 合并去重
      3. RRF:  融合两组候选文档
      4. Rerank: BGE 跨编码器精排 → Top-K
    """

    def __init__(
        self,
        vectorstore,
        bm25_docs: Optional[List[Dict[str, Any]]] = None,
        rrf_k: int = 60,
        hyde_generator=None,
        bge_reranker=None,
        embeddings=None,
    ):
        """
        Args:
            vectorstore: LangChain Chroma 向量库实例。
            bm25_docs: BM25 文档列表，为 None 时从 Chroma 自动提取。
            rrf_k: RRF 平滑参数。
            hyde_generator: HyDEGenerator 实例 (可选)。
            bge_reranker: BGEReranker 实例 (可选)。
            embeddings: Embeddings 实例 (HyDE 向量检索需要)。
        """
        self.vectorstore = vectorstore
        self.rrf_k = rrf_k
        self._embeddings = embeddings

        if bm25_docs is None:
            bm25_docs = self._extract_docs_from_chroma()

        self.bm25 = BM25Retriever(bm25_docs) if bm25_docs else None
        self._llm = None
        self._hyde_gen = hyde_generator
        self._reranker = bge_reranker

    # ---- Setters ----

    def set_llm(self, llm) -> None:
        """注入 LLM 实例以启用查询扩展。"""
        self._llm = llm

    def set_hyde_generator(self, hyde_gen) -> None:
        """注入 HyDE 生成器。"""
        self._hyde_gen = hyde_gen

    def set_reranker(self, reranker) -> None:
        """注入 BGE Reranker。"""
        self._reranker = reranker

    def set_embeddings(self, embeddings) -> None:
        """注入 Embeddings 实例（HyDE 需要）。"""
        self._embeddings = embeddings

    # ---- Query Expansion (增强版) ----

    _EXPANSION_PROMPT = """将用户问题改写为 3-5 个不同的关键词搜索查询，用于 BM25 搜索引擎检索 Dify 技术文档。
每个查询使用不同的关键词组合和表述角度，以覆盖更多相关文档。

核心原则：只做同义词映射和术语补全，绝对不编造任何 API 名称、参数名或端点。

术语映射参考（正确的做法）：
- "对话历史" → "记忆 memory 对话窗口 conversation"
- "阻塞式返回" → "streaming 流式 阻塞 blocking 模式"
- "人工介入" → "human_input_required workflow_paused 人工审批"
- "上传文件" → "file upload 文件上传 接口 API"
- "外部知识库" → "external knowledge API 外部知识库 检索"

错误示例（绝对不要）：
- 不要编造 "vector_keyword_hybrid" 等不存在的 API 名
- 不要猜测具体的端点路径

用户问题: {question}

要求：
- 每个查询是关键词组合（非完整句子），2-6 个词
- 包含技术术语、API 名称、英文关键词的混合
- 从不同角度覆盖（如：配置角度、API 角度、使用场景角度）

返回 JSON 格式，不要输出其他内容：
{{"queries": ["关键词查询1", "关键词查询2", ...]}}"""

    def _expand_query(self, question: str) -> List[str]:
        """用 LLM 将用户问题改写为 3-5 个关键词搜索查询。

        Args:
            question: 原始用户问题。

        Returns:
            改写后的查询列表（包含原问题）。
        """
        if self._llm is None:
            return [question]

        import json
        import re

        prompt = self._EXPANSION_PROMPT.format(question=question)

        try:
            response = self._llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)

            # 解析 JSON
            data = None
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                pass

            if data is None:
                m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
                if m:
                    try:
                        data = json.loads(m.group(1))
                    except json.JSONDecodeError:
                        pass

            if data is None:
                m = re.search(r"\{[\s\S]*\}", text)
                if m:
                    try:
                        data = json.loads(m.group())
                    except json.JSONDecodeError:
                        pass

            queries = data.get("queries", []) if data else []
            if not queries:
                return [question]

            # 去重并限制数量（原问题优先）
            seen = {question}
            unique = [question]
            for q in queries:
                q = q.strip()
                if q and q not in seen:
                    seen.add(q)
                    unique.append(q)
            return unique[:5]  # 原问题 + 最多 4 个扩展

        except Exception:
            return [question]

    # ---- HyDE Vector Search ----

    def _hyde_search(self, query: str, top_k: int) -> List:
        """HyDE 向量检索：生成假设文档 → Embedding → 向量搜索。

        Args:
            query: 用户原始问题。
            top_k: 返回数量。

        Returns:
            LangChain Document 列表。
        """
        if self._hyde_gen is None or self._embeddings is None:
            # 降级：使用原始 query 进行向量检索
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
            return retriever.invoke(query)

        try:
            # 1) 生成假设文档
            hyde_text = self._hyde_gen.generate(query)

            # 2) 嵌入假设文档
            hyde_embedding = self._embeddings.embed_query(hyde_text)

            # 3) 向量相似度搜索
            return self.vectorstore.similarity_search_by_vector(
                hyde_embedding, k=top_k
            )
        except Exception as exc:
            print(f"  ⚠️ HyDE 检索失败 ({exc})，降级使用原始查询")
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
            return retriever.invoke(query)

    # ---- BM25 Multi-Query Search ----

    def _bm25_multi_search(
        self,
        queries: List[str],
        top_k_per_query: int,
    ) -> List[Dict[str, Any]]:
        """对每个扩展查询执行 BM25 检索，合并去重。

        Args:
            queries: 查询列表（含原问题）。
            top_k_per_query: 每个查询返回的结果数。

        Returns:
            去重合并后的 BM25 文档列表。
        """
        if self.bm25 is None:
            return []

        seen_keys: set = set()
        merged: List[Dict[str, Any]] = []

        for q in queries:
            results = self.bm25.search(q, top_k_per_query)
            for doc in results:
                key = f"{doc.get('source', '')}::{doc.get('chunk_index', 0)}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    merged.append(doc)

        return merged

    # ---- 文档管理 ----

    def _extract_docs_from_chroma(self) -> List[Dict[str, Any]]:
        """从 Chroma 集合中提取所有文档，供 BM25 索引。"""
        try:
            collection = self.vectorstore._collection
            result = collection.get(include=["documents", "metadatas"])
            docs: List[Dict[str, Any]] = []
            for content, meta in zip(result["documents"], result["metadatas"]):
                if content and content.strip():
                    docs.append({
                        "content": content,
                        "source": meta.get("source", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                        "file_stem": meta.get("file_stem", ""),
                        "title": meta.get("title", ""),
                    })
            return docs
        except Exception:
            return []

    @staticmethod
    def _doc_key(doc) -> str:
        """生成文档唯一标识: source::chunk_index"""
        if hasattr(doc, "metadata"):
            src = doc.metadata.get("source", "")
            ci = doc.metadata.get("chunk_index", 0)
        elif isinstance(doc, dict):
            src = doc.get("source", "")
            ci = doc.get("chunk_index", 0)
        else:
            return str(id(doc))
        return f"{src}::{ci}"

    # ---- RRF Fusion ----

    def _rrf_fuse(
        self,
        vector_results: List,
        bm25_results: List[Dict],
        top_k: int,
    ) -> List:
        """RRF 融合两组结果，返回 LangChain Document 列表。

        RRF 公式: score(d) = Σ 1 / (k + rank_i(d))
        """
        from langchain_core.documents import Document

        scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}

        # 向量检索结果 → LangChain Document
        for rank, doc in enumerate(vector_results, 1):
            key = self._doc_key(doc)
            scores[key] = scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
            doc_map[key] = doc

        # BM25 结果 → 转换为 LangChain Document
        for rank, bm25_doc in enumerate(bm25_results, 1):
            key = f"{bm25_doc['source']}::{bm25_doc['chunk_index']}"
            scores[key] = scores.get(key, 0) + 1.0 / (self.rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = Document(
                    page_content=bm25_doc["content"],
                    metadata={
                        "source": bm25_doc["source"],
                        "chunk_index": bm25_doc["chunk_index"],
                        "file_stem": bm25_doc.get("file_stem", ""),
                        "title": bm25_doc.get("title", ""),
                    },
                )

        # 按 RRF 分数降序
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked[:top_k]]

    # ---- 主检索入口 ----

    def hybrid_search(self, query: str, top_k: int = 5, *, use_qe: bool = True) -> List:
        """执行完整混合检索流水线。

        流水线:
          1. HyDE 向量检索 (parallel with step 2 conceptually)
          2. Query Expansion → BM25 多路检索
          3. RRF 融合
          4. BGE Reranker 重排序
          5. 返回 Top-K

        Args:
            query: 用户问题（中文口语化）。
            top_k: 最终返回数量。
            use_qe: 是否启用 Query Expansion（LLM 改写多关键词查询）。

        Returns:
            LangChain Document 列表。
        """
        # ── Step 1: HyDE 向量检索 ──
        vector_candidates = self._hyde_search(
            query, top_k=min(top_k * 3, 30)
        )

        # ── Step 2: Query Expansion + BM25 ──
        if use_qe:
            expanded_queries = self._expand_query(query)
        else:
            expanded_queries = [query]  # 不做扩展，直接用原问题

        bm25_candidates = self._bm25_multi_search(
            expanded_queries, top_k_per_query=min(top_k * 2, 15)
        )

        # ── Step 3: RRF 融合 ──
        if bm25_candidates:
            candidates = self._rrf_fuse(
                vector_candidates, bm25_candidates,
                top_k=min(top_k * 3, 30),  # 多取一些给 reranker 用
            )
        else:
            candidates = vector_candidates[:min(top_k * 3, 30)]

        # ── Step 4: BGE Reranker 精排 ──
        if self._reranker is not None and self._reranker.is_available:
            candidates = self._reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def hybrid_search_with_expansion(self, query: str, top_k: int = 5) -> List:
        """向后兼容的查询扩展+混合检索。

        现在等同于 hybrid_search()，因为主流程已内置 QE。
        保留此方法以维持旧接口兼容。

        Args:
            query: 用户原始问题。
            top_k: 返回结果数量。

        Returns:
            LangChain Document 列表。
        """
        return self.hybrid_search(query, top_k)


# ---------------------------------------------------------------------------
# 全局缓存
# ---------------------------------------------------------------------------

_hybrid_cache: Optional[HybridRetriever] = None
_cache_vectorstore_id: Optional[int] = None


def get_hybrid_retriever(
    vectorstore,
    embeddings=None,
    llm=None,
    hyde_generator=None,
    bge_reranker=None,
) -> HybridRetriever:
    """获取或创建 HybridRetriever（懒加载 + 缓存）。

    Args:
        vectorstore: LangChain Chroma 向量库实例。
        embeddings: Embeddings 实例（HyDE 需要）。
        llm: LLM 实例（Query Expansion 需要）。
        hyde_generator: HyDEGenerator 实例（可选，自动创建）。
        bge_reranker: BGEReranker 实例（可选，自动创建）。

    Returns:
        HybridRetriever 实例。
    """
    global _hybrid_cache, _cache_vectorstore_id
    vs_id = id(vectorstore)

    if _hybrid_cache is None or _cache_vectorstore_id != vs_id:
        _hybrid_cache = HybridRetriever(
            vectorstore,
            hyde_generator=hyde_generator,
            bge_reranker=bge_reranker,
            embeddings=embeddings,
        )
        _hybrid_cache.set_llm(llm)
        _cache_vectorstore_id = vs_id

    return _hybrid_cache
