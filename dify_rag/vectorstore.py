"""
向量库管理 — Chroma 懒加载、索引构建、健康检查。
"""

from __future__ import annotations

from pathlib import Path

from dify_rag.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME, DIFY_DOCS_DIR

_vectorstore_cache = None


def _get_embeddings():
    """懒加载 Embeddings 实例 (延迟导入避免循环引用)。"""
    from dify_rag.embedding import DashScopeEmbeddings

    return DashScopeEmbeddings()


def get_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR):
    """懒加载并缓存 Chroma 向量库，避免重复磁盘 I/O。"""
    global _vectorstore_cache

    if _vectorstore_cache is not None:
        return _vectorstore_cache

    from langchain_chroma import Chroma

    embeddings = _get_embeddings()
    _vectorstore_cache = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    return _vectorstore_cache


def _collection_has_data(persist_dir: str) -> bool:
    """检查当前模型的 Chroma collection 是否包含实际记录。"""
    try:
        import chromadb

        client = chromadb.PersistentClient(path=persist_dir)
        try:
            collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
            return collection.count() > 0
        except Exception:
            return False
    except Exception:
        return False


def ensure_index(
    docs_dir: str = DIFY_DOCS_DIR,
    persist_dir: str = CHROMA_PERSIST_DIR,
):
    """确保向量库存在, 不存在则自动清洗 + 向量化。"""
    global _vectorstore_cache

    if Path(persist_dir).exists() and _collection_has_data(persist_dir):
        print(f"✅ 向量库已存在且包含数据: {persist_dir}  (collection: {CHROMA_COLLECTION_NAME})")
        get_vectorstore(persist_dir)  # 预热缓存
        return

    print("🔨 向量库不存在, 开始自动构建...")
    print(f"📂 文档目录: {docs_dir}")

    from dify_rag.text_cleaner import build_rag_documents
    from dify_rag.meta_filter import filter_complex_metadata
    from langchain_chroma import Chroma

    # 清洗 + 分块
    documents = build_rag_documents(docs_dir, as_langchain=True)
    documents = [d for d in documents if d.page_content.strip()]
    print(f"📦 共 {len(documents)} 个 LangChain Document (已过滤空块)")

    # 过滤复杂 metadata — ChromaDB 只接受 str/int/float/bool/None
    for d in documents:
        d.metadata = filter_complex_metadata(d.metadata)
    print(f"🔧 已过滤复杂 metadata (ChromaDB 兼容)")

    # 向量化
    embeddings = _get_embeddings()
    print(f"🔢 Embedding 模型: {embeddings.model}  (百炼原生 API)")

    # 入库并缓存
    _vectorstore_cache = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    print(f"💾 向量库已持久化至: {persist_dir}")
    print(f"📊 共 {_vectorstore_cache._collection.count()} 条记录")
    print()
