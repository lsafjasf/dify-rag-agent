"""
Dify RAG Agent
==============
直接运行即可 —— 自动建索引 → 交互问答。

用法:
  python main.py                          # 交互问答 (自动建索引)
  python main.py "什么是 Dify Agent?"     # 单次问答

配置:
  所有环境变量统一在 .env 文件中管理。
  支持的环境变量见项目根目录 .env 文件的注释。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 加载 .env 文件 (优先级最高)
# ---------------------------------------------------------------------------

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# 可选依赖导入
# ---------------------------------------------------------------------------

try:
    from langchain_chroma import Chroma
    HAS_CHROMA = True
except ImportError:
    Chroma = None  # type: ignore[assignment]
    HAS_CHROMA = False

from text_cleaner import build_rag_documents

# ---------------------------------------------------------------------------
# 配置 (从 .env / 环境变量读取)
# ---------------------------------------------------------------------------

DIFY_DOCS_DIR = os.environ.get("DIFY_DOCS_DIR", "C:/Users/Administrator/Desktop/zh")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")

# ---- Embedding (阿里云百炼, 原生 API) ----
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v1")
EMBEDDING_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("EMBEDDING_API_KEY", "")
# 百炼原生 embedding 端点 (非 compatible-mode, 避免兼容层翻译 bug)
EMBEDDING_URL = os.environ.get(
    "EMBEDDING_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
)

# ---- LLM (DeepSeek) ----
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("LLM_API_KEY", "")

TOP_K = int(os.environ.get("TOP_K", "5"))


# ---------------------------------------------------------------------------
# 向量库缓存 —— 避免每次 ask() 都重新从磁盘加载
# ---------------------------------------------------------------------------

_vectorstore_cache: "Chroma | None" = None


def _get_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR) -> "Chroma":
    """懒加载并缓存 Chroma 向量库，避免重复磁盘 I/O。"""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        embeddings = DashScopeEmbeddings()
        _vectorstore_cache = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    return _vectorstore_cache


# ---------------------------------------------------------------------------
# 自定义 Embedding —— 直连百炼原生 API, 绕过兼容层
# ---------------------------------------------------------------------------

class DashScopeEmbeddings(BaseModel):
    """阿里云百炼原生 Embedding, 实现 LangChain Embeddings 接口。

    为什么不用 OpenAIEmbeddings + compatible-mode?
      → 百炼兼容层把 OpenAI 的 "input" 字段翻译为嵌套的 "contents",
        翻译过程中某些批次触发类型校验失败, 报:
          "contents is neither str nor list of str."
        换成直连原生 API 彻底绕过这个问题。
    """

    model: str = Field(default=EMBEDDING_MODEL)
    api_key: str = Field(default=EMBEDDING_API_KEY, repr=False)
    url: str = Field(default=EMBEDDING_URL)
    max_retries: int = 3
    retry_delay: float = 1.0

    model_config = {"extra": "forbid"}

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """调用百炼原生 embedding API, 带重试。"""
        # 过滤空白
        texts = [t for t in texts if t.strip()]
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {"texts": texts},
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url, json=payload, headers=headers, timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("output", {}).get("embeddings", [])
                    # 按 text_index 排序，保证顺序
                    embeddings.sort(key=lambda e: e.get("text_index", 0))
                    return [e["embedding"] for e in embeddings]
                elif resp.status_code == 401:
                    # 认证失败，重试无意义，立即抛出
                    raise RuntimeError(
                        f"百炼 API 认证失败（401），请检查 DASHSCOPE_API_KEY 是否正确: "
                        f"{resp.text[:300]}"
                    )
                elif resp.status_code == 429:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                elif resp.status_code >= 500:
                    # 服务端错误，可重试
                    last_error = resp.text
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    # 其他客户端错误（4xx），不重试
                    raise RuntimeError(
                        f"百炼 Embedding API 返回 {resp.status_code}: {resp.text[:300]}"
                    )
            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"百炼 Embedding API 调用失败: {last_error}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding, Chroma 入库用。"""
        # 百炼原生 API 单次最多 25 条, 这里保守用 20
        batch_size = 20
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            all_embeddings.extend(self._call_api(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """单条 embedding, 检索用。"""
        results = self._call_api([text])
        if results:
            return results[0]
        raise RuntimeError("Embedding 返回为空")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _filter_complex_metadata(metadata: dict) -> dict:
    """过滤 metadata 中的嵌套结构，ChromaDB 只接受 str/int/float/bool/None。

    - 嵌套 dict → JSON 字符串
    - list → 保持 (ChromaDB 原生支持 list of simple values)
    - 其余类型原样返回
    """
    import json
    result = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            result[key] = json.dumps(value, ensure_ascii=False)
        elif value is None or isinstance(value, (str, int, float, bool, list)):
            result[key] = value
        else:
            result[key] = str(value)
    return result


# ---------------------------------------------------------------------------
# 启动前校验
# ---------------------------------------------------------------------------

def _check_config():
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


# ---------------------------------------------------------------------------
# 步骤 1: 构建向量索引 (自动)
# ---------------------------------------------------------------------------

def _collection_has_data(persist_dir: str) -> bool:
    """检查 Chroma 集合是否包含实际记录。"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=persist_dir)
        collections = client.list_collections()
        if not collections:
            return False
        # 至少有一个集合包含 > 0 条记录
        return any(c.count() > 0 for c in collections)
    except Exception:
        return False


def ensure_index(docs_dir: str = DIFY_DOCS_DIR, persist_dir: str = CHROMA_PERSIST_DIR):
    """确保向量库存在, 不存在则自动清洗 + 向量化。"""
    global _vectorstore_cache

    if Path(persist_dir).exists() and _collection_has_data(persist_dir):
        print(f"✅ 向量库已存在且包含数据: {persist_dir}")
        _get_vectorstore(persist_dir)  # 预热缓存
        return

    if not HAS_CHROMA:
        raise RuntimeError("请安装 langchain-chroma chromadb: pip install langchain-chroma chromadb")

    print("🔨 向量库不存在, 开始自动构建...")
    print(f"📂 文档目录: {docs_dir}")

    # 清洗 + 分块
    documents = build_rag_documents(docs_dir, as_langchain=True)
    documents = [d for d in documents if d.page_content.strip()]
    print(f"📦 共 {len(documents)} 个 LangChain Document (已过滤空块)")

    # 过滤复杂 metadata — ChromaDB 只接受 str/int/float/bool/None
    for d in documents:
        d.metadata = _filter_complex_metadata(d.metadata)
    print(f"🔧 已过滤复杂 metadata (ChromaDB 兼容)")

    # 向量化 (百炼原生 API, 不走兼容层)
    embeddings = DashScopeEmbeddings()
    print(f"🔢 Embedding 模型: {EMBEDDING_MODEL}  (百炼原生 API)")

    # 入库并缓存
    _vectorstore_cache = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"💾 向量库已持久化至: {persist_dir}")
    print(f"📊 共 {_vectorstore_cache._collection.count()} 条记录")
    print()


# ---------------------------------------------------------------------------
# 步骤 2: 检索问答
# ---------------------------------------------------------------------------

def ask(question: str, persist_dir: str = CHROMA_PERSIST_DIR, top_k: int = TOP_K):
    """检索并回答单个问题。"""
    if not HAS_CHROMA:
        raise RuntimeError("请安装 langchain-chroma chromadb: pip install langchain-chroma chromadb")

    from langchain_openai import ChatOpenAI

    # 复用缓存的向量库，避免每次从磁盘重新加载
    vectorstore = _get_vectorstore(persist_dir)

    # 检索
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )
    docs = retriever.invoke(question)

    # 构建上下文
    context = "\n\n---\n\n".join(
        f"[来源: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in docs
    )

    # 调用 LLM（带重试）
    llm_kwargs: dict = {"model": LLM_MODEL, "temperature": 0.3}
    if LLM_BASE_URL:
        llm_kwargs["base_url"] = LLM_BASE_URL
    if LLM_API_KEY:
        llm_kwargs["api_key"] = LLM_API_KEY
    llm = ChatOpenAI(**llm_kwargs)

    prompt = f"""你是一个 Dify 平台的专家助手。请只根据以下参考文档回答用户问题。
如果文档中没有相关信息, 请明确告知用户。

## 参考文档

{context}

## 用户问题

{question}

## 回答要求

- 使用中文回答。
- 引用文档中的具体信息。
- 如果信息不足, 如实说明。
- 回答结构清晰, 可使用 Markdown 格式。
"""

    max_retries = 3
    retry_delay = 1.0
    last_error = None
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            break
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
    else:
        raise RuntimeError(f"LLM 调用失败（已重试 {max_retries} 次）: {last_error}")

    print("\n" + "=" * 60)
    print(f"🔍 问题: {question}")
    print(f"📚 检索到 {len(docs)} 个相关片段")
    print("=" * 60)
    print(f"\n💬 回答:\n{response.content}")
    print("\n" + "-" * 60)
    print("📌 参考来源:")
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        title = d.metadata.get("title", d.metadata.get("file_stem", "?"))
        print(f"  {i}. {title}  ({source})")

    return response


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    _check_config()

    # 确保向量库就绪
    ensure_index()

    # 如果命令行传了问题, 单次问答
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        ask(question)
        return

    # 否则进入交互循环
    print("=" * 60)
    print("🤖 Dify RAG Agent (DeepSeek)")
    print("   输入问题后回车, 输入 quit / exit / q 退出")
    print("=" * 60)
    while True:
        print()
        q = input("❓ 你的问题: ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("👋 再见!")
            break
        try:
            ask(q)
        except Exception as e:
            print(f"\n⚠️ 出错了: {e}")


if __name__ == "__main__":
    main()
