"""
Dify RAG Agent
==============
直接运行即可 —— 自动建索引 → 交互问答。

用法:
  python main.py                          # 交互问答 (自动建索引)
  python main.py "什么是 Dify Agent?"     # 单次问答
  python main.py --eval                   # 运行完整 RAG 评估
  python main.py --eval --eval-mode retrieval  # 仅检索评估
  python main.py --eval --eval-output report.json  # 保存评估报告

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
from eval_dataset import get_default_dataset, load_dataset_from_json, dataset_summary
from eval_rag import run_evaluation, save_report, EvalReport
from hybrid_retriever import get_hybrid_retriever
from hyde_generator import HyDEGenerator
from bge_reranker import BGEReranker

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

# ---- 高级检索开关 ----
USE_HYDE = os.environ.get("USE_HYDE", "true").lower() not in ("false", "0", "no", "off")
USE_QE = os.environ.get("USE_QE", "true").lower() not in ("false", "0", "no", "off")
USE_RERANKER = os.environ.get("USE_RERANKER", "true").lower() not in ("false", "0", "no", "off")


# ---------------------------------------------------------------------------
# 向量库缓存 —— 避免每次 ask() 都重新从磁盘加载
# ---------------------------------------------------------------------------

_vectorstore_cache: "Chroma | None" = None
_embeddings_cache: "DashScopeEmbeddings | None" = None
_hyde_cache: "HyDEGenerator | None" = None
_reranker_cache: "BGEReranker | None" = None
_llm_cache = None


def _get_embeddings() -> "DashScopeEmbeddings":
    """懒加载 Embeddings 实例。"""
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = DashScopeEmbeddings()
    return _embeddings_cache


def _get_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR) -> "Chroma":
    """懒加载并缓存 Chroma 向量库，避免重复磁盘 I/O。"""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        embeddings = _get_embeddings()
        _vectorstore_cache = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    return _vectorstore_cache


def _get_hyde_gen() -> "HyDEGenerator | None":
    """懒加载 HyDE 生成器（需要 LLM）。"""
    global _hyde_cache
    if not USE_HYDE:
        return None
    if _hyde_cache is None:
        try:
            _hyde_cache = HyDEGenerator(_get_llm())
            print("✅ HyDE 生成器已就绪")
        except Exception as exc:
            print(f"⚠️ HyDE 生成器初始化失败: {exc}")
            return None
    return _hyde_cache


def _get_reranker() -> "BGEReranker | None":
    """懒加载 BGE Reranker。

    优先级: 百炼 DashScope Rerank API → FlagEmbedding 本地 → LLM fallback
    """
    global _reranker_cache
    if not USE_RERANKER:
        return None
    if _reranker_cache is None:
        try:
            _reranker_cache = BGEReranker(
                dashscope_api_key=EMBEDDING_API_KEY,
                llm=_get_llm(),
            )
        except Exception as exc:
            print(f"⚠️ BGE Reranker 初始化失败: {exc}")
            return None
    return _reranker_cache


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
# 检索与生成 (可复用的核心函数, 供 ask() 和 eval 使用)
# ---------------------------------------------------------------------------


def retrieve_docs(question: str, persist_dir: str = CHROMA_PERSIST_DIR, top_k: int = TOP_K):
    """HyDE 向量检索 + QE BM25 + RRF 融合 + BGE Reranker 精排。

    完整流水线:
      1. HyDE: LLM 生成假设文档 → Embedding → 向量搜索
      2. QE:   LLM 生成 3-5 个关键词查询 → 每个 BM25 检索 → 合并去重
      3. RRF:  Reciprocal Rank Fusion 融合两组候选
      4. Reranker: BGE 跨编码器 / LLM 对候选精排 → Top-K

    通过环境变量 USE_HYDE / USE_RERANKER 可独立开关各模块。

    Args:
        question: 用户问题（中文口语化）。
        persist_dir: Chroma 持久化目录。
        top_k: 返回结果数量。

    Returns:
        LangChain Document 列表。
    """
    if not HAS_CHROMA:
        raise RuntimeError("请安装 langchain-chroma chromadb: pip install langchain-chroma chromadb")

    vectorstore = _get_vectorstore(persist_dir)
    embeddings = _get_embeddings()
    llm = _get_llm()
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


def _get_llm():
    """获取配置好的 LLM 实例 (懒加载缓存)。

    Returns:
        ChatOpenAI 实例。
    """
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    from langchain_openai import ChatOpenAI

    llm_kwargs: dict = {"model": LLM_MODEL, "temperature": 0.3}
    if LLM_BASE_URL:
        llm_kwargs["base_url"] = LLM_BASE_URL
    if LLM_API_KEY:
        llm_kwargs["api_key"] = LLM_API_KEY
    _llm_cache = ChatOpenAI(**llm_kwargs)
    return _llm_cache


def _build_prompt(question: str, docs) -> str:
    """构建 RAG prompt (复用与 ask() 相同的模板)。

    Args:
        question: 用户问题。
        docs: 检索到的 LangChain Document 列表。

    Returns:
        格式化的 prompt 字符串。
    """
    context = "\n\n---\n\n".join(
        f"[来源: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in docs
    )
    return f"""你是一个 Dify 平台的专家助手。请只根据以下参考文档回答用户问题。
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


def _eval_generate(question: str, contexts: List[str], llm) -> str:
    """评估用生成函数: 根据检索上下文生成回答。

    Args:
        question: 用户问题。
        contexts: 检索到的上下文字符串列表 (已提取 page_content)。
        llm: LangChain LLM 实例。

    Returns:
        生成的回答文本。
    """
    context_text = "\n\n---\n\n".join(contexts)
    prompt = f"""你是一个 Dify 平台的专家助手。请只根据以下参考文档回答用户问题。
如果文档中没有相关信息, 请明确告知用户。

## 参考文档

{context_text}

## 用户问题

{question}

## 回答要求

- 使用中文回答。
- 引用文档中的具体信息。
- 如果信息不足, 如实说明。
- 回答结构清晰, 可使用 Markdown 格式。
"""
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _retrieve_fn_for_eval(question: str, k: int = TOP_K):
    """适配 eval_rag 检索接口: 返回带 source/page_content 的 dict 列表。

    Args:
        question: 用户问题。
        k: Top-K 参数。

    Returns:
        [{"source": str, "page_content": str}, ...]
    """
    docs = retrieve_docs(question, top_k=k)
    return [
        {
            "source": d.metadata.get("source", ""),
            "page_content": d.page_content,
        }
        for d in docs
    ]


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
    embeddings = _get_embeddings()
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

def ask(question: str, persist_dir: str = CHROMA_PERSIST_DIR, top_k: int = TOP_K, *, _return_data: bool = False):
    """检索并回答单个问题。

    Args:
        question: 用户问题。
        persist_dir: Chroma 持久化目录。
        top_k: 检索数量。
        _return_data: 内部使用 — 为 True 时返回 (answer, docs) 元组, 不打印。
    """
    if not HAS_CHROMA:
        raise RuntimeError("请安装 langchain-chroma chromadb: pip install langchain-chroma chromadb")

    # 检索
    docs = retrieve_docs(question, persist_dir=persist_dir, top_k=top_k)

    # 构建 prompt
    prompt = _build_prompt(question, docs)

    # 调用 LLM（带重试）
    llm = _get_llm()
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

    answer = response.content if hasattr(response, "content") else str(response)

    if _return_data:
        return (answer, docs)

    print("\n" + "=" * 60)
    print(f"🔍 问题: {question}")
    print(f"📚 检索到 {len(docs)} 个相关片段")
    print("=" * 60)
    print(f"\n💬 回答:\n{answer}")
    print("\n" + "-" * 60)
    print("📌 参考来源:")
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        title = d.metadata.get("title", d.metadata.get("file_stem", "?"))
        print(f"  {i}. {title}  ({source})")

    return response


# ---------------------------------------------------------------------------
# 评估 CLI 辅助
# ---------------------------------------------------------------------------


def _parse_eval_args(argv: list) -> dict | None:
    """从命令行参数中解析 --eval 相关参数。

    返回 None 表示不是 eval 模式; 返回 dict 表示 eval 配置。
    """
    if "--eval" not in argv:
        return None

    args = {
        "mode": "all",
        "output": None,
        "dataset": None,
        "top_k": TOP_K,
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--eval-mode" and i + 1 < len(argv):
            args["mode"] = argv[i + 1]
            i += 2
        elif arg == "--eval-output" and i + 1 < len(argv):
            args["output"] = argv[i + 1]
            i += 2
        elif arg == "--eval-dataset" and i + 1 < len(argv):
            args["dataset"] = argv[i + 1]
            i += 2
        elif arg == "--eval-top-k" and i + 1 < len(argv):
            args["top_k"] = int(argv[i + 1])
            i += 2
        elif arg == "--eval":
            i += 1
        else:
            i += 1

    return args


def _run_eval_cli(eval_args: dict) -> None:
    """执行评估并输出结果。"""
    # 加载数据集
    if eval_args["dataset"]:
        dataset = load_dataset_from_json(eval_args["dataset"])
        print(f"📂 从文件加载数据集: {eval_args['dataset']}")
    else:
        dataset = get_default_dataset()
        print("📦 使用内置默认数据集")

    print(dataset_summary(dataset))

    # 确保向量库就绪
    ensure_index()

    # 获取 LLM 和 Embeddings
    llm = _get_llm()
    embeddings = _get_embeddings()

    # 运行评估
    mode = eval_args["mode"]

    report = run_evaluation(
        dataset=dataset,
        mode=mode,
        llm=llm,
        embeddings=embeddings,
        retrieve_fn=_retrieve_fn_for_eval,
        generate_fn=lambda q, ctxs: _eval_generate(q, ctxs, llm),
        top_k=eval_args["top_k"],
        config={
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "top_k": eval_args["top_k"],
        },
        verbose=True,
    )

    # 打印报告
    print(report.format_summary())

    # 保存 JSON 报告
    output_path = eval_args["output"]
    if output_path:
        save_report(report, output_path)
        print(f"📄 报告已保存至: {output_path}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    _check_config()

    # ---- 评估模式 ----
    eval_args = _parse_eval_args(sys.argv)
    if eval_args:
        _run_eval_cli(eval_args)
        return

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
