"""
RAG 对话 — 问答、prompt 构建、评估用生成。
"""

from __future__ import annotations

import time
from typing import List

from dify_rag.config import CHROMA_PERSIST_DIR, TOP_K

_PROMPT_TEMPLATE = """你是一个 Dify 平台的专家助手。请只根据以下参考文档回答用户问题。
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


def build_prompt(question: str, docs) -> str:
    """构建 RAG prompt。"""
    context = "\n\n---\n\n".join(
        f"[来源: {d.metadata.get('source', '?')}]\n{d.page_content}"
        for d in docs
    )
    return _PROMPT_TEMPLATE.format(context=context, question=question)


def eval_generate(question: str, contexts: List[str], llm) -> str:
    """评估用生成: 根据检索上下文生成回答。"""
    context_text = "\n\n---\n\n".join(contexts)
    prompt = _PROMPT_TEMPLATE.format(context=context_text, question=question)
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def ask(
    question: str,
    persist_dir: str = CHROMA_PERSIST_DIR,
    top_k: int = TOP_K,
    *,
    _return_data: bool = False,
):
    """检索并回答单个问题。

    Args:
        question: 用户问题。
        persist_dir: Chroma 持久化目录。
        top_k: 检索数量。
        _return_data: 内部使用 — 为 True 时返回 (answer, docs) 元组。
    """
    from dify_rag.retrieval import retrieve_docs
    from dify_rag.llm import get_llm

    # 检索
    docs = retrieve_docs(question, persist_dir=persist_dir, top_k=top_k)

    # 构建 prompt
    prompt = build_prompt(question, docs)

    # 调用 LLM（带重试）
    llm = get_llm()
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
