"""
检索调试工具 —— 逐条展示每条问题检索到的实际文档和内容片段，
帮助你判断是「标注错了」还是「检索真的没找到」。

用法:
  python scripts/debug_retrieval.py           # 只显示失败的问题
  python scripts/debug_retrieval.py --all     # 显示全部 30 条
"""
from __future__ import annotations

import sys

from dify_rag.retrieval import retrieve_docs
from dify_rag.vectorstore import ensure_index
from dify_rag.eval.dataset import get_default_dataset

K = 5


def run(show_all: bool = False):
    ensure_index()
    dataset = get_default_dataset()

    failed = 0
    for i, sample in enumerate(dataset):
        docs = retrieve_docs(sample.question, top_k=K)
        sources = [d.metadata.get("source", "?") for d in docs]

        # 检查匹配
        hits = []
        for j, src in enumerate(sources):
            matched = any(
                name.replace("\\", "/").lower() in src.replace("\\", "/").lower()
                for name in sample.relevant_sources
            )
            hits.append((j + 1, src, matched))

        is_hit = any(h[2] for h in hits)

        # 过滤逻辑
        if not show_all and is_hit:
            continue

        status = "HIT" if is_hit else "MISS"
        failed += 0 if is_hit else 1

        print(f"\n{'=' * 70}")
        print(f"[{status}] #{i + 1} | {sample.question}")
        print(f"  期望来源: {sample.relevant_sources}")
        print(f"  实际检索 (Top-{K}):")
        for rank, src, matched in hits:
            marker = " OK" if matched else "   "
            filename = src.replace("\\", "/").split("/")[-1] if src else "?"
            print(f"    [{rank}] {marker} {filename}")
            print(f"         {src}")

        # 展示第一条检索结果的摘要
        if docs:
            preview = docs[0].page_content[:120].replace("\n", " ")
            print(f"  Top-1 内容预览: {preview}...")

    print(f"\n{'=' * 70}")
    print(f"失败: {failed} / {len(dataset)} 条")


if __name__ == "__main__":
    show_all = "--all" in sys.argv
    run(show_all=show_all)
