"""
Dify RAG Agent — CLI 入口
=========================
用法:
  python main.py                          # 交互问答 (自动建索引)
  python main.py "什么是 Dify Agent?"     # 单次问答
  python main.py --eval                   # 运行完整 RAG 评估
  python main.py --eval --eval-mode retrieval  # 仅检索评估
  python main.py --eval --eval-output report.json  # 保存评估报告
"""

from __future__ import annotations

import sys

from dify_rag.config import check_config
from dify_rag.vectorstore import ensure_index
from dify_rag.rag import ask


# ---------------------------------------------------------------------------
# 评估 CLI
# ---------------------------------------------------------------------------


def _parse_eval_args(argv: list) -> dict | None:
    """从命令行参数中解析 --eval 相关参数。"""
    if "--eval" not in argv:
        return None

    from dify_rag.config import TOP_K

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
    from dify_rag.config import EMBEDDING_MODEL, LLM_MODEL
    from dify_rag.llm import get_llm
    from dify_rag.embedding import DashScopeEmbeddings
    from dify_rag.eval.engine import run_evaluation, save_report
    from dify_rag.eval.dataset import get_default_dataset, load_dataset_from_json, dataset_summary
    from dify_rag.retrieval import _retrieve_fn_for_eval
    from dify_rag.rag import eval_generate

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
    llm = get_llm()
    embeddings = DashScopeEmbeddings()

    # 运行评估
    mode = eval_args["mode"]

    report = run_evaluation(
        dataset=dataset,
        mode=mode,
        llm=llm,
        embeddings=embeddings,
        retrieve_fn=_retrieve_fn_for_eval,
        generate_fn=lambda q, ctxs: eval_generate(q, ctxs, llm),
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
    check_config()

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
