"""调试混合检索在 6 条失败查询上的表现."""
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.dirname(__file__))

from main import retrieve_docs, ensure_index

ensure_index()

queries = [
    (3, "如何在 Dify 中创建一个能够自动调用工具的 AI Agent？"),
    (11, "Dify 工作流中 LLM 节点支持哪些功能？"),
    (12, "Dify 工作流中如何在 LLM 节点里使用对话历史？"),
    (28, "Dify 中如何为不同终端用户提供个性化体验？"),
    (29, "Dify 的对话变量是什么？怎么用？"),
    (30, "通过 API 上传文件到 Dify 有什么限制？"),
]

for num, q in queries:
    docs = retrieve_docs(q, top_k=5)
    print(f"--- #{num}: {q[:60]}")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?").replace("\\", "/").split("/")[-1]
        preview = d.page_content[:100].replace("\n", " ")
        print(f"  [{i}] {src} | {preview}...")
    print()
