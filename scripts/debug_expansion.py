"""调试查询扩展效果"""
from dify_rag.llm import get_llm
from dify_rag.vectorstore import get_vectorstore
from dify_rag.config import CHROMA_PERSIST_DIR
from dify_rag.hybrid_retriever import get_hybrid_retriever

vectorstore = get_vectorstore(CHROMA_PERSIST_DIR)
llm = get_llm()
hybrid = get_hybrid_retriever(vectorstore)
hybrid.set_llm(llm)

queries = [
    (9, "知识库 API 支持哪些检索方式？"),
    (11, "Dify 工作流中 LLM 节点支持哪些功能？"),
    (12, "Dify 工作流中如何在 LLM 节点里使用对话历史？"),
    (28, "Dify 中如何为不同终端用户提供个性化体验？"),
    (29, "Dify 的对话变量是什么？怎么用？"),
    (30, "通过 API 上传文件到 Dify 有什么限制？"),
]

for num, q in queries:
    expanded = hybrid._expand_query(q)
    print(f"--- #{num}: {q}")
    for i, eq in enumerate(expanded):
        marker = "(原)" if eq == q else f"({i})"
        print(f"  {marker} {eq}")
    print()
