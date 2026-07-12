"""逐个分析失败查询：看 top-5 检索结果 vs ground truth"""
import sys
sys.path.insert(0, '.')

from dify_rag.retrieval import retrieve_docs
from dify_rag.eval.dataset import get_default_dataset

dataset = get_default_dataset()

# 失败问题的 0-based index
failed_indices = [7, 8, 9, 10, 11, 14, 27, 28, 29, 31, 39, 46, 48, 51, 55, 57]

for idx in failed_indices:
    s = dataset[idx]
    print(f"=== [{idx+1}] {s.question[:100]} ===")
    print(f"  Ground truth: {s.relevant_sources}")

    try:
        docs = retrieve_docs(s.question, top_k=5)
        for i, doc in enumerate(docs):
            src = doc.metadata.get('source', '?')
            match = any(gt.replace('\\', '/').lower() in src.replace('\\', '/').lower()
                       for gt in s.relevant_sources)
            marker = '✅' if match else '❌'
            # Show last 100 chars of content
            content = doc.page_content[:120].replace('\n', ' ')
            print(f"  [{i+1}] {marker} {src[-90:]}")
            print(f"       {content}")
    except Exception as e:
        print(f"  Error: {e}")
    print()
