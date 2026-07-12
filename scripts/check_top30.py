"""Check if GT docs are in top-20 results for remaining failures."""
import sys
sys.path.insert(0, '.')
from dify_rag.retrieval import retrieve_docs
from dify_rag.eval.dataset import get_default_dataset

dataset = get_default_dataset()
failed = [3, 11, 12, 18, 30, 47, 49, 52, 56]

for idx in failed:
    idx0 = idx - 1  # 0-based
    s = dataset[idx0]
    gt = [g.replace("\\", "/").lower() for g in s.relevant_sources]

    docs = retrieve_docs(s.question, top_k=30)

    found = False
    for rank, d in enumerate(docs, 1):
        src = d.metadata.get("source", "").replace("\\", "/").lower()
        cat = d.metadata.get("doc_category", "?")
        if any(g in src for g in gt):
            print(f"#{idx:2d} GT at rank {rank:2d} (cat={cat}) | {s.question[:60]}")
            found = True
            break

    if not found:
        cats = [d.metadata.get("doc_category", "?") for d in docs[:15]]
        print(f"#{idx:2d} GT NOT in top-30 | top-15 cats: {cats}")
        print(f"     GT: {gt}")
        # Show top-3 sources
        for i, d in enumerate(docs[:3]):
            src = d.metadata.get("source", "").replace("\\", "/")
            print(f"     [{i+1}] {src.split('/')[-1]}")
    print()
