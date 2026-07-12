"""Debug remaining failures: check if ground truth docs are in top-k results."""
import sys
sys.path.insert(0, '.')
from dify_rag.retrieval import retrieve_docs, _analyze_query_intent
from dify_rag.eval.dataset import get_default_dataset

dataset = get_default_dataset()
failed_indices = [10, 11, 27, 28, 31, 39, 46, 48, 49, 51]

for idx in failed_indices:
    s = dataset[idx]
    q = s.question
    gt = [g.replace('\\', '/').lower() for g in s.relevant_sources]

    try:
        docs = retrieve_docs(q, top_k=10)
    except Exception as e:
        print(f'[{idx+1}] Error: {e}')
        continue

    gt_found = False
    for i, d in enumerate(docs):
        src = d.metadata.get('source', '').replace('\\', '/').lower()
        cat = d.metadata.get('doc_category', '?')
        match = any(g in src for g in gt)
        if match:
            gt_found = True
            print(f'[{idx+1}] ✅ GT at rank {i+1} (cat={cat})')
            break

    if not gt_found:
        cats_5 = [d.metadata.get('doc_category', '?') for d in docs[:5]]
        cats_10 = [d.metadata.get('doc_category', '?') for d in docs]
        print(f'[{idx+1}] ❌ GT not in top-10')
        print(f'       Query: {q[:80]}')
        print(f'       Intent: {_analyze_query_intent(q)}')
        print(f'       GT: {gt}')
        print(f'       Top-5 cats: {cats_5}')
        print(f'       Top-10 cats: {cats_10}')
        # Show what IS at top
        for i, d in enumerate(docs[:3]):
            src = d.metadata.get('source', '').replace('\\', '/')
            print(f'       Rank {i+1}: {src[-70:]}')
    print()
