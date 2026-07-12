"""Compare reference doc chunks vs tutorial chunks."""
import sys
sys.path.insert(0, '.')
from dify_rag.vectorstore import get_vectorstore

vs = get_vectorstore()
all_data = vs.get()
sources = all_data['metadatas']
docs = all_data['documents']

# Find llm.mdx chunks
llm_ref = []
tutorial_llm = []
for i, meta in enumerate(sources):
    src = meta.get('source', '')
    src_norm = src.replace('\\', '/')
    if 'nodes/llm.mdx' in src_norm and 'use-dify' in src_norm:
        llm_ref.append(i)
    if 'tutorials' in src_norm and 'workflow' in src_norm:
        tutorial_llm.append(i)

for label, indices in [('参考文档 nodes/llm.mdx', llm_ref), ('教程 workflow lessons', tutorial_llm[:5])]:
    print(f"=== {label}: {len(indices)} chunks ===")
    for idx in indices[:3]:
        content = docs[idx][:400].replace('\n', ' ')
        print(f"  [{len(docs[idx])} chars] {content}")
        print()
