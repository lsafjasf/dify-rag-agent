"""分析 eval 失败的 16 个问题，检查 ground truth 文件是否存在并被索引。"""
import sys
sys.path.insert(0, '.')

import os
import glob as gb

from dify_rag.vectorstore import get_vectorstore
from dify_rag.embedding import DashScopeEmbeddings, CachedEmbeddings

docs_dir = 'C:/Users/Administrator/Desktop/zh'

# 失败问题的 ground truth 来源 (去重)
failed_gt = [
    'api-reference/guides/knowledge.mdx',
    'use-dify/nodes/llm.mdx',
    'use-dify/nodes/loop.mdx',
    'api-reference/guides/chat.mdx',
    'api-reference/guides/end-user-identity.mdx',
    'api-reference/guides/chatflow.mdx',
    'use-dify/build/workflow-chatflow.mdx',
    'use-dify/nodes/code.mdx',
    'use-dify/nodes/template.mdx',
    'use-dify/nodes/answer.mdx',
    'use-dify/build/version-control.mdx',
    'use-dify/workspace/plugins.mdx',
    'use-dify/debug/variable-inspect.mdx',
    'cli/install.mdx',
    'cli/authenticate.mdx',
]

embeddings = CachedEmbeddings(DashScopeEmbeddings())
vs = get_vectorstore()

print("=" * 70)
print("检查失败问题的 ground truth 文件是否存在于文档目录和索引中")
print("=" * 70)

for gt in failed_gt:
    fname = os.path.basename(gt)

    # 1. 文档目录中是否存在
    full_path = os.path.join(docs_dir, gt)
    file_exists = os.path.exists(full_path)

    # 2. 搜索相似文件
    similar = list(gb.glob(os.path.join(docs_dir, '**', fname), recursive=True))

    # 3. 索引中是否存在 (搜索 source 包含此文件名)
    results = vs.similarity_search('dify', k=200)
    found_in_index = []
    for r in results:
        src = r.metadata.get('source', '')
        if fname.lower() in src.lower():
            found_in_index.append(src)
    found_in_index = list(set(found_in_index))

    print(f"\n{gt}")
    print(f"  文件存在: {file_exists}")
    if similar:
        print(f"  相似文件: {similar[:3]}")
    if found_in_index:
        print(f"  索引中存在: {len(found_in_index)} 个 source")
        for s in found_in_index[:3]:
            print(f"    → {s}")
    else:
        print(f"  索引中存在: ❌ 没有找到包含 '{fname}' 的 source")

# 4. 也查下通过的问题，对比一下
print("\n" + "=" * 70)
print("对比: 通过的问题的 ground truth (验证匹配逻辑)")
print("=" * 70)

passed_gt = [
    'api-reference/guides/agent.mdx',
    'use-dify/nodes/ifelse.mdx',
    'use-dify/nodes/http-request.mdx',
    'self-host/deploy/quick-start/docker-compose.mdx',
    'use-dify/knowledge/metadata.mdx',
]

for gt in passed_gt:
    fname = os.path.basename(gt)
    results = vs.similarity_search('dify', k=200)
    found = []
    for r in results:
        src = r.metadata.get('source', '')
        if fname.lower() in src.lower():
            found.append(src)
    found_unique = list(set(found))
    print(f"\n{gt}")
    print(f"  索引中: {len(found_unique)} source(s)")
    for s in found_unique[:2]:
        print(f"    → {s}")
