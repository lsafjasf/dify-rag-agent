"""
深度诊断文档清洗流水线 —— 对各类源文档运行清洗，发现问题。
"""
import sys
sys.path.insert(0, '.')
from pathlib import Path
from dify_rag.text_cleaner import clean_text, clean_file, CleanConfig

ROOT = Path('C:/Users/Administrator/Desktop/zh')

# 选取代表性文档
SAMPLES = [
    # 1. Steps/Step 嵌套 + 链接密集 (之前在 knowledge.mdx 看到)
    "api-reference/guides/knowledge.mdx",
    # 2. Tabs/Tab 组件
    "cloud/use-dify/build/agent.mdx",
    # 3. Card/CardGroup
    "cloud/use-dify/knowledge/readme.mdx",
    # 4. 代码块密集
    "cli/reference/environment-variables.mdx",
    # 5. HTML 表格
    "api-reference/guides/errors.mdx",
    # 6. 首页
    "home.mdx",
]

for rel_path in SAMPLES:
    fp = ROOT / rel_path
    if not fp.exists():
        print(f"\n{'='*60}\nSKIP (not found): {rel_path}\n")
        continue

    raw = fp.read_text(encoding='utf-8')
    cleaned = clean_text(raw)

    print(f"\n{'='*60}")
    print(f"FILE: {rel_path}")
    print(f"RAW:  {len(raw):>6} chars")
    print(f"CLEAN: {len(cleaned):>6} chars ({len(cleaned)/max(1,len(raw))*100:.0f}%)")
    print(f"{'='*60}")

    # 检查是否有残留的 HTML/MDX 标签
    import re
    residual_tags = re.findall(r'<[A-Z][a-zA-Z]*(?:\s[^>]*)?>', cleaned)
    if residual_tags:
        print(f"  ⚠️ 残留标签: {residual_tags[:10]}")

    residual_close = re.findall(r'</[A-Z][a-zA-Z]*>', cleaned)
    if residual_close:
        print(f"  ⚠️ 残留闭合标签: {residual_close[:10]}")

    # 检查残留的 Markdown 链接语法
    residual_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', cleaned)
    if residual_links:
        print(f"  ⚠️ 残留 Markdown 链接: {residual_links[:5]}")

    # 检查残留图片
    residual_imgs = re.findall(r'!\[.*?\]\(.*?\)', cleaned)
    if residual_imgs:
        print(f"  ⚠️ 残留图片: {residual_imgs[:5]}")

    # 检查代码块标记残留
    residual_fence = re.findall(r'^```', cleaned, re.MULTILINE)
    if residual_fence:
        print(f"  ⚠️ 残留代码围栏: {len(residual_fence)} 处")

    # 检查空行占比
    lines = cleaned.split('\n')
    empty_lines = sum(1 for l in lines if not l.strip())
    print(f"  Lines: {len(lines)}, empty: {empty_lines} ({empty_lines/max(1,len(lines))*100:.0f}%)")

    # 打印前 500 字符
    print(f"  --- FIRST 500 CHARS ---")
    print(cleaned[:500])
