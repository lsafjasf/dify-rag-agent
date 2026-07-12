"""
逐步追踪清洗流水线，精确定位每个问题。
"""
import sys
sys.path.insert(0, '.')
from dify_rag.text_cleaner import (
    extract_frontmatter, remove_translation_notice,
    strip_mdx_tags, strip_html_tags, clean_markdown,
    normalize_whitespace, normalize_chinese_punctuation,
    clean_text
)

# ==== 测试1: home.mdx - 丢失77%内容 ====
print("="*60)
print("TEST 1: home.mdx")
print("="*60)

raw = open('C:/Users/Administrator/Desktop/zh/home.mdx', encoding='utf-8').read()
meta, body = extract_frontmatter(raw)

print(f"frontmatter keys: {list(meta.keys())}")
print(f"body length: {len(body)}")
print(f"\n--- RAW BODY ---")
print(body[:800])

print(f"\n--- After strip_mdx_tags ---")
step1 = strip_mdx_tags(body)
print(f"len: {len(step1)}")
print(step1[:600])

print(f"\n--- After strip_html_tags ---")
step2 = strip_html_tags(step1)
print(f"len: {len(step2)}")
print(step2[:600])

print(f"\n--- After clean_markdown ---")
step3 = clean_markdown(step2)
print(f"len: {len(step3)}")
print(step3[:600])

print(f"\n--- After normalize_whitespace ---")
step4 = normalize_whitespace(step3)
print(f"len: {len(step4)}")
print(step4[:600])

print(f"\n--- FINAL ---")
final = clean_text(body)
print(f"len: {len(final)}")
print(final[:800])

# ==== 测试2: errors.mdx - 代码块和表格 ====
print("\n\n" + "="*60)
print("TEST 2: errors.mdx — 代码块+表格")
print("="*60)

raw2 = open('C:/Users/Administrator/Desktop/zh/api-reference/guides/errors.mdx', encoding='utf-8').read()
print(f"raw: {len(raw2)} chars")
final2 = clean_text(raw2)
print(f"clean: {len(final2)} chars")
print(final2[:800])

# ==== 测试3: 检查 strip_mdx_tags 对 props={} 的处理 ====
print("\n\n" + "="*60)
print("TEST 3: MDX props with {} — Columns cols={2}")
print("="*60)

test_mdx = """<Columns cols={2}>
<Card title="Cloud" icon="cloud" href="/zh/cloud">
在托管平台上立即开始构建。
</Card>
<Card title="Self-Host" icon="server" href="/zh/self-host">
在你自己的基础设施上运行。
</Card>
</Columns>"""

result = strip_mdx_tags(test_mdx)
print(f"Input:  {repr(test_mdx[:100])}...")
print(f"Output: {repr(result[:200])}")

# ==== 测试4: 检查 Card 的 href 属性中的 URL 是否保留 ====
print("\n\n" + "="*60)
print("TEST 4: Card href URL 保留检查")
print("="*60)

test_card = """<Card title="构建你的第一个应用" icon="rocket" href="/zh/quick-start" horizontal>
刚接触 Dify？从这里开始。
</Card>"""

# 模拟完整清洗
result = clean_text(test_card)
print(f"Input:  {test_card.strip()}")
print(f"Output: {result}")
print(f"URL '/zh/quick-start' in output: {'/zh/quick-start' in result}")
