"""
Verify all discovered hidden issues in the cleaning pipeline.
"""
import sys, re, json
sys.path.insert(0, '.')
from pathlib import Path
from dify_rag.text_cleaner import clean_text, strip_mdx_tags, normalize_whitespace

print("=" * 60)
print("ISSUE 1: Card component href/title attributes discarded")
print("=" * 60)

test1 = '<Card title="Build First App" icon="rocket" href="/zh/quick-start" horizontal>\nNew to Dify? Start here.\n</Card>'
print(f"  Input: {test1.strip()}")
print(f"  Output: {clean_text(test1).strip()}")
has_href = '/zh/quick-start' in clean_text(test1)
has_title = 'Build First App' in clean_text(test1)
print(f"  [{'PASS' if has_href else 'FAIL'}] href=/zh/quick-start preserved: {has_href}")
print(f"  [{'PASS' if has_title else 'FAIL'}] title='Build First App' preserved: {has_title}")
print()

print("=" * 60)
print("ISSUE 2: JSON code block braces removed by min_line_length")
print("=" * 60)

test2 = '```json\n{\n  "code": "invalid_param",\n  "message": "user is required"\n}\n```'
print(f"  Input: {repr(test2)}")
result2 = clean_text(test2)
print(f"  Output: {repr(result2.strip())}")
has_braces = '{' in result2 and '}' in result2
print(f"  [{'PASS' if has_braces else 'FAIL'}] JSON braces preserved: {has_braces}")
print()

# Verify root cause
print("  --- Root cause: normalize_whitespace min_line_length=2 ---")
lines_test = "line1\n{\nline2\n}\nline3"
r1 = normalize_whitespace(lines_test, min_line_length=2)
r2 = normalize_whitespace(lines_test, min_line_length=1)
print(f"  Input: {repr(lines_test)}")
print(f"  min_line_length=2: {repr(r1)}")
print(f"  min_line_length=1: {repr(r2)}")
print()

print("=" * 60)
print("ISSUE 3: 6 OpenAPI JSON files not indexed")
print("=" * 60)

json_files = sorted(Path('C:/Users/Administrator/Desktop/zh').rglob('*.json'))
total_paths = 0
for jf in json_files:
    size_kb = jf.stat().st_size / 1024
    with open(jf, encoding='utf-8') as f:
        data = json.load(f)
    paths_count = len(data.get('paths', {})) if isinstance(data, dict) else 0
    total_paths += paths_count
    endpoints = list(data.get('paths', {}).keys())[:3] if isinstance(data, dict) else []
    print(f"  {jf.name}: {size_kb:.0f}KB, {paths_count} API paths, e.g. {endpoints}")
print(f"  [FAIL] {len(json_files)} OpenAPI JSON files with {total_paths} API endpoints NOT indexed!")
print()

print("=" * 60)
print("ISSUE 4: Card content without title/href loses semantic meaning")
print("=" * 60)
# Show what home.mdx cards become
home_cards = """<Columns cols={4}>
<Card title="API Reference" icon="code" href="/zh/api-reference/guides/get-started">
Integrate Dify apps via REST API into your own products.
</Card>
<Card title="CLI" icon="terminal" href="/zh/cli/overview">
Run your Dify apps from terminal, scripts, CI and AI Agent via difyctl.
</Card>
</Columns>"""
result = clean_text(home_cards)
print(f"  Input: has titles 'API Reference', 'CLI' and URLs")
print(f"  Output: {result.strip()[:200]}")
print(f"  Card titles preserved: {'API Reference' in result}")
print(f"  Card URLs preserved: {'/zh/api-reference' in result}")
print(f"  [FAIL] All Card title/href info is lost!")
