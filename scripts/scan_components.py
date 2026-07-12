"""扫描所有 mdx 文件，统计包含哪些 MDX 组件和元素。"""
import re
from pathlib import Path

root = Path('C:/Users/Administrator/Desktop/zh')
components = {
    'Steps': 0, 'Step': 0, 'Tabs': 0, 'Tab': 0,
    'Card': 0, 'CardGroup': 0, 'Accordion': 0, 'AccordionGroup': 0,
    'Columns': 0, 'Frame': 0, 'Info': 0, 'Warning': 0,
    'Note': 0, 'Tip': 0, 'Check': 0,
    'table': 0, 'code_block': 0, 'img': 0,
}
for f in sorted(root.rglob('*.mdx')):
    content = f.read_text(encoding='utf-8')
    for tag in ['Steps','Step','Tabs','Tab','Card','CardGroup',
                 'Accordion','AccordionGroup','Columns','Frame',
                 'Info','Warning','Note','Tip','Check']:
        if f'<{tag}' in content:
            components[tag] += 1
    if '<table' in content.lower():
        components['table'] += 1
    if '```' in content:
        components['code_block'] += 1
    if '![' in content:
        components['img'] += 1

for k, v in sorted(components.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v} files')
print(f'  total mdx: {sum(1 for _ in root.rglob("*.mdx"))}')
print(f'  total json: {sum(1 for _ in root.rglob("*.json"))}')
