"""
ChromaDB metadata 兼容工具 — 过滤嵌套结构。
"""

from __future__ import annotations

import json


def filter_complex_metadata(metadata: dict) -> dict:
    """过滤 metadata 中的嵌套结构，ChromaDB 只接受 str/int/float/bool/None/List。

    - 嵌套 dict → JSON 字符串
    - list → 保持 (ChromaDB 原生支持 list of simple values)
    - 其余类型原样返回
    """
    result = {}
    for key, value in metadata.items():
        if isinstance(value, dict):
            result[key] = json.dumps(value, ensure_ascii=False)
        elif value is None or isinstance(value, (str, int, float, bool, list)):
            result[key] = value
        else:
            result[key] = str(value)
    return result
