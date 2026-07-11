"""
LLM 工厂 — 懒加载 DeepSeek ChatOpenAI 实例。
"""

from __future__ import annotations

_llm_cache = None


def get_llm():
    """获取配置好的 LLM 实例 (懒加载缓存)。

    Returns:
        ChatOpenAI 实例 (langchain_openai)。
    """
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    from langchain_openai import ChatOpenAI
    from dify_rag.config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY

    llm_kwargs: dict = {"model": LLM_MODEL, "temperature": 0.3}
    if LLM_BASE_URL:
        llm_kwargs["base_url"] = LLM_BASE_URL
    if LLM_API_KEY:
        llm_kwargs["api_key"] = LLM_API_KEY
    _llm_cache = ChatOpenAI(**llm_kwargs)
    return _llm_cache
