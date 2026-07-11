"""
阿里云百炼原生 Embedding — 直连 API, 绕过兼容层翻译 bug。
"""

from __future__ import annotations

import time
from typing import List

import requests
from pydantic import BaseModel, Field

from dify_rag.config import EMBEDDING_API_KEY, EMBEDDING_MODEL, EMBEDDING_URL


class DashScopeEmbeddings(BaseModel):
    """阿里云百炼原生 Embedding, 实现 LangChain Embeddings 接口。

    为什么不用 OpenAIEmbeddings + compatible-mode?
      → 百炼兼容层把 OpenAI 的 "input" 字段翻译为嵌套的 "contents",
        翻译过程中某些批次触发类型校验失败, 报:
          "contents is neither str nor list of str."
        换成直连原生 API 彻底绕过这个问题。
    """

    model: str = Field(default=EMBEDDING_MODEL)
    api_key: str = Field(default=EMBEDDING_API_KEY, repr=False)
    url: str = Field(default=EMBEDDING_URL)
    max_retries: int = 3
    retry_delay: float = 1.0

    model_config = {"extra": "forbid"}

    def _call_api(self, texts: List[str], text_type: str = "document") -> List[List[float]]:
        """调用百炼原生 embedding API, 带重试。

        Args:
            texts: 待编码文本列表。
            text_type: text-embedding-v4 必须区分 "query" (检索) 和 "document" (入库)。
                       v1/v3 忽略此参数。
        """
        # 过滤空白
        texts = [t for t in texts if t.strip()]
        if not texts:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {"texts": texts},
        }
        # text-embedding-v4 是不对称模型，必须区分 query/document
        if "v4" in self.model:
            payload["parameters"] = {"text_type": text_type}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url, json=payload, headers=headers, timeout=60
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("output", {}).get("embeddings", [])
                    embeddings.sort(key=lambda e: e.get("text_index", 0))
                    return [e["embedding"] for e in embeddings]
                elif resp.status_code == 401:
                    raise RuntimeError(
                        f"百炼 API 认证失败（401），请检查 DASHSCOPE_API_KEY 是否正确: "
                        f"{resp.text[:300]}"
                    )
                elif resp.status_code == 429:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                elif resp.status_code >= 500:
                    last_error = resp.text
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    raise RuntimeError(
                        f"百炼 Embedding API 返回 {resp.status_code}: {resp.text[:300]}"
                    )
            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(f"百炼 Embedding API 调用失败: {last_error}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding, Chroma 入库用 — 使用 text_type="document"。"""
        batch_size = 10  # text-embedding-v4 限制每批最多 10 条
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._call_api(batch, text_type="document"))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """单条 embedding, 检索用 — 使用 text_type="query"。"""
        results = self._call_api([text], text_type="query")
        if results:
            return results[0]
        raise RuntimeError("Embedding 返回为空")
