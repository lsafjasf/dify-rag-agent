"""
Reranker — 对 RRF 融合候选进行精度重排序
=========================================
支持多种后端，按优先级自动降级：

  1. 阿里云百炼 DashScope Rerank API (gte-rerank) —— 零额外依赖，复用现有 API Key
  2. FlagEmbedding 本地模型 (BAAI/bge-reranker-v2-m3) —— 需 pip install FlagEmbedding
  3. LLM 打分 —— 用 DeepSeek 逐条评估相关性（最慢但零依赖）
  4. None —— 直接返回原始 RRF 顺序

用法:
  from bge_reranker import BGEReranker
  reranker = BGEReranker(
      dashscope_api_key="sk-xxx",   # 用现有百炼 Key, 优先使用
      llm=llm,                       # LLM fallback
  )
  top_docs = reranker.rerank(query, candidates, top_k=5)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Reranker 接口
# ---------------------------------------------------------------------------


class BaseReranker:
    """Reranker 基类。"""

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """对候选文档重新排序并返回 top_k。"""
        if not documents:
            return []
        if len(documents) <= top_k:
            return documents
        scores = self.compute_scores(query, documents)
        # 将分数写入 metadata 便于调试
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = round(score, 4)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    def compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        """计算每个文档与查询的相关性分数。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 1) 阿里云百炼 DashScope Rerank API (首选, 零额外依赖)
# ---------------------------------------------------------------------------


class DashScopeReranker(BaseReranker):
    """使用阿里云百炼原生 Rerank API (gte-rerank 模型) 进行精排。

    优势:
      - 零额外依赖，直接复用 DASHSCOPE_API_KEY
      - 云端推理，不占本地资源
      - 专为中文优化的交叉编码器

    注意: 需要在百炼控制台开通 gte-rerank 模型服务。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen3-rerank",
        url: str = "https://ws-czxggevcvus6bxx5.cn-beijing.maas.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        max_retries: int = 2,
    ):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.max_retries = max_retries
        # 验证 API 可用性（轻量调用）
        self._validate()

    def _validate(self):
        """发送最小请求验证 API Key 是否有权限。"""
        import requests

        payload = {
            "model": self.model,
            "input": {
                "query": "test",
                "documents": ["test document"],
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(
                self.url, json=payload, headers=headers, timeout=15
            )
            if resp.status_code == 200:
                return  # OK
            data = resp.json()
            code = data.get("code", "")
            if code == "AccessDenied":
                raise RuntimeError(
                    f"百炼 Rerank API 未开通。请在控制台开通 gte-rerank 模型: "
                    f"https://help.aliyun.com/zh/model-studio/error-code#access-denied"
                )
            # 其他错误不抛 — 可能是临时问题，运行时再处理
        except requests.RequestException as e:
            raise RuntimeError(f"百炼 Rerank API 连接失败: {e}")

    def compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        """调用百炼 Rerank API 计算相关性分数。"""
        import requests

        if not documents:
            return []

        # 提取文档文本
        texts = [doc.page_content for doc in documents]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": {
                "query": query,
                "documents": texts,
            },
            "parameters": {
                "return_documents": False,
            },
        }

        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.url, json=payload, headers=headers, timeout=30
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("output", {}).get("results", [])
                    # 构建 index → score 映射
                    score_map: Dict[int, float] = {}
                    for r in results:
                        score_map[r.get("index", 0)] = r.get("relevance_score", 0.5)
                    # 按原始顺序返回分数
                    return [score_map.get(i, 0.0) for i in range(len(texts))]

                elif resp.status_code == 401:
                    raise RuntimeError(
                        f"百炼 Rerank API 认证失败（401），请检查 DASHSCOPE_API_KEY: "
                        f"{resp.text[:200]}"
                    )
                elif resp.status_code >= 500:
                    last_error = resp.text
                    time.sleep(1.0 * (attempt + 1))
                else:
                    last_error = resp.text
                    break  # 4xx 不重试

            except requests.RequestException as e:
                last_error = str(e)
                time.sleep(1.0 * (attempt + 1))

        print(f"  ⚠️ DashScope Rerank API 失败: {last_error}")
        # 降级：返回等分
        return [0.5] * len(documents)


# ---------------------------------------------------------------------------
# 2) FlagEmbedding 本地模型 (备选, 需额外安装)
# ---------------------------------------------------------------------------


class FlagReranker(BaseReranker):
    """基于 FlagEmbedding 的本地 BGE Reranker。

    需要: pip install FlagEmbedding  (自动装 torch + transformers)
    模型: BAAI/bge-reranker-v2-m3 (~2GB, 首次自动下载)
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._model = None
        self._device = device

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker as _FlagReranker
            self._model = _FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self._device,
            )
        return self._model

    def compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        if not documents:
            return []
        pairs = [[query, doc.page_content] for doc in documents]
        try:
            raw_scores = self.model.compute_score(pairs, normalize=True)
            if isinstance(raw_scores, list):
                return [float(s) for s in raw_scores]
            return [float(raw_scores)]
        except Exception as exc:
            print(f"  ⚠️ FlagReranker 失败: {exc}")
            return [0.5] * len(documents)


# ---------------------------------------------------------------------------
# 3) LLM 打分 (兜底, 零额外依赖)
# ---------------------------------------------------------------------------


class LLMReranker(BaseReranker):
    """使用 LLM 对候选文档逐条打分。

    不依赖任何额外模型。缺点是较慢且消耗 token。
    """

    SCORE_PROMPT = """评估以下文档片段与用户问题的相关性，给出 0-10 的分数。

## 用户问题
{question}

## 文档片段
{text}

## 评分标准
- 10: 完全直接回答问题的核心内容
- 7-9: 高度相关，包含问题所需的大部分信息
- 4-6: 部分相关，涉及问题的某个侧面
- 1-3: 弱相关，只有少数关键词匹配
- 0: 完全不相关

请仅返回一个 0-10 的数字分数，不要输出其他内容。"""

    def __init__(self, llm, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries

    def compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        import re

        if not documents:
            return []

        scores: List[float] = []
        for doc in documents:
            prompt = self.SCORE_PROMPT.format(question=query, text=doc.page_content[:2000])
            score = 0.5  # 默认中等
            for attempt in range(self.max_retries):
                try:
                    response = self.llm.invoke(prompt)
                    content = response.content if hasattr(response, "content") else str(response)
                    m = re.search(r"(\d+(?:\.\d+)?)", content.strip())
                    if m:
                        score = max(0.0, min(1.0, float(m.group(1)) / 10.0))
                    break
                except Exception:
                    if attempt < self.max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
            scores.append(score)

        return scores


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------


class BGEReranker(BaseReranker):
    """Reranker 统一入口，自动选择最佳可用后端。

    检测优先级:
      1. DashScope Rerank API (gte-rerank) — 提供 api_key 则使用
      2. FlagEmbedding 本地模型 — 已安装则使用
      3. LLM 打分 — 提供 llm 则使用
      4. 跳过 — 返回原始顺序
    """

    def __init__(
        self,
        dashscope_api_key: str = "",
        llm=None,
    ):
        """
        Args:
            dashscope_api_key: 百炼 API Key。提供后优先使用云端 Rerank API。
            llm: LLM 实例（最终兜底）。
        """
        self._backend: Optional[BaseReranker] = None

        # 1) 优先 DashScope Rerank API
        if dashscope_api_key:
            try:
                self._backend = DashScopeReranker(api_key=dashscope_api_key)
                print("✅ Reranker: 阿里云百炼 DashScope Rerank API (gte-rerank)")
                return
            except Exception as exc:
                self._backend = None
                print(f"⚠️ DashScope Rerank 不可用: {exc}")

        # 2) 其次 FlagEmbedding 本地模型
        try:
            self._backend = FlagReranker()
            _ = self._backend.model  # 触发加载
            print("✅ Reranker: BGE-Reranker 本地模型 (FlagEmbedding)")
            return
        except Exception as exc:
            self._backend = None
            print(f"⚠️ FlagEmbedding 不可用: {exc}")

        # 3) 最后 LLM 打分
        if llm is not None:
            self._backend = LLMReranker(llm=llm)
            print("✅ Reranker: LLM 打分 (DeepSeek)")
            return

        # 4) 无可用后端
        self._backend = None
        print("⚠️ 无可用 Reranker，跳过精排")

    @property
    def is_available(self) -> bool:
        return self._backend is not None

    def compute_scores(self, query: str, documents: List[Document]) -> List[float]:
        if not self._backend or not documents:
            return [0.5] * len(documents)
        return self._backend.compute_scores(query, documents)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        if not self._backend:
            return documents[:top_k]
        return self._backend.rerank(query, documents, top_k)


# ---------------------------------------------------------------------------
# 轻量测试
# ---------------------------------------------------------------------------


def _test():
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            temperature=0.0,
        )
    except ImportError:
        llm = None

    reranker = BGEReranker(dashscope_api_key=api_key, llm=llm)

    from langchain_core.documents import Document
    docs = [
        Document(page_content="Dify 支持多种向量数据库，包括 Weaviate、Qdrant 和 Milvus。", metadata={"source": "1"}),
        Document(page_content="今天天气很好，适合出去散步。", metadata={"source": "2"}),
        Document(page_content="向量数据库通过 embedding 模型将文本转换为向量进行相似度搜索。", metadata={"source": "3"}),
    ]
    query = "Dify 支持哪些向量数据库？"

    scores = reranker.compute_scores(query, docs)
    for doc, score in zip(docs, scores):
        print(f"  [{score:.3f}] {doc.page_content[:60]}...")

    top = reranker.rerank(query, docs, top_k=2)
    print(f"\n  Top-2:")
    for i, doc in enumerate(top, 1):
        print(f"  [{i}] {doc.page_content[:60]}...")


if __name__ == "__main__":
    _test()
