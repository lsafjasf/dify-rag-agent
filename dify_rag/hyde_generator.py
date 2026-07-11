"""
HyDE (Hypothetical Document Embeddings) 生成器
=============================================
使用 LLM 从用户问题生成假设性的技术文档段落，用于替代原始问题进行向量检索。

HyDE 原理:
  用户问题通常是简短的自然语言句子，而文档库中存储的是长篇技术文档。
  直接对问题做 embedding 会导致语义空间不匹配。
  HyDE 先让 LLM 根据问题"编"一段可能存在的技术文档，
  再对这段假设文档做 embedding 进行检索 — 因为"伪文档"使用的
  词汇和句法更接近真实文档，检索效果更好。

用法:
  from hyde_generator import HyDEGenerator
  hyde = HyDEGenerator(llm)
  hypothetical_doc = hyde.generate("如何配置 Dify 的 MCP 服务器？")
  # hypothetical_doc 是一段 ~300 字的技术文档文本
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


DEFAULT_HYDE_PROMPT_CN = """你是一个Dify平台的技术文档撰写专家。请根据以下问题，撰写一段技术文档段落来回答它。

要求：
- 使用技术文档的写作风格，包含API名称、参数、配置步骤等技术细节
- 写200-400字的技术文档段落
- 不要使用"根据文档"、"文档中说"等引用语，直接写技术内容
- 如果是API相关问题，请写出API端点、请求参数、返回格式等具体信息
- 如果是配置相关问题，请写出配置步骤和注意事项
- 使用中文撰写，但保留API名称、参数名等英文原文

问题：{question}

请直接输出技术文档段落（不要输出JSON或其他格式）："""


class HyDEGenerator:
    """HyDE 假设文档生成器。

    使用 LLM 将用户问题转换为假设性的技术文档段落，
    生成的文本用于替代原问题进行向量检索。
    """

    def __init__(
        self,
        llm,
        prompt_template: str = DEFAULT_HYDE_PROMPT_CN,
        max_retries: int = 2,
    ):
        """
        Args:
            llm: LangChain LLM 实例 (如 ChatOpenAI)。
            prompt_template: 提示词模板，需包含 `{question}` 占位符。
            max_retries: LLM 调用最大重试次数。
        """
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries

    def generate(self, question: str) -> str:
        """根据用户问题生成假设性技术文档段落。

        Args:
            question: 用户问题（中文）。

        Returns:
            假设的技术文档文本。如果 LLM 调用失败，返回原始问题作为降级。
        """
        import time

        prompt = self.prompt_template.format(question=question)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(prompt)
                text = response.content if hasattr(response, "content") else str(response)
                text = text.strip()

                # 如果 LLM 意外返回了 JSON 格式，尝试提取文本
                if text.startswith("{") and text.endswith("}"):
                    try:
                        data = json.loads(text)
                        # 尝试常见 key
                        for key in ("document", "text", "content", "answer", "response"):
                            if key in data:
                                text = data[key]
                                break
                    except json.JSONDecodeError:
                        pass

                # 如果被包裹在代码块中，去除标记
                m = re.search(r"```(?:markdown|text)?\s*([\s\S]*?)```", text)
                if m:
                    text = m.group(1).strip()

                if len(text) >= 20:  # 合理的长度下限
                    return text

                # 太短，可能有问题
                last_error = f"生成的文档太短 ({len(text)} 字符)"

            except Exception as exc:
                last_error = str(exc)
                if attempt < self.max_retries - 1:
                    time.sleep(1.0 * (attempt + 1))

        # 降级：返回原始问题
        print(f"  ⚠️ HyDE 生成失败 ({last_error})，降级使用原始问题")
        return question


def _test():
    """简单测试 HyDE 生成。"""
    import os
    import sys
    from dotenv import load_dotenv
    load_dotenv(override=True)

    # 需要 LLM
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("需要安装 langchain-openai")
        return

    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        temperature=0.3,
    )

    hyde = HyDEGenerator(llm)
    questions = [
        "如何在 Dify 中创建一个能够自动调用工具的 AI Agent？",
        "Dify 工作流中 LLM 节点支持哪些功能？",
        "如何通过 API 上传文件到 Dify？",
    ]

    for q in questions:
        doc = hyde.generate(q)
        print(f"\n{'='*60}")
        print(f"问题: {q}")
        print(f"生成文档 ({len(doc)} 字):")
        print(doc[:500])


if __name__ == "__main__":
    _test()
