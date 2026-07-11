"""
RAG 评估标注测试集
================
内置 ~30 条中文问答对, 覆盖 Dify 文档的主要领域。
每个样本标注了问题、参考答案和相关文档文件名, 用于检索评估和生成评估。

领域覆盖:
  - Agent / 新 Agent API
  - Chatbot / Chatflow API
  - 知识库 API
  - Workflow 节点 (LLM / 工具 / HTTP / 循环 / If-Else 等)
  - 部署 (Docker / 源码)
  - MCP / 外部知识库 / 工具认证
  - 模型配置 / 监控
  - 标注系统 / 对话变量
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class EvalSample:
    """单条评估样本。

    Attributes:
        question: 用户问题 (中文)。
        ground_truth_answer: 参考答案 (可为空, 用于 Answer Correctness 评估)。
        relevant_sources: 相关文档的文件名列表 (不含路径, 用于检索评估的 source 匹配)。
        category: 问题所属领域, 便于分组分析。
    """

    question: str
    ground_truth_answer: str = ""
    relevant_sources: List[str] = field(default_factory=list)
    category: str = "通用"


# ---------------------------------------------------------------------------
# 内置测试集 (30 条)
# ---------------------------------------------------------------------------

BUILTIN_DATASET: List[EvalSample] = [
    # ===== Agent / 新 Agent (3 条) =====
    EvalSample(
        question="新 Agent 应用和旧的 Agent 应用（agent-chat）有什么区别？",
        ground_truth_answer="新 Agent 是 Dify 1.16.0-rc1 引入的 Beta 功能，运行在 agent 模式下。与旧 Agent 不同，新 Agent 通过 agent_message 事件增量流式返回回复文本，每个推理步骤和工具调用以 agent_thought 事件同步返回。旧 Agent 应用返回 agent_thought 和 agent_message 事件，而聊天助手应用只返回 message 事件。",
        relevant_sources=["api-reference/guides/agent.mdx"],
        category="Agent",
    ),
    EvalSample(
        question="新 Agent API 支持阻塞式返回吗？",
        ground_truth_answer="不支持。新 Agent 应用仅支持流式返回，使用阻塞式返回会收到 400 bad_request 错误。",
        relevant_sources=["api-reference/guides/agent.mdx"],
        category="Agent",
    ),
    EvalSample(
        question="如何在 Dify 中创建一个能够自动调用工具的 AI Agent？",
        ground_truth_answer="可以在 Dify 中创建 Agent 应用（agent-chat 模式）或新 Agent 应用。Agent 会自主推理并调用工具（如知识库检索、API 调用等），推理过程和工具调用结果通过 agent_thought 事件返回。需要配置 Agent 策略、选择工具并设置提示词。",
        relevant_sources=["use-dify/build/agent.mdx", "api-reference/guides/agent.mdx"],
        category="Agent",
    ),

    # ===== Chatflow API (3 条) =====
    EvalSample(
        question="Chatflow 应用和普通的 Chatbot 应用在 API 返回事件上有什么不同？",
        ground_truth_answer="Chatflow 应用运行在 advanced-chat 模式下，除了回复文本外还会流式传输工作流级别的事件，包括节点开始、完成、迭代、暂停等事件。普通 Chatbot 应用只返回 message 事件。",
        relevant_sources=["api-reference/guides/chatflow.mdx"],
        category="Chatflow",
    ),
    EvalSample(
        question="如何通过 API 停止 Chatflow 应用的流式回复？",
        ground_truth_answer="调用停止响应接口可以在回复完成前中断流式输出。",
        relevant_sources=["api-reference/guides/chatflow.mdx"],
        category="Chatflow",
    ),
    EvalSample(
        question="Chatflow 中的人工介入节点暂停后，如何通过 API 恢复执行？",
        ground_truth_answer="当工作流到达人工介入节点时，事件流会推送 human_input_required 事件（携带 form_token 和 workflow_run_id），然后以 workflow_paused 事件结束。需要通过以下步骤恢复：1) 用 form_token 调用获取人工介入表单接口；2) 上传需要的文件；3) 调用提交人工介入表单接口提交填写内容；4) 用 workflow_run_id 调用流式获取工作流事件接口重新连接事件流。",
        relevant_sources=["api-reference/guides/chatflow.mdx", "api-reference/guides/human-input-flow.mdx"],
        category="Chatflow",
    ),

    # ===== 知识库 API (4 条) =====
    EvalSample(
        question="如何通过 API 创建一个新的知识库？",
        ground_truth_answer="调用创建空知识库 API 可以创建一个还没有文档的知识库。可以设置名称、权限、嵌入模型和检索设置。如果创建时未设置 indexing_technique，可以在上传第一个文档时设置，后续文档会自动继承。",
        relevant_sources=["api-reference/guides/knowledge.mdx"],
        category="知识库",
    ),
    EvalSample(
        question="通过 API 上传文档到知识库后，如何知道文档索引完成了？",
        ground_truth_answer="文档创建是异步的，需要轮询获取文档嵌入状态接口。使用创建文档时返回的 batch ID，不断查询 indexing_status，直到变为 completed 或 error。期间会依次经过 waiting、parsing、cleaning、splitting、indexing 几个阶段。",
        relevant_sources=["api-reference/guides/knowledge.mdx"],
        category="知识库",
    ),
    EvalSample(
        question="知识库 API 支持哪些检索方式？",
        ground_truth_answer="知识库 API 支持从知识库检索分段/测试检索接口，可以搜索知识库并返回最相关的分段。生产检索和召回测试共用同一个接口。检索设置（如检索方式、top_k、分数阈值等）可以在创建或更新知识库时配置。",
        relevant_sources=["api-reference/guides/knowledge.mdx"],
        category="知识库",
    ),
    EvalSample(
        question="如何通过 API 获取知识库中的文档列表？",
        ground_truth_answer="调用获取知识库的文档列表接口，返回分页列表，支持按关键词或索引状态（indexing_status）筛选。还可以通过获取文档详情接口查看单个文档的索引状态、元数据和处理统计信息。",
        relevant_sources=["api-reference/guides/knowledge.mdx"],
        category="知识库",
    ),

    # ===== Workflow 节点 (6 条) =====
    EvalSample(
        question="Dify 工作流中 LLM 节点支持哪些功能？",
        ground_truth_answer="LLM 节点是工作流的核心推理节点，支持选择模型、编写系统提示词和用户提示词、配置记忆（对话历史）窗口、设置温度等参数。可以选择对话补全或文本补全模式，支持结构化输出（JSON Schema）。",
        relevant_sources=["use-dify/nodes/llm.mdx"],
        category="Workflow",
    ),
    EvalSample(
        question="Dify 工作流中如何在 LLM 节点里使用对话历史？",
        ground_truth_answer="在 LLM 节点中开启「记忆」功能，可以设置对话窗口大小（最近 N 轮对话）。节点会自动从会话中获取历史消息并注入到上下文中。可以通过会话变量在不同轮次之间传递信息。",
        relevant_sources=["use-dify/nodes/llm.mdx"],
        category="Workflow",
    ),
    EvalSample(
        question="Dify 工作流中的 If-Else 节点怎么用？",
        ground_truth_answer="If-Else 节点用于条件分支，支持对变量进行逻辑判断（等于、不等于、大于、小于、包含、不包含、为空、不为空等），根据判断结果分别执行不同的分支。可以设置多个条件（ELIF）和默认分支（ELSE）。",
        relevant_sources=["use-dify/nodes/ifelse.mdx"],
        category="Workflow",
    ),
    EvalSample(
        question="Dify 工作流中 HTTP 请求节点支持哪些请求方法和鉴权方式？",
        ground_truth_answer="HTTP 请求节点支持 GET、POST、PUT、PATCH、DELETE、HEAD、OPTIONS 等请求方法。鉴权支持无鉴权、Basic Auth、Bearer Token、API Key（Header/Query）、自定义鉴权等方式。可以自定义请求头、查询参数和请求体。",
        relevant_sources=["use-dify/nodes/http-request.mdx"],
        category="Workflow",
    ),
    EvalSample(
        question="Dify 工作流中的循环节点有什么限制？",
        ground_truth_answer="循环节点用于对数组中的每个元素执行相同的操作。可以设置最大循环次数和超时时间。循环内部可以嵌套其他节点，但需要注意循环嵌套层级的限制和性能影响。支持 break 条件提前退出循环。",
        relevant_sources=["use-dify/nodes/loop.mdx"],
        category="Workflow",
    ),
    EvalSample(
        question="Dify 工作流中如何使用工具节点调用外部 API？",
        ground_truth_answer="工具节点用于在工作流中调用自定义工具（API 工具）。需要先在工具配置中定义 API 的端点、参数、鉴权方式等，然后在工作流中拖入工具节点并选择对应的工具。工具的输入可以从上游节点引用变量，输出可以供下游节点使用。",
        relevant_sources=["use-dify/nodes/tools.mdx"],
        category="Workflow",
    ),

    # ===== 部署相关 (3 条) =====
    EvalSample(
        question="如何使用 Docker Compose 部署 Dify？",
        ground_truth_answer="克隆 Dify 仓库后，进入 docker 目录，复制 .env.example 为 .env 并按需修改配置，然后运行 docker compose up -d 启动所有服务。包括 PostgreSQL、Redis、Weaviate（向量数据库）、Nginx 和 Dify API/Worker/Web 等服务。",
        relevant_sources=["self-host/deploy/quick-start/docker-compose.mdx"],
        category="部署",
    ),
    EvalSample(
        question="从源代码本地部署 Dify 需要哪些前置条件？",
        ground_truth_answer="需要 Python 3.10+、Node.js 18+、PostgreSQL、Redis 和 Weaviate（或其他向量数据库）。后端使用 Flask + Celery，前端使用 Next.js。需要分别启动 API 服务、Worker 进程和前端开发服务器。",
        relevant_sources=["self-host/deploy/advanced-deployments/local-source-code.mdx"],
        category="部署",
    ),
    EvalSample(
        question="Dify 支持哪些向量数据库？",
        ground_truth_answer="Dify 支持多种向量数据库，包括 Weaviate、Qdrant、Milvus、Pinecone、Chroma、PGVector 等。可以通过环境变量配置使用的向量数据库类型和连接参数。不同版本可能有不同的默认数据库。",
        relevant_sources=["self-host/deploy/configuration/environments.mdx"],
        category="部署",
    ),

    # ===== MCP / 外部知识库 (3 条) =====
    EvalSample(
        question="Dify 如何配置 MCP（Model Context Protocol）服务器？",
        ground_truth_answer="在 Dify 中可以通过 MCP 配置接入外部工具。需要提供 MCP 服务器的连接信息（如 SSE 端点或 stdio 命令）。配置后，MCP 提供的工具会自动出现在 Agent 或工作流的可用工具列表中。",
        relevant_sources=["use-dify/publish/publish-mcp.mdx"],
        category="MCP",
    ),
    EvalSample(
        question="Dify 的外部知识库 API 是做什么的？",
        ground_truth_answer="外部知识库 API 允许不经过 Dify 控制台，直接在外部系统中使用 Dify 的知识库检索能力。提供了检索接口用于搜索或 RAG 场景。单个 API 密钥可访问创建该密钥的账户下所有可见的知识库。",
        relevant_sources=["use-dify/knowledge/external-knowledge-api.mdx", "api-reference/guides/knowledge.mdx"],
        category="外部知识库",
    ),
    EvalSample(
        question="Dify 如何从 Notion 导入文档到知识库？",
        ground_truth_answer="Dify 支持从 Notion 同步文档。需要在 Notion 中创建集成（Integration），获取 API Key，然后在 Dify 知识库中选择 Notion 作为数据源，配置授权后即可同步 Notion 中的页面和数据库内容。",
        relevant_sources=["use-dify/knowledge/create-knowledge/import-text-data/sync-from-notion.mdx"],
        category="外部知识库",
    ),

    # ===== 模型与监控 (3 条) =====
    EvalSample(
        question="Dify 支持哪些模型供应商？",
        ground_truth_answer="Dify 支持 OpenAI、Anthropic（Claude）、Google（Gemini）、DeepSeek、阿里云百炼、百度文心、讯飞星火、智谱 AI、月之暗面（Moonshot/Kimi）、MiniMax 等众多模型供应商。此外还支持通过 OpenAI 兼容接口接入自定义模型。",
        relevant_sources=["use-dify/workspace/model-providers.mdx"],
        category="模型",
    ),
    EvalSample(
        question="如何在 Dify 中接入 LangSmith 进行可观测性监控？",
        ground_truth_answer="在 Dify 中配置 LangSmith 需要设置 LangSmith API Key 和相关环境变量。配置后，Dify 会将 LLM 调用的 trace 数据发送到 LangSmith，可以在 LangSmith 平台上查看调用链路、延迟、token 用量和错误信息。",
        relevant_sources=["use-dify/monitor/integrations/integrate-langsmith.mdx"],
        category="监控",
    ),
    EvalSample(
        question="Dify 支持哪些可观测性/监控平台集成？",
        ground_truth_answer="Dify 支持集成 LangSmith、Langfuse、Opik、Phoenix（Arize）、W&B Weave 等可观测性平台。通过配置相应的 API Key 和环境变量即可启用，用于追踪 LLM 调用、性能监控和调试。",
        relevant_sources=["use-dify/monitor/integrations/"],
        category="监控",
    ),

    # ===== 应用与 WebApp (3 条) =====
    EvalSample(
        question="如何将 Dify 应用嵌入到自己的网站中？",
        ground_truth_answer="Dify 提供了 WebApp 嵌入方案。在应用的概览页面可以获取嵌入代码（iframe 或 script 标签），将其复制到网站 HTML 中即可。还可以自定义 WebApp 的主题、Logo、按钮样式等。也可以通过 API 完全自定义前端。",
        relevant_sources=["use-dify/publish/webapp/embedding-in-websites.mdx"],
        category="WebApp",
    ),
    EvalSample(
        question="Dify 的标注系统有什么作用？",
        ground_truth_answer="标注系统用于为应用预设固定答案。命中标注问题时，应用直接返回预设答案，不再调用 LLM 生成新回复。可以通过 API 创建、更新、删除标注，也可以配置标注回复的开关。适合用于 FAQ 场景或需要精确控制回答的场景。",
        relevant_sources=["use-dify/monitor/annotation-reply.mdx"],
        category="WebApp",
    ),
    EvalSample(
        question="Dify 中如何为不同终端用户提供个性化体验？",
        ground_truth_answer="通过 API 的 user 字段区分终端用户。不同用户可以有不同的会话历史、对话变量和文件访问权限。在知识库中也可以按用户进行数据隔离。WebApp 支持用户登录和身份认证。",
        relevant_sources=["api-reference/guides/chat.mdx", "api-reference/guides/end-user-identity.mdx"],
        category="WebApp",
    ),

    # ===== 对话变量与文件 (2 条) =====
    EvalSample(
        question="Dify 的对话变量（Conversation Variables）是什么？怎么用？",
        ground_truth_answer="对话变量是在多轮对话间保留的变量，可以在工作流中读取和更新。通过获取对话变量接口读取当前值，通过更新对话变量接口修改值。适合用于在多轮对话中保存用户偏好、收集信息或追踪状态。",
        relevant_sources=["api-reference/guides/chatflow.mdx", "use-dify/build/workflow-chatflow.mdx"],
        category="对话管理",
    ),
    EvalSample(
        question="通过 API 上传文件到 Dify 有什么限制？",
        ground_truth_answer="通过上传文件接口可以上传图片、文档、音频或视频文件。文件仅限上传的终端用户使用（通过 user 字段关联）。文件可在发送对话消息时作为附件引用。支持的文件类型和大小限制取决于应用的配置。",
        relevant_sources=["api-reference/guides/chat.mdx"],
        category="对话管理",
    ),
]


def get_default_dataset() -> List[EvalSample]:
    """返回内置的默认评估数据集。"""
    return list(BUILTIN_DATASET)


def load_dataset_from_json(path: str | Path) -> List[EvalSample]:
    """从 JSON 文件加载评估数据集。

    JSON 格式:
    [
        {
            "question": "...",
            "ground_truth_answer": "...",
            "relevant_sources": ["file1.mdx", "file2.mdx"],
            "category": "Agent"
        },
        ...
    ]

    Args:
        path: JSON 文件路径。

    Returns:
        EvalSample 列表。
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: List[EvalSample] = []
    for item in data:
        samples.append(EvalSample(
            question=item["question"],
            ground_truth_answer=item.get("ground_truth_answer", ""),
            relevant_sources=item.get("relevant_sources", []),
            category=item.get("category", "通用"),
        ))
    return samples


def save_dataset_to_json(samples: List[EvalSample], path: str | Path) -> None:
    """将评估数据集保存为 JSON 文件。

    Args:
        samples: EvalSample 列表。
        path: 输出 JSON 文件路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "question": s.question,
            "ground_truth_answer": s.ground_truth_answer,
            "relevant_sources": s.relevant_sources,
            "category": s.category,
        }
        for s in samples
    ]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_summary(samples: List[EvalSample]) -> str:
    """返回数据集的统计摘要。

    Args:
        samples: EvalSample 列表。

    Returns:
        格式化的摘要字符串。
    """
    from collections import Counter

    categories = Counter(s.category for s in samples)
    has_gt = sum(1 for s in samples if s.ground_truth_answer)
    has_sources = sum(1 for s in samples if s.relevant_sources)

    lines = [
        f"📊 数据集摘要",
        f"   总样本数: {len(samples)}",
        f"   有参考答案: {has_gt}/{len(samples)}",
        f"   有标注来源: {has_sources}/{len(samples)}",
        f"   领域分布:",
    ]
    for cat, count in categories.most_common():
        lines.append(f"     - {cat}: {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: 导出内置测试集为 JSON (方便编辑)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python eval_dataset.py <output.json>")
        print("  将内置测试集导出为 JSON 文件, 方便手动编辑和扩展。")
        sys.exit(1)

    out_path = sys.argv[1]
    samples = get_default_dataset()
    save_dataset_to_json(samples, out_path)
    print(f"✅ 已导出 {len(samples)} 条样本至: {out_path}")
    print(dataset_summary(samples))
