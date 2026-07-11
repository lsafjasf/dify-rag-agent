"""
RAG 效果评估引擎
================
双模式评估:

  模式 A: 检索评估 (纯 Python 实现, 无额外依赖)
    - Hit Rate@K
    - MRR (Mean Reciprocal Rank)
    - Precision@K / Recall@K

  模式 B: RAGAS 全链路评估 (需 pip install ragas)
    - Faithfulness (忠实度)
    - Answer Relevancy (回答相关性)
    - Context Precision (上下文精度)
    - Answer Correctness (答案正确性)

用法:
  from eval_rag import run_evaluation
  report = run_evaluation(dataset, mode="all", llm=..., embeddings=...)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_dataset import EvalSample

# ---------------------------------------------------------------------------
# 可选依赖检测
# ---------------------------------------------------------------------------

RAGAS_AVAILABLE = False
RAGAS_VERSION: Optional[str] = None

try:
    import ragas  # noqa: F401

    # 检测 RAGAS 版本
    try:
        from ragas.dataset_schema import SingleTurnSample  # noqa: F401
        from ragas.metrics import (  # noqa: F401
            AnswerCorrectness,
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
        )

        RAGAS_AVAILABLE = True
        RAGAS_VERSION = "0.2.x"
    except ImportError:
        try:
            from ragas.metrics import (  # noqa: F401
                answer_correctness,
                answer_relevancy,
                faithfulness,
            )

            RAGAS_AVAILABLE = True
            RAGAS_VERSION = "0.1.x"
        except ImportError:
            RAGAS_AVAILABLE = False
except ImportError:
    RAGAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# 匹配工具
# ---------------------------------------------------------------------------


def _match_source(retrieved_source: str, relevant_file_names: List[str]) -> bool:
    """检查检索到的 source 路径是否匹配任一相关文件名。

    支持 full path → filename 匹配, 也支持相对路径匹配。

    Args:
        retrieved_source: 检索结果中的 source 路径 (如 'C:\\...\\agent.mdx')。
        relevant_file_names: 标注的相关文件名列表 (如 ['agent.mdx', 'chat.mdx'])。

    Returns:
        是否匹配。
    """
    retrieved_lower = retrieved_source.replace("\\", "/").lower()
    for name in relevant_file_names:
        name_lower = name.replace("\\", "/").lower()
        if name_lower in retrieved_lower:
            return True
    return False


# ---------------------------------------------------------------------------
# 模式 A: 检索评估
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """单条样本的检索评估结果。"""

    hit: bool = False
    first_relevant_rank: Optional[int] = None  # 从 1 开始; None 表示未命中
    precision: float = 0.0
    recall: float = 0.0


@dataclass
class RetrievalReport:
    """检索评估汇总报告。"""

    hit_rate: float = 0.0
    mrr: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    total_questions: int = 0
    k: int = 5
    per_sample: List[RetrievalMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "retrieval",
            "total_questions": self.total_questions,
            "k": self.k,
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
        }

    def format_table(self) -> str:
        return (
            f"┌──────────────────────┬──────────┐\n"
            f"│ 指标                   │ 值       │\n"
            f"├──────────────────────┼──────────┤\n"
            f"│ Hit Rate@{self.k}         │ {self.hit_rate:>7.1%} │\n"
            f"│ MRR                    │ {self.mrr:>7.4f} │\n"
            f"│ Precision@{self.k}        │ {self.precision_at_k:>7.1%} │\n"
            f"│ Recall@{self.k}           │ {self.recall_at_k:>7.1%} │\n"
            f"├──────────────────────┼──────────┤\n"
            f"│ 样本数                   │ {self.total_questions:>6d}   │\n"
            f"└──────────────────────┴──────────┘"
        )


def evaluate_retrieval(
    dataset: List[EvalSample],
    retrieve_fn,
    k: int = 5,
) -> RetrievalReport:
    """运行检索评估。

    对每条问题调用 retrieve_fn 获取 top-k 文档, 与标注的
    relevant_sources 对比计算检索指标。

    Args:
        dataset: 评估样本列表。
        retrieve_fn: 检索函数, 签名为 (question: str, k: int) -> List[dict]。
                     每个 dict 需包含 "source" (str) 字段。
        k: Top-K 参数。

    Returns:
        RetrievalReport 对象。
    """
    per_sample: List[RetrievalMetrics] = []

    for sample in dataset:
        try:
            docs = retrieve_fn(sample.question, k=k)
        except Exception as exc:
            print(f"  ⚠️ 检索失败 [{sample.question[:30]}...]: {exc}")
            per_sample.append(RetrievalMetrics())
            continue

        # 提取 source
        retrieved_sources = [
            d.get("source", d.get("metadata", {}).get("source", ""))
            if isinstance(d, dict)
            else getattr(getattr(d, "metadata", None), "source", "")
            if hasattr(d, "metadata")
            else ""
            for d in docs
        ]

        relevant = sample.relevant_sources

        # Hit Rate
        hit = any(
            _match_source(src, relevant) for src in retrieved_sources
        )

        # First relevant rank
        first_rank: Optional[int] = None
        for rank, src in enumerate(retrieved_sources, 1):
            if _match_source(src, relevant):
                first_rank = rank
                break

        # Precision@K
        hits = sum(
            1 for src in retrieved_sources if _match_source(src, relevant)
        )
        precision = hits / len(retrieved_sources) if retrieved_sources else 0.0

        # Recall@K
        recall = (
            hits / len(relevant) if relevant else 0.0
        )

        per_sample.append(RetrievalMetrics(
            hit=hit,
            first_relevant_rank=first_rank,
            precision=precision,
            recall=recall,
        ))

    # 汇总
    n = len(per_sample)
    hit_rate = sum(1 for m in per_sample if m.hit) / n if n else 0.0
    mrr = sum(
        1.0 / m.first_relevant_rank
        for m in per_sample
        if m.first_relevant_rank
    ) / n if n else 0.0
    precision_at_k = sum(m.precision for m in per_sample) / n if n else 0.0
    recall_at_k = sum(m.recall for m in per_sample) / n if n else 0.0

    return RetrievalReport(
        hit_rate=hit_rate,
        mrr=mrr,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        total_questions=n,
        k=k,
        per_sample=per_sample,
    )


# ---------------------------------------------------------------------------
# 模式 B: RAGAS 评估
# ---------------------------------------------------------------------------


@dataclass
class RAGASReport:
    """RAGAS 全链路评估汇总报告。"""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    answer_correctness: float = 0.0
    total_questions: int = 0
    per_sample: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": "ragas",
            "total_questions": self.total_questions,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "answer_correctness": round(self.answer_correctness, 4),
        }

    def format_table(self) -> str:
        return (
            f"┌──────────────────────────┬──────────┐\n"
            f"│ RAGAS 指标                 │ 值       │\n"
            f"├──────────────────────────┼──────────┤\n"
            f"│ Faithfulness (忠实度)       │ {self.faithfulness:>7.1%} │\n"
            f"│ Answer Relevancy (相关性)   │ {self.answer_relevancy:>7.1%} │\n"
            f"│ Context Precision (上下文)  │ {self.context_precision:>7.1%} │\n"
            f"│ Answer Correctness (正确性) │ {self.answer_correctness:>7.1%} │\n"
            f"├──────────────────────────┼──────────┤\n"
            f"│ 样本数                       │ {self.total_questions:>6d}   │\n"
            f"└──────────────────────────┴──────────┘"
        )


def evaluate_ragas(
    dataset: List[EvalSample],
    generated_results: List[Dict[str, Any]],
    llm,
    embeddings,
) -> RAGASReport:
    """使用 RAGAS 运行全链路评估。

    Args:
        dataset: 评估样本列表。
        generated_results: 每个样本的生成结果, 格式:
            [{"question": str, "answer": str, "contexts": List[str]}, ...]
        llm: LangChain LLM 实例 (用于 RAGAS judge)。
        embeddings: LangChain Embeddings 实例 (用于 RAGAS 部分指标)。

    Returns:
        RAGASReport 对象。

    Raises:
        ImportError: 如果 ragas 未安装。
    """
    if not RAGAS_AVAILABLE:
        raise ImportError(
            "RAGAS 未安装。请执行: pip install ragas"
        )

    from langchain_openai import ChatOpenAI

    # 包装 LLM 为 RAGAS 兼容格式
    try:
        from ragas.llms import LangchainLLMWrapper
        judge_llm = LangchainLLMWrapper(llm)
    except ImportError:
        judge_llm = llm

    # 包装 Embeddings
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        judge_embeddings = LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        judge_embeddings = embeddings

    # 构建 RAGAS 样本
    if RAGAS_VERSION and RAGAS_VERSION.startswith("0.2"):
        # RAGAS 0.2.x API
        from ragas import evaluate
        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            AnswerCorrectness,
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithoutReference,
        )

        samples = []
        for item, sample in zip(generated_results, dataset):
            samples.append(SingleTurnSample(
                user_input=item["question"],
                response=item["answer"],
                retrieved_contexts=item["contexts"],
                reference=sample.ground_truth_answer or None,
            ))

        metrics = [
            Faithfulness(llm=judge_llm),
            AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
            LLMContextPrecisionWithoutReference(llm=judge_llm),
        ]
        if any(s.ground_truth_answer for s in dataset):
            metrics.append(AnswerCorrectness(llm=judge_llm, embeddings=judge_embeddings))

        result = evaluate(
            metrics=metrics,
            dataset=samples,
        )

    elif RAGAS_VERSION and RAGAS_VERSION.startswith("0.1"):
        # RAGAS 0.1.x API (旧版)
        from ragas import evaluate
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            faithfulness,
        )

        ds = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        for item, sample in zip(generated_results, dataset):
            ds["question"].append(item["question"])
            ds["answer"].append(item["answer"])
            ds["contexts"].append(item["contexts"])
            ds["ground_truth"].append(sample.ground_truth_answer or "")

        from datasets import Dataset
        hf_dataset = Dataset.from_dict(ds)

        result = evaluate(
            metrics=[faithfulness, answer_relevancy, answer_correctness],
            dataset=hf_dataset,
        )
    else:
        raise RuntimeError("RAGAS 版本无法识别")

    # 解析结果
    result_dict = dict(result) if hasattr(result, "__iter__") else {}

    def _safe_float(key: str) -> float:
        v = result_dict.get(key, 0.0)
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    report = RAGASReport(
        faithfulness=_safe_float("faithfulness"),
        answer_relevancy=_safe_float("answer_relevancy"),
        context_precision=_safe_float("llm_context_precision_without_reference"),
        answer_correctness=_safe_float("answer_correctness"),
        total_questions=len(dataset),
    )

    return report


# ---------------------------------------------------------------------------
# 综合评估
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """综合评估报告。"""

    retrieval: Optional[RetrievalReport] = None
    ragas: Optional[RAGASReport] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "config": self.config,
        }
        if self.retrieval:
            result["retrieval"] = self.retrieval.to_dict()
        if self.ragas:
            result["ragas"] = self.ragas.to_dict()
        return result

    def format_summary(self) -> str:
        lines = [
            "=" * 60,
            "📊 RAG 评估报告",
            f"⏰ 时间: {self.timestamp}",
            f"📋 配置: {json.dumps(self.config, ensure_ascii=False)}",
            "=" * 60,
        ]
        if self.retrieval:
            lines.append("")
            lines.append("🔍 检索评估结果")
            lines.append(self.retrieval.format_table())
        if self.ragas:
            lines.append("")
            lines.append("🧪 RAGAS 全链路评估结果")
            lines.append(self.ragas.format_table())
        if self.retrieval is None and self.ragas is None:
            lines.append("")
            lines.append("⚠️ 没有可用的评估结果。")
        lines.append("")
        return "\n".join(lines)


def run_evaluation(
    dataset: List[EvalSample],
    *,
    mode: str = "all",
    llm=None,
    embeddings=None,
    retrieve_fn=None,
    generate_fn=None,
    top_k: int = 5,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> EvalReport:
    """运行完整的 RAG 评估流水线。

    Args:
        dataset: 评估样本列表。
        mode: 评估模式 — "retrieval" | "ragas" | "all"。
        llm: LangChain LLM 实例 (RAGAS 模式必需)。
        embeddings: LangChain Embeddings 实例 (RAGAS 模式必需)。
        retrieve_fn: 检索函数 (question, k) -> List[dict]。
        generate_fn: 生成函数 (question, contexts) -> str。
                     如果不提供, 则使用简单的拼接+LLM调用。
        top_k: 检索 Top-K 参数。
        config: 评估配置信息 (记录在报告中)。
        verbose: 是否打印进度信息。

    Returns:
        EvalReport 对象。
    """
    mode = mode.lower()
    if mode not in ("retrieval", "ragas", "all"):
        raise ValueError(f"无效的评估模式: {mode}。可选: retrieval | ragas | all")

    if config is None:
        config = {}
    config["top_k"] = top_k
    config["mode"] = mode
    config["dataset_size"] = len(dataset)

    report = EvalReport(config=config)
    generated_results: List[Dict[str, Any]] = []

    # ---- 检索评估 ----
    if mode in ("retrieval", "all"):
        if retrieve_fn is None:
            raise ValueError("检索评估需要提供 retrieve_fn")

        if verbose:
            print(f"\n🔍 运行检索评估 (Top-{top_k})...")
            print(f"   样本数: {len(dataset)}")

        report.retrieval = evaluate_retrieval(
            dataset=dataset,
            retrieve_fn=retrieve_fn,
            k=top_k,
        )

        if verbose:
            print(f"   ✅ 检索评估完成")
            for i, m in enumerate(report.retrieval.per_sample):
                status = "✅" if m.hit else "❌"
                rank_str = f" rank={m.first_relevant_rank}" if m.first_relevant_rank else ""
                q_preview = dataset[i].question[:50]
                print(f"   {status} [{i+1:2d}] {q_preview}...{rank_str}")

    # ---- RAGAS 评估 ----
    if mode in ("ragas", "all"):
        if not RAGAS_AVAILABLE:
            print("⚠️ RAGAS 未安装, 跳过全链路评估。执行: pip install ragas")
        elif llm is None:
            print("⚠️ 缺少 LLM 实例, 跳过 RAGAS 评估。")
        elif retrieve_fn is None:
            print("⚠️ 缺少 retrieve_fn, 跳过 RAGAS 评估。")
        else:
            if verbose:
                print(f"\n🧪 运行 RAGAS 全链路评估...")

            # 先生成所有回答
            for i, sample in enumerate(dataset):
                if verbose:
                    print(f"   [{i+1}/{len(dataset)}] 处理: {sample.question[:50]}...")

                try:
                    # 检索
                    docs = retrieve_fn(sample.question, k=top_k)
                    contexts = [
                        d.get("page_content", d.get("content", str(d)))
                        if isinstance(d, dict)
                        else getattr(d, "page_content", str(d))
                        for d in docs
                    ]
                    # 生成
                    if generate_fn:
                        answer = generate_fn(sample.question, contexts)
                    else:
                        # 降级: 没有 provide_fn 时用空回答
                        answer = "[请提供 generate_fn]"

                    generated_results.append({
                        "question": sample.question,
                        "answer": answer,
                        "contexts": contexts,
                    })
                except Exception as exc:
                    if verbose:
                        print(f"      ⚠️ 失败: {exc}")
                    generated_results.append({
                        "question": sample.question,
                        "answer": f"[生成失败: {exc}]",
                        "contexts": [],
                    })

            # 运行 RAGAS
            try:
                report.ragas = evaluate_ragas(
                    dataset=dataset,
                    generated_results=generated_results,
                    llm=llm,
                    embeddings=embeddings,
                )
                if verbose:
                    print(f"   ✅ RAGAS 评估完成")
            except Exception as exc:
                if verbose:
                    print(f"   ⚠️ RAGAS 评估失败: {exc}")

    # 保存逐样本详情
    for i, sample in enumerate(dataset):
        entry: Dict[str, Any] = {
            "question": sample.question,
            "category": sample.category,
        }
        if report.retrieval and i < len(report.retrieval.per_sample):
            rm = report.retrieval.per_sample[i]
            entry["retrieval"] = {
                "hit": rm.hit,
                "first_relevant_rank": rm.first_relevant_rank,
                "precision": rm.precision,
                "recall": rm.recall,
            }
        if i < len(generated_results):
            entry["answer"] = generated_results[i]["answer"]
        report.per_sample_results.append(entry)

    return report


def save_report(report: EvalReport, path: str | Path) -> None:
    """保存评估报告为 JSON 文件。

    Args:
        report: EvalReport 对象。
        path: 输出路径。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": report.to_dict(),
        "per_sample": report.per_sample_results,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
