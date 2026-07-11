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
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_dataset import EvalSample


# ---------------------------------------------------------------------------
# 轻量级 RAGAS 指标实现 (零外部依赖, 基于现有 LLM + Embedding)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """计算两个向量的余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _call_llm_json(llm, prompt: str, max_retries: int = 2) -> dict:
    """调用 LLM 并尝试从响应中提取 JSON。"""
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            # 尝试直接解析
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            # 尝试匹配 ```json ... ``` 代码块
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if m:
                try:
                    return json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
            # 尝试匹配 { ... }
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                try:
                    return json.loads(m.group())
                except json.JSONDecodeError:
                    pass
            return {}
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.0 * (attempt + 1))
            else:
                return {}
    return {}


def _eval_faithfulness(question: str, answer: str, contexts: List[str], llm) -> float:
    """评估忠实度: 回答是否能从检索到的上下文中推断出来。

    1. 让 LLM 把回答拆解为独立陈述 (claims)
    2. 逐一判断每个陈述是否被上下文支持
    3. faithfulness = 被支持的陈述数 / 总陈述数
    """
    if not answer.strip() or not contexts:
        return 0.0

    ctx_text = "\n\n---\n\n".join(
        f"[{i + 1}] {c}" for i, c in enumerate(contexts)
    )

    prompt = f"""你是一个严格的评估助手。请评估以下回答是否忠实于提供的参考上下文。

## 问题
{question}

## 参考上下文
{ctx_text}

## 回答
{answer}

## 任务
1. 将回答拆解为独立的陈述句 (claims)，每个陈述句应是一个可以独立验证的事实断言。
2. 逐一判断每个陈述句是否可以从参考上下文中推断或得到支持。
3. 以 JSON 格式返回结果，不要输出其他内容。

返回格式：
{{"claims": [{{"statement": "陈述内容", "supported": true/false, "reason": "判断依据"}}]}}"""

    result = _call_llm_json(llm, prompt)
    claims = result.get("claims", [])
    if not claims:
        return 0.5  # 无法解析时返回中间值

    supported = sum(1 for c in claims if c.get("supported", False))
    return supported / len(claims)


def _eval_answer_relevancy(
    question: str, answer: str, llm, embeddings
) -> float:
    """评估回答相关性: 回答是否切题。

    1. 让 LLM 根据回答生成 3 个可能触发该回答的问题
    2. 用 embedding 计算生成问题与原问题的余弦相似度
    3. relevancy = 平均相似度
    """
    if not answer.strip():
        return 0.0

    prompt = f"""针对以下回答，生成 3 个该回答能够回答的问题。问题应语义多样但都与回答内容相关。

## 回答
{answer}

返回格式：
{{"questions": ["问题1", "问题2", "问题3"]}}"""

    result = _call_llm_json(llm, prompt)
    generated = result.get("questions", [])
    if not generated or len(generated) < 1:
        return 0.5

    try:
        orig_emb = embeddings.embed_query(question)
        sims = []
        for gq in generated:
            gen_emb = embeddings.embed_query(gq)
            sims.append(_cosine_similarity(orig_emb, gen_emb))
        return sum(sims) / len(sims)
    except Exception:
        return 0.5


def _eval_context_precision(
    question: str, contexts: List[str], llm
) -> float:
    """评估上下文精度: 检索到的上下文是否与问题相关。

    1. 让 LLM 逐一判断每个上下文片段与问题的相关性
    2. precision = 相关的上下文数 / 总上下文数
    """
    if not contexts:
        return 0.0

    ctx_list = "\n\n".join(
        f"[{i + 1}] {c[:400]}" for i, c in enumerate(contexts)
    )

    prompt = f"""判断以下每个上下文片段是否与回答该问题相关。

## 问题
{question}

## 上下文片段
{ctx_list}

返回格式（数组长度必须为 {len(contexts)}）：
{{"relevant": [true/false, ...]}}"""

    result = _call_llm_json(llm, prompt)
    relevant = result.get("relevant", [])
    if not relevant or len(relevant) != len(contexts):
        return 0.5

    return sum(1 for r in relevant if r) / len(relevant)


def _eval_answer_correctness(
    question: str, answer: str, ground_truth: str, llm, embeddings
) -> float:
    """评估答案正确性: 回答与参考答案的一致程度。

    综合两个维度:
    - LLM 语义评分 (权重 60%): 让 LLM 判断事实正确性和完整性
    - Embedding 相似度 (权重 40%): 计算回答与参考答案的语义向量相似度
    """
    if not answer.strip() or not ground_truth.strip():
        return 0.0

    # 1) Embedding 语义相似度
    try:
        ans_emb = embeddings.embed_query(answer)
        gt_emb = embeddings.embed_query(ground_truth)
        semantic_score = _cosine_similarity(ans_emb, gt_emb)
    except Exception:
        semantic_score = 0.5

    # 2) LLM 语义评判
    prompt = f"""评估以下回答与参考答案的一致程度，考虑事实正确性和信息完整性。

## 问题
{question}

## 回答
{answer}

## 参考答案
{ground_truth}

返回格式：
{{"correctness": 0.0-1.0的分数, "reason": "简短评估"}}"""

    result = _call_llm_json(llm, prompt)
    try:
        llm_score = float(result.get("correctness", semantic_score))
        llm_score = max(0.0, min(1.0, llm_score))
    except (TypeError, ValueError):
        llm_score = semantic_score

    # 综合: LLM 60% + Embedding 40%
    return 0.6 * llm_score + 0.4 * semantic_score


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
    """使用内置轻量指标运行全链路评估 (无需 ragas 依赖)。

    对每条样本顺序计算 4 个指标:
    - Faithfulness: 回答是否忠实于检索上下文
    - Answer Relevancy: 回答是否切题
    - Context Precision: 检索到的上下文是否与问题相关
    - Answer Correctness: 回答与参考答案的一致程度 (需要 ground_truth)

    Args:
        dataset: 评估样本列表。
        generated_results: [{"question": str, "answer": str, "contexts": [str]}, ...]
        llm: LangChain LLM 实例。
        embeddings: Embeddings 实例 (有 embed_query 方法)。

    Returns:
        RAGASReport 对象。
    """
    n = len(dataset)
    faith_scores: List[float] = []
    relevancy_scores: List[float] = []
    precision_scores: List[float] = []
    correctness_scores: List[float] = []
    per_sample: List[Dict[str, Any]] = []

    for i, (sample, gen) in enumerate(zip(dataset, generated_results)):
        question = gen["question"]
        answer = gen["answer"]
        contexts = gen.get("contexts", [])

        entry: Dict[str, Any] = {"question": question[:60]}

        # 1) Faithfulness
        try:
            f = _eval_faithfulness(question, answer, contexts, llm)
            faith_scores.append(f)
            entry["faithfulness"] = round(f, 4)
        except Exception:
            faith_scores.append(0.0)
            entry["faithfulness"] = 0.0

        # 2) Answer Relevancy
        try:
            ar = _eval_answer_relevancy(question, answer, llm, embeddings)
            relevancy_scores.append(ar)
            entry["answer_relevancy"] = round(ar, 4)
        except Exception:
            relevancy_scores.append(0.0)
            entry["answer_relevancy"] = 0.0

        # 3) Context Precision
        try:
            cp = _eval_context_precision(question, contexts, llm)
            precision_scores.append(cp)
            entry["context_precision"] = round(cp, 4)
        except Exception:
            precision_scores.append(0.0)
            entry["context_precision"] = 0.0

        # 4) Answer Correctness (有参考答案时才计算)
        if sample.ground_truth_answer:
            try:
                ac = _eval_answer_correctness(
                    question, answer, sample.ground_truth_answer, llm, embeddings
                )
                correctness_scores.append(ac)
                entry["answer_correctness"] = round(ac, 4)
            except Exception:
                pass  # 不参与平均

        per_sample.append(entry)

    report = RAGASReport(
        faithfulness=sum(faith_scores) / n if n else 0.0,
        answer_relevancy=sum(relevancy_scores) / n if n else 0.0,
        context_precision=sum(precision_scores) / n if n else 0.0,
        answer_correctness=(
            sum(correctness_scores) / len(correctness_scores)
            if correctness_scores
            else 0.0
        ),
        total_questions=n,
        per_sample=per_sample,
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

    # ---- 全链路评估 (内置轻量指标, 无需 ragas) ----
    if mode in ("ragas", "all"):
        if llm is None:
            print("⚠️ 缺少 LLM 实例, 跳过全链路评估。")
        elif retrieve_fn is None:
            print("⚠️ 缺少 retrieve_fn, 跳过全链路评估。")
        else:
            if verbose:
                print(f"\n🧪 运行全链路评估 (内置指标)...")

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

            # 运行全链路评估
            try:
                report.ragas = evaluate_ragas(
                    dataset=dataset,
                    generated_results=generated_results,
                    llm=llm,
                    embeddings=embeddings,
                )
                if verbose:
                    print(f"   ✅ 全链路评估完成")
            except Exception as exc:
                if verbose:
                    print(f"   ⚠️ 全链路评估失败: {exc}")

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
