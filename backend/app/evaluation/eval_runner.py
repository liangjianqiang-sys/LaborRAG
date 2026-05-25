"""评估执行器：三阶段执行流程（生成答案 → 检索指标 → RAGAS+响应指标）。

三元组评估体系：
  - triad.context_relevancy:  上下文相关性（检索质量）
  - triad.faithfulness:       忠实度（生成质量）
  - triad.answer_relevancy:   答案相关性（端到端）
"""
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI
from app.evaluation.eval_dataset import EVAL_DATASET
from app.evaluation.retrieval_metrics import (
    compute_retrieval_metrics,
    compute_retrieval_metrics_by_type,
)
from app.evaluation.response_metrics import (
    compute_response_metrics,
    compute_response_metrics_by_type,
)
from app.evaluation.utils import sanitize_floats, strip_per_query
from app.config import settings
from app.core.embeddings import get_embeddings

_SKIP_COLS = frozenset(('question', 'answer', 'contexts', 'ground_truth', 'question_type'))

# 中间结果保存目录（相对于backend根目录）
_BACKEND_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CACHE_DIR = os.path.join(_BACKEND_ROOT, "evaluation_reports", "cache")


def _mean_scores(df, skip_extra=frozenset()) -> Dict[str, float]:
    """从DataFrame提取指标列均值。"""
    scores = {}
    skip = _SKIP_COLS | skip_extra
    for col in df.columns:
        if col in skip:
            continue
        try:
            scores[col] = round(float(df[col].mean()), 4)
        except (TypeError, ValueError):
            continue
    return scores


class EvalRunner:
    """评估执行器，对RAG系统进行三元组量化评估。"""

    METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
    MAX_RETRIES = 3  # RAGAS评分最大重试次数

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.ragas_llm = ChatOpenAI(
            model=settings.EVAL_LLM_MODEL_NAME,
            openai_api_key=settings.EVAL_LLM_API_KEY,
            openai_api_base=settings.EVAL_LLM_API_BASE,
            temperature=0,
            max_tokens=4096,
            request_timeout=120,
        )
        self.ragas_embeddings = get_embeddings()

    def run(self, sample_count: int = None, rag_mode: str = None, question_type: str = None) -> Dict[str, Any]:
        """运行评估（三阶段：生成答案 → 检索指标 → RAGAS+响应指标）。

        Args:
            sample_count: 评估样本数量，None表示全部
            rag_mode: 指定RAG模式评估，None使用当前配置
            question_type: 过滤问题类型(retrieve/calculate/compare)，None表示全部
        """
        dataset = EVAL_DATASET
        if question_type:
            dataset = [d for d in dataset if d.get("question_type") == question_type]
        dataset = dataset[:sample_count] if sample_count else dataset

        original_mode = settings.RAG_MODE
        if rag_mode:
            settings.RAG_MODE = rag_mode
            from app.core.rag_engine import RAGEngine
            self.rag_engine = RAGEngine()
            self.rag_engine.initialize()

        try:
            return self._execute(dataset, rag_mode or original_mode)
        finally:
            if rag_mode:
                settings.RAG_MODE = original_mode

    def _execute(self, dataset: list, rag_mode_label: str) -> Dict[str, Any]:
        """三阶段执行核心逻辑。"""
        from app.models.schemas import ChatRequest

        # ── 阶段1：生成答案（token消耗60-70%）──
        questions, answers, contexts, ground_truths = [], [], [], []
        sources_raw: List[List[Dict]] = []  # 用于检索指标
        relevant_articles_list: List[List[str]] = []
        question_types: List[str] = []
        details = []
        total = len(dataset)
        print(f"[Phase 1] Generating answers on {total} samples (mode={rag_mode_label})...")

        for i, item in enumerate(dataset, 1):
            print(f"  [{i}/{total}] {item['question'][:30]}...")
            response = self.rag_engine.chat(ChatRequest(question=item["question"]))
            questions.append(item["question"])
            answers.append(response.answer)
            contexts.append([src.content for src in response.sources])
            ground_truths.append(item["ground_truth"])
            relevant_articles_list.append(item.get("relevant_articles", []))
            question_types.append(item.get("question_type", "retrieve"))

            # 保存原始source信息（用于检索指标计算）
            src_list = [{"source": s.source, "content": s.content} for s in response.sources]
            sources_raw.append(src_list)

            details.append({
                "question": item["question"],
                "question_type": item.get("question_type", "retrieve"),
                "law": item.get("law", ""),
                "answer": response.answer,
                "ground_truth": item["ground_truth"],
                "relevant_articles": item.get("relevant_articles", []),
                "source_count": len(response.sources),
                "rag_mode": response.rag_mode,
            })

        # 立即保存中间结果，评分失败时可直接重试
        self._save_cache(rag_mode_label, questions, answers, contexts, ground_truths, details)
        print(f"[Phase 1] Done. Answers cached. Safe to retry if scoring fails.")

        # ── 阶段2：检索指标计算（零LLM调用）──
        print(f"[Phase 2] Computing retrieval metrics (zero LLM calls)...")
        retrieval = compute_retrieval_metrics(sources_raw, relevant_articles_list, k=5)
        type_retrieval = compute_retrieval_metrics_by_type(
            sources_raw, relevant_articles_list, question_types, k=5
        )
        # 将每题检索指标写入details
        for i, pq in enumerate(retrieval.get("per_query", [])):
            details[i]["retrieval"] = {k: v for k, v in pq.items() if k not in ("retrieved_ids", "relevant_ids")}
            details[i]["retrieved_ids"] = pq.get("retrieved_ids", [])
            details[i]["relevant_ids"] = pq.get("relevant_ids", [])
        print(f"[Phase 2] Done. retrieval={_summary(retrieval)}")

        # ── 阶段3：RAGAS + 响应指标（token消耗30-40%，可重试）──
        scores, type_scores, faithfulness_per_query = self._score_with_retry(
            questions, answers, contexts, ground_truths, details
        )

        # 响应指标（ROUGE-L/BLEU零成本，幻觉率复用faithfulness，完整性需LLM）
        print(f"[Phase 3] Computing response metrics...")
        response_metrics = compute_response_metrics(
            answers, ground_truths,
            faithfulness_scores=faithfulness_per_query,
            llm=self.ragas_llm,  # 复用评估LLM计算完整性
        )
        type_response = compute_response_metrics_by_type(
            answers, ground_truths, question_types,
            faithfulness_scores=faithfulness_per_query,
            llm=self.ragas_llm,
        )
        # 将每题响应指标写入details
        for i, pq in enumerate(response_metrics.get("per_query", [])):
            details[i]["response"] = pq
        print(f"[Phase 3] Done. response={_summary(response_metrics)}")

        # ── 构建三元组核心指标 ──
        triad = {
            "context_relevancy": scores.get("context_recall", 0.0),
            "faithfulness": scores.get("faithfulness", 0.0),
            "answer_relevancy": scores.get("answer_relevancy", 0.0),
        }

        return {
            "triad": triad,
            "retrieval": strip_per_query(retrieval),
            "response": strip_per_query(response_metrics),
            "scores": scores,
            "type_scores": type_scores,
            "type_retrieval": {k: strip_per_query(v) for k, v in type_retrieval.items()},
            "type_response": {k: strip_per_query(v) for k, v in type_response.items()},
            "rag_mode": rag_mode_label,
            "sample_count": total,
            "details": details,
        }

    def _score_with_retry(self, questions, answers, contexts, ground_truths, details) -> tuple:
        """RAGAS评分，带重试逻辑。返回 (scores, type_scores, faithfulness_per_query)。"""
        ds = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                print(f"[Phase 3] Computing RAGAS metrics (attempt {attempt}/{self.MAX_RETRIES})...")
                result = evaluate(
                    ds, metrics=self.METRICS,
                    llm=self.ragas_llm, embeddings=self.ragas_embeddings,
                )
                result_df = result.to_pandas()
                scores = _mean_scores(result_df)
                type_scores = {}
                if any(d["question_type"] != "retrieve" for d in details):
                    result_df["question_type"] = [d["question_type"] for d in details]
                    for qtype in set(d["question_type"] for d in details):
                        type_scores[qtype] = _mean_scores(result_df[result_df["question_type"] == qtype])

                # 提取每题faithfulness（用于幻觉率计算）
                faithfulness_per_query = []
                if "faithfulness" in result_df.columns:
                    for val in result_df["faithfulness"]:
                        try:
                            faithfulness_per_query.append(float(val))
                        except (TypeError, ValueError):
                            faithfulness_per_query.append(None)

                return scores, type_scores, faithfulness_per_query
            except Exception as e:
                print(f"[Phase 3] RAGAS attempt {attempt} failed: {e}")
                if attempt < self.MAX_RETRIES:
                    wait = attempt * 30
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"[Phase 3] All {self.MAX_RETRIES} attempts failed. Returning partial results.")
                    return {}, {}, []

    def _save_cache(self, rag_mode_label, questions, answers, contexts, ground_truths, details):
        """保存中间结果（答案生成后立即保存，评分失败可重试）。"""
        os.makedirs(_CACHE_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = os.path.join(_CACHE_DIR, f"answers_{rag_mode_label}_{ts}.json")
        data = sanitize_floats({
            "rag_mode": rag_mode_label,
            "timestamp": ts,
            "questions": questions,
            "answers": answers,
            "contexts": contexts,
            "ground_truths": ground_truths,
            "details": details,
        })
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Cache saved: {cache_file}")


def _summary(metrics: Dict) -> str:
    """生成指标摘要字符串，用于日志输出。"""
    items = [f"{k}={v}" for k, v in metrics.items() if k != "per_query" and isinstance(v, (int, float))]
    return "{" + ", ".join(items) + "}"
