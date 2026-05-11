"""RAGAS评估执行器：运行评估并计算指标。"""
import os
import time
from typing import List, Dict, Any
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
from app.config import settings
from app.core.embeddings import get_embeddings


class EvalRunner:
    """RAGAS评估执行器，对RAG系统进行量化评估。"""

    METRICS = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        # 为ragas配置LLM（使用百炼API）
        self.ragas_llm = ChatOpenAI(
            model=settings.LLM_MODEL_NAME,
            openai_api_key=settings.LLM_API_KEY,
            openai_api_base=settings.LLM_API_BASE,
            temperature=0,
            max_tokens=2048,
        )
        # 为ragas配置Embedding（使用本地BGE-M3）
        self.ragas_embeddings = get_embeddings()

    def run(self, sample_count: int = None) -> Dict[str, Any]:
        """运行RAGAS评估。

        Args:
            sample_count: 评估样本数量，None表示全部

        Returns:
            评估结果字典，包含各指标分数和详情
        """
        dataset = EVAL_DATASET[:sample_count] if sample_count else EVAL_DATASET

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        print(f"Running evaluation on {len(dataset)} samples...")
        for i, item in enumerate(dataset):
            print(f"  [{i+1}/{len(dataset)}] {item['question'][:30]}...")

            # 调用RAG引擎获取回答
            from app.models.schemas import ChatRequest
            request = ChatRequest(question=item["question"])
            response = self.rag_engine.chat(request)

            questions.append(item["question"])
            answers.append(response.answer)
            contexts.append([src.content for src in response.sources])
            ground_truths.append(item["ground_truth"])

        # 构建RAGAS数据集
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # 运行RAGAS评估，指定LLM和Embedding
        print("Computing RAGAS metrics...")
        result = evaluate(
            eval_dataset,
            metrics=self.METRICS,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
        )

        # 转换为可序列化的结果
        # ragas 0.4.x 返回 EvaluationResult 对象
        try:
            result_df = result.to_pandas()
            scores = {}
            for col in result_df.columns:
                # 跳过非指标列
                if col in ('question', 'answer', 'contexts', 'ground_truth'):
                    continue
                try:
                    val = float(result_df[col].mean())
                    scores[col] = round(val, 4)
                except (TypeError, ValueError):
                    # 无法转为数值的列跳过
                    continue
        except Exception as e:
            print(f"Failed to parse EvaluationResult: {e}")
            scores = {}

        return {
            "scores": scores,
            "sample_count": len(dataset),
            "details": [
                {
                    "question": q,
                    "answer": a[:100] + "...",
                    "ground_truth": gt[:100] + "...",
                }
                for q, a, gt in zip(questions, answers, ground_truths)
            ],
        }
