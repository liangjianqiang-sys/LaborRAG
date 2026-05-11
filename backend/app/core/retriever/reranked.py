from typing import List, Tuple
from langchain_core.documents import Document
from app.core.retriever.base import BaseRetriever
from app.core.retriever.hybrid import HybridRetriever
from app.config import settings


class RerankedRetriever(BaseRetriever):
    """V2: 带重排序的混合检索器。先Hybrid检索候选集，再用Cross-Encoder精排。"""

    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid_retriever = hybrid_retriever
        self._reranker = None

    def _get_reranker(self):
        """懒加载Reranker模型。"""
        if self._reranker is None:
            import os
            from sentence_transformers import CrossEncoder
            # 设置HuggingFace镜像源（国内加速）
            if settings.HF_ENDPOINT and "mirror" in settings.HF_ENDPOINT:
                os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
            print(f"Loading reranker model: {settings.RERANKER_MODEL_NAME}...")
            self._reranker = CrossEncoder(settings.RERANKER_MODEL_NAME, device="cpu")
            print("Reranker model loaded.")
        return self._reranker

    def retrieve(
        self, query: str, k: int = 5, score_threshold: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """混合检索 + 重排序。"""
        if not self.is_ready():
            return []

        # 1. 先用Hybrid检索获取候选集（扩大到Top20）
        candidate_k = min(k * 4, 20)
        candidates = self.hybrid_retriever.retrieve(
            query, k=candidate_k, score_threshold=0.0
        )

        if not candidates:
            return []

        # 2. Cross-Encoder重排序
        reranker = self._get_reranker()
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = reranker.predict(pairs)

        # 3. 按重排序分数排序
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 4. 取Top-K
        results = []
        for (doc, _old_score), rerank_score in scored_results[:k]:
            # Cross-Encoder分数归一化到0-1
            normalized = 1.0 / (1.0 + max(0, -rerank_score))
            if normalized >= score_threshold:
                results.append((doc, round(normalized, 4)))
        return results

    def is_ready(self) -> bool:
        return self.hybrid_retriever.is_ready()
