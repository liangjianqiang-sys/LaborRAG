from typing import List, Tuple
from langchain_core.documents import Document
from app.core.retriever.base import BaseRetriever
from app.core.retriever.vector import VectorRetriever
from app.core.retriever.bm25 import BM25Retriever
from app.config import settings


class HybridRetriever(BaseRetriever):
    """V2: 混合检索器，向量检索 + BM25关键词检索 + RRF融合。"""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k  # RRF常数，通常60

    def retrieve(
        self, query: str, k: int = 5, score_threshold: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """混合检索：两路并行 + RRF融合。"""
        if not self.is_ready():
            return []

        # 候选集扩大，确保融合后有足够结果
        candidate_k = min(k * 4, 20)

        # 1. 向量检索
        vector_results = self.vector_retriever.retrieve(
            query, k=candidate_k, score_threshold=0.0
        )

        # 2. BM25检索
        bm25_results = self.bm25_retriever.retrieve(
            query, k=candidate_k, score_threshold=0.0
        )

        # 3. Reciprocal Rank Fusion (RRF) 融合
        doc_scores: dict[str, float] = {}  # content_hash -> rrf_score
        doc_map: dict[str, Document] = {}  # content_hash -> Document

        # 向量检索结果融合
        for rank, (doc, _score) in enumerate(vector_results):
            key = doc.page_content
            doc_scores[key] = doc_scores.get(key, 0.0) + self.vector_weight / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        # BM25检索结果融合
        for rank, (doc, _score) in enumerate(bm25_results):
            key = doc.page_content
            doc_scores[key] = doc_scores.get(key, 0.0) + self.bm25_weight / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        # 4. 按RRF分数排序，取Top-K
        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for content, rrf_score in sorted_items[:k]:
            doc = doc_map[content]
            # RRF分数归一化到0-1范围（近似）
            normalized_score = min(rrf_score * self.rrf_k, 1.0)
            if normalized_score >= score_threshold:
                results.append((doc, round(normalized_score, 4)))
        return results

    def is_ready(self) -> bool:
        return self.vector_retriever.is_ready() and self.bm25_retriever.is_ready()
