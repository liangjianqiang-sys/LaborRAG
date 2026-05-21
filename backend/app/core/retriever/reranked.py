from typing import List, Tuple
from langchain_core.documents import Document
from app.core.retriever.base import BaseRetriever
from app.core.retriever.hybrid import HybridRetriever
from app.core.vectorstore import extract_article_ref
from app.config import settings


class RerankedRetriever(BaseRetriever):
    """V2: 带重排序的混合检索器。先Hybrid检索候选集，再用Cross-Encoder精排。

    支持层次化父子分块：重排序在子块级别执行（精准排序），
    排序后再调用 promote_children_to_parents 将子块提升为完整父块。
    """

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
        """混合检索 + 重排序 + 父子分块提升。"""
        if not self.is_ready():
            return []

        # 1. 先用Hybrid检索获取候选集（扩大到Top20）
        #    注意：HybridRetriever内部已做子块→父块提升，
        #    但重排序需要在子块级别执行以获得更精准的排序，
        #    所以这里绕过Hybrid的提升，直接获取原始子块结果
        candidate_k = min(k * 4, 20)

        # 元数据过滤：自动提取法条编号
        filter_dict = extract_article_ref(query)

        # 获取向量检索的原始子块结果（未提升）
        vector_mgr = self.hybrid_retriever.vector_retriever.vector_store_manager
        raw_vector_results = vector_mgr.similarity_search_with_score(
            query, k=candidate_k, score_threshold=0.0, filter_dict=filter_dict
        )
        # 获取BM25的原始子块结果（同样支持元数据过滤）
        raw_bm25_results = self.hybrid_retriever.bm25_retriever.retrieve(
            query, k=candidate_k, score_threshold=0.0, filter_dict=filter_dict
        )

        # RRF融合（在子块级别）
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        for rank, (doc, _score) in enumerate(raw_vector_results):
            key = doc.page_content
            doc_scores[key] = doc_scores.get(key, 0.0) + self.hybrid_retriever.vector_weight / (self.hybrid_retriever.rrf_k + rank + 1)
            doc_map[key] = doc
        for rank, (doc, _score) in enumerate(raw_bm25_results):
            key = doc.page_content
            doc_scores[key] = doc_scores.get(key, 0.0) + self.hybrid_retriever.bm25_weight / (self.hybrid_retriever.rrf_k + rank + 1)
            doc_map[key] = doc

        sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        candidates = []
        for content, rrf_score in sorted_items[:candidate_k]:
            doc = doc_map[content]
            normalized_score = min(rrf_score * self.hybrid_retriever.rrf_k, 1.0)
            candidates.append((doc, round(normalized_score, 4)))

        if not candidates:
            return []

        # 2. Cross-Encoder重排序（在子块级别，排序更精准）
        reranker = self._get_reranker()
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = reranker.predict(pairs)

        # 3. 按重排序分数排序
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 4. 取Top-K（仍在子块级别）
        results = []
        for (doc, _old_score), rerank_score in scored_results[:k]:
            # Cross-Encoder分数归一化到0-1
            normalized = 1.0 / (1.0 + max(0, -rerank_score))
            if normalized >= score_threshold:
                results.append((doc, round(normalized, 4)))

        # 5. 层次化父子分块：将重排序后的子块提升为完整父块
        return vector_mgr.promote_children_to_parents(results)

    def is_ready(self) -> bool:
        return self.hybrid_retriever.is_ready()
