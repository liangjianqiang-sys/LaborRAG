import threading
from typing import List, Tuple
from langchain_core.documents import Document
from app.core.retriever.base import BaseRetriever
from app.core.retriever.hybrid import HybridRetriever
from app.core.vectorstore import extract_article_ref
from app.core.retriever.query_enhance import enhance_query
from app.config import settings

# ── 模块级Reranker单例（避免并发时重复加载560MB模型）──
_reranker_instance = None
_reranker_lock = threading.Lock()


def _get_reranker_singleton():
    """获取全局Reranker单例（线程安全，双重检查锁）。"""
    global _reranker_instance
    if _reranker_instance is None:
        with _reranker_lock:
            if _reranker_instance is None:
                import os
                from sentence_transformers import CrossEncoder
                if settings.HF_ENDPOINT and "mirror" in settings.HF_ENDPOINT:
                    os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT
                print(f"Loading reranker model: {settings.RERANKER_MODEL_NAME}...")
                _reranker_instance = CrossEncoder(settings.RERANKER_MODEL_NAME, device="cpu")
                print("Reranker model loaded.")
    return _reranker_instance


class RerankedRetriever(BaseRetriever):
    """V2: 带重排序的混合检索器。先Hybrid检索候选集，再用Cross-Encoder精排。

    支持层次化父子分块：重排序在子块级别执行（精准排序），
    排序后再调用 promote_children_to_parents 将子块提升为完整父块。
    """

    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid_retriever = hybrid_retriever

    def _get_reranker(self):
        """获取Reranker模型（模块级单例，线程安全）。"""
        return _get_reranker_singleton()

    def retrieve(
        self, query: str, k: int = 5, score_threshold: float = 0.2,
        rerank_top_k: int = None,
    ) -> List[Tuple[Document, float]]:
        """混合检索 + 重排序 + 父子分块提升。

        Args:
            rerank_top_k: 重排序后返回数量，覆盖settings.RERANK_TOP_K。
                          简单查询传3，复杂查询传5，None则用配置默认值。
        """
        if not self.is_ready():
            return []

        # 0. 查询增强：口语化关键词→法言法语+法条编号
        enhanced_query = enhance_query(query)

        # 1. 先用Hybrid检索获取候选集（扩大到Top50）
        #    注意：HybridRetriever内部已做子块→父块提升，
        #    但重排序需要在子块级别执行以获得更精准的排序，
        #    所以这里绕过Hybrid的提升，直接获取原始子块结果
        candidate_k = min(k * 5, 50)

        # 元数据过滤：用增强查询提取法条编号和法律名（query_enhance注入的法条编号也会被捕获）
        filter_dict = extract_article_ref(enhanced_query)

        # 获取向量检索的原始子块结果（用增强查询提高命中率）
        vector_mgr = self.hybrid_retriever.vector_retriever.vector_store_manager
        raw_vector_results = vector_mgr.similarity_search_with_score(
            enhanced_query, k=candidate_k, score_threshold=0.0, filter_dict=filter_dict
        )
        # 获取BM25的原始子块结果（同样用增强查询）
        raw_bm25_results = self.hybrid_retriever.bm25_retriever.retrieve(
            enhanced_query, k=candidate_k, score_threshold=0.0, filter_dict=filter_dict
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

        # 2. Cross-Encoder重排序（用原始query，CrossEncoder理解自然语言更好）
        reranker = self._get_reranker()
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = reranker.predict(pairs)

        # 3. 按重排序分数排序
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # 4. 取Top-K（仍在子块级别，受RERANK_TOP_K上限约束）
        #    CrossEncoder原始分数含义：>0 相关，<0 不相关
        #    直接用原始分数阈值过滤，比归一化后用SCORE_THRESHOLD更准确
        effective_k = min(k, rerank_top_k if rerank_top_k is not None else settings.RERANK_TOP_K)
        rerank_threshold = settings.RERANK_SCORE_THRESHOLD
        results = []
        for (doc, _old_score), rerank_score in scored_results[:effective_k]:
            if rerank_score < rerank_threshold:
                continue
            # 归一化到0-1用于展示（sigmoid映射，仅用于返回分数，不影响过滤）
            normalized = 1.0 / (1.0 + max(0, -rerank_score))
            results.append((doc, round(normalized, 4)))

        # 5. 层次化父子分块：将重排序后的子块提升为完整父块
        return vector_mgr.promote_children_to_parents(results)

    def is_ready(self) -> bool:
        return self.hybrid_retriever.is_ready()
