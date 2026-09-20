from typing import List, Tuple
from langchain_core.documents import Document
from app.retrieval.base import BaseRetriever
from app.knowledge.store import VectorStoreManager, extract_article_ref


class VectorRetriever(BaseRetriever):
    """V1: 纯向量语义检索，支持层次化父子分块提升和元数据过滤。"""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

    def retrieve(
        self, query: str, k: int = 5, score_threshold: float = 0.2
    ) -> List[Tuple[Document, float]]:
        if not self.is_ready():
            return []
        # 元数据过滤：自动提取法条编号
        filter_dict = extract_article_ref(query)
        results = self.vector_store_manager.similarity_search_with_score(
            query, k=k, score_threshold=score_threshold, filter_dict=filter_dict
        )
        # 层次化父子分块：将子块替换为完整父块
        return self.vector_store_manager.promote_children_to_parents(results)

    def is_ready(self) -> bool:
        return self.vector_store_manager.is_ready
