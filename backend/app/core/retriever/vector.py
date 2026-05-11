from typing import List, Tuple
from langchain_core.documents import Document
from app.core.retriever.base import BaseRetriever
from app.core.vectorstore import VectorStoreManager


class VectorRetriever(BaseRetriever):
    """V1: 纯向量语义检索。"""

    def __init__(self, vector_store_manager: VectorStoreManager):
        self.vector_store_manager = vector_store_manager

    def retrieve(
        self, query: str, k: int = 5, score_threshold: float = 0.3
    ) -> List[Tuple[Document, float]]:
        if not self.is_ready():
            return []
        return self.vector_store_manager.similarity_search_with_score(
            query, k=k, score_threshold=score_threshold
        )

    def is_ready(self) -> bool:
        return self.vector_store_manager.is_ready
