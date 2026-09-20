from abc import ABC, abstractmethod
from typing import List, Tuple
from langchain_core.documents import Document


class BaseRetriever(ABC):
    """检索器抽象基类，V2/V3只需新增实现类即可扩展。"""

    @abstractmethod
    def retrieve(
        self, query: str, k: int = 5, score_threshold: float = 0.2
    ) -> List[Tuple[Document, float]]:
        """检索与query相关的文档，返回(文档, 分数)列表。"""
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """检索器是否可用。"""
        pass
