from abc import ABC, abstractmethod
from typing import List, Tuple
from langchain_core.documents import Document


class BaseGenerator(ABC):
    """生成器抽象基类，V3只需新增CRAG实现类即可扩展。"""

    @abstractmethod
    def generate(
        self,
        question: str,
        context_docs: List[Tuple[Document, float]],
    ) -> str:
        """基于检索到的文档生成回答。"""
        pass
