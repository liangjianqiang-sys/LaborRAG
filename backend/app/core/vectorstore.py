import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.core.embeddings import get_embeddings
from app.core.text_splitter import LawArticleSplitter


class VectorStoreManager:
    """FAISS向量库管理器，负责文档切分、向量化、检索、保存和加载。"""

    def __init__(self):
        self.embeddings = get_embeddings()
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )
        self.law_article_splitter = LawArticleSplitter()

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """根据切分策略切分文档。"""
        if settings.CHUNK_STRATEGY == "law_article":
            return self.law_article_splitter.split_documents(documents)
        else:
            return self.text_splitter.split_documents(documents)

    def build_from_documents(self, documents: List[Document]) -> int:
        """从文档列表构建向量库，返回chunk数量。"""
        chunks = self._split_documents(documents)
        if not chunks:
            return 0

        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.save()
        return len(chunks)

    def add_documents(self, documents: List[Document]) -> int:
        """向已有向量库添加文档。"""
        chunks = self._split_documents(documents)
        if not chunks:
            return 0

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)
        self.save()
        return len(chunks)

    def similarity_search_with_score(
        self, query: str, k: int = None, score_threshold: float = None
    ) -> List[Tuple[Document, float]]:
        """带分数的相似度检索。"""
        if self.vector_store is None:
            return []

        k = k or settings.TOP_K
        score_threshold = score_threshold or settings.SCORE_THRESHOLD

        # FAISS返回的是L2距离，需要转换为相似度分数
        # 距离越小越相似，转换为0-1的分数：score = 1 / (1 + distance)
        raw_results = self.vector_store.similarity_search_with_score(
            query, k=k
        )
        results = []
        for doc, distance in raw_results:
            score = 1.0 / (1.0 + distance)
            if score >= score_threshold:
                results.append((doc, score))
        return results

    def save(self):
        """保存向量库到磁盘。"""
        if self.vector_store is not None:
            os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
            self.vector_store.save_local(settings.VECTOR_STORE_PATH)

    def load(self) -> bool:
        """从磁盘加载向量库，成功返回True。"""
        index_path = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
        if os.path.exists(index_path):
            self.vector_store = FAISS.load_local(
                settings.VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return True
        return False

    @property
    def is_ready(self) -> bool:
        """向量库是否可用。"""
        return self.vector_store is not None
