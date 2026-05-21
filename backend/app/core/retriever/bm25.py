import os
import json
from typing import List, Tuple
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from app.core.retriever.base import BaseRetriever
from app.config import settings


class BM25Retriever(BaseRetriever):
    """BM25关键词检索器，基于jieba中文分词。"""

    def __init__(self):
        self.documents: List[Document] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: List[List[str]] = []
        self._index_path = os.path.join(settings.VECTOR_STORE_PATH, "bm25_index.json")

    def build_index(self, documents: List[Document]) -> int:
        """构建BM25索引，返回文档数量。"""
        self.documents = documents
        self.tokenized_corpus = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._save_index()
        return len(documents)

    def load_index(self) -> bool:
        """从磁盘加载BM25索引。"""
        if not os.path.exists(self._index_path):
            return False
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in data["documents"]
            ]
            self.tokenized_corpus = data["tokenized_corpus"]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            return True
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
            return False

    def _save_index(self):
        """保存BM25索引到磁盘。"""
        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        data = {
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self.documents
            ],
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def retrieve(
        self, query: str, k: int = 5, score_threshold: float = 0.0,
        filter_dict: dict = None
    ) -> List[Tuple[Document, float]]:
        """BM25检索，返回(文档, 分数)列表，支持元数据过滤。

        Args:
            query: 查询文本
            k: 返回数量
            score_threshold: 分数阈值
            filter_dict: 元数据过滤条件，如 {"article": "第四十四条"}
        """
        if not self.is_ready():
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        # 获取Top-K结果（扩大候选集以补偿过滤后的损失）
        search_k = k * 3 if filter_dict else k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:search_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < score_threshold:
                continue
            doc = self.documents[idx]
            # 元数据过滤：检查文档的metadata是否匹配过滤条件
            if filter_dict:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_dict.items()
                )
                if not match:
                    continue
            results.append((doc, score))
            if len(results) >= k:
                break
        return results

    def is_ready(self) -> bool:
        return self.bm25 is not None and len(self.documents) > 0
