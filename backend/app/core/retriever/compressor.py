"""上下文压缩器：基于Embedding相似度过滤低相关文档。

对应总结文档3.6节 EmbeddingsFilter——低成本快速过滤。
核心思想：检索结果中混入无关文档，直接喂给LLM浪费token，
用Embedding余弦相似度快速过滤，零LLM调用成本。
"""
from typing import List, Tuple

import numpy as np
from langchain_core.documents import Document

from app.config import settings
from app.core.embeddings import get_embeddings


class EmbeddingsCompressor:
    """基于Embedding相似度的上下文压缩器。"""

    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or settings.COMPRESSION_THRESHOLD
        self._embeddings = None

    @property
    def embeddings(self):
        """延迟加载Embedding模型（避免启动时加载）。"""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings

    def compress(
        self,
        query: str,
        docs: List[Tuple[Document, float]],
        similarity_threshold: float = None,
    ) -> List[Tuple[Document, float]]:
        """过滤与查询相似度低于阈值的文档。

        Args:
            query: 用户查询文本
            docs: 检索结果列表，每项为(Document, score)
            similarity_threshold: 相似度阈值，None则用默认值

        Returns:
            过滤后的文档列表
        """
        if not docs:
            return docs

        threshold = similarity_threshold or self.similarity_threshold

        # 编码query和所有文档
        query_embedding = self.embeddings.embed_query(query)
        doc_texts = [doc.page_content for doc, _ in docs]
        doc_embeddings = self.embeddings.embed_documents(doc_texts)

        # 计算余弦相似度并过滤
        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)

        filtered = []
        for i, (doc, retrieval_score) in enumerate(docs):
            doc_vec = np.array(doc_embeddings[i])
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0 or query_norm == 0:
                continue
            similarity = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)

            if similarity >= threshold:
                filtered.append((doc, retrieval_score))
            else:
                print(f"[Compressor] ✂️ 过滤低相关文档: "
                      f"similarity={similarity:.3f} < {threshold}, "
                      f"内容='{doc.page_content[:40]}...'")

        print(f"[Compressor] ✂️ 压缩: {len(docs)} → {len(filtered)} 个文档 "
              f"(阈值={threshold})")
        return filtered
