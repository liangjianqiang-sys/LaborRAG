"""BGE-M3 Sparse Retriever — 用学习型稀疏表示替代传统 BM25。

BGE-M3 的 lexical_weights 是 token_id → weight 的稀疏字典，
比 BM25Okapi + jieba 分词更精准（模型学习到"经济补偿金"是整体概念）。

检索逻辑：query sparse · doc sparse 点积 → 排序取 Top-K
"""
import os
import json
import threading
from typing import List, Tuple, Dict

import numpy as np
from langchain_core.documents import Document
from app.core.retriever.base import BaseRetriever
from app.config import settings


# ── 模块级 BGE-M3 单例（避免并发重复加载）──
_bge3_instance = None
_bge3_lock = threading.Lock()


def _get_bge3_singleton():
    """获取全局 BGE-M3 模型单例（线程安全）。"""
    global _bge3_instance
    if _bge3_instance is None:
        with _bge3_lock:
            if _bge3_instance is None:
                from FlagEmbedding import BGEM3FlagModel
                print(f"Loading BGE-M3 model for sparse retrieval...")
                _bge3_instance = BGEM3FlagModel(
                    settings.EMBEDDING_MODEL_NAME,
                    use_fp16=True,
                    device="cpu",
                )
                print("BGE-M3 model loaded (dense + sparse).")
    return _bge3_instance


class SparseRetriever(BaseRetriever):
    """BGE-M3 Sparse Retriever：学习型稀疏表示替代 BM25。

    索引格式：每个文档存储其 sparse 向量（token_id → weight），
    检索时计算 query sparse 与 doc sparse 的点积。
    """

    def __init__(self):
        self.documents: List[Document] = []
        self.sparse_weights: List[Dict[str, float]] = []  # 每个文档的 sparse 向量
        self._index_path = os.path.join(settings.VECTOR_STORE_PATH, "sparse_index.json")

    def build_index(self, documents: List[Document], batch_size: int = 32) -> int:
        """构建 sparse 索引，返回文档数量。

        使用 BGE-M3 对所有文档生成 sparse 向量（lexical_weights）。
        """
        model = _get_bge3_singleton()

        self.documents = documents
        texts = [doc.page_content for doc in documents]

        # 批量编码
        print(f"Building sparse index for {len(texts)} documents (batch={batch_size})...")
        all_sparse = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            output = model.encode(batch, return_sparse=True)
            all_sparse.extend(output["lexical_weights"])
            print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")

        # 转换为可序列化格式
        self.sparse_weights = []
        for sw in all_sparse:
            # {token_id_str: float_value}
            self.sparse_weights.append({k: float(v) for k, v in sw.items()})

        self._save_index()
        print(f"Sparse index built: {len(self.documents)} documents")
        return len(documents)

    def load_index(self) -> bool:
        """从磁盘加载 sparse 索引。"""
        if not os.path.exists(self._index_path):
            return False
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in data["documents"]
            ]
            self.sparse_weights = data["sparse_weights"]
            print(f"Sparse index loaded: {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Failed to load sparse index: {e}")
            return False

    def _save_index(self):
        """保存 sparse 索引到磁盘。"""
        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        data = {
            "documents": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in self.documents
            ],
            "sparse_weights": self.sparse_weights,
        }
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def retrieve(
        self, query: str, k: int = 5, score_threshold: float = 0.0,
        filter_dict: dict = None
    ) -> List[Tuple[Document, float]]:
        """Sparse 检索：计算 query·doc 点积，返回 Top-K。

        Args:
            query: 查询文本
            k: 返回数量
            score_threshold: 分数阈值
            filter_dict: 元数据过滤条件
        """
        if not self.is_ready():
            return []

        model = _get_bge3_singleton()

        # 编码 query 的 sparse 向量
        query_output = model.encode([query], return_sparse=True)
        query_sparse = query_output["lexical_weights"][0]

        # 计算所有文档的点积
        scores = np.array([
            self._sparse_dot(query_sparse, doc_sparse)
            for doc_sparse in self.sparse_weights
        ])

        # 取 Top-K（扩大候选集以补偿过滤后的损失）
        search_k = k * 3 if filter_dict else k
        top_indices = np.argsort(scores)[::-1][:search_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < score_threshold:
                continue
            doc = self.documents[int(idx)]
            if filter_dict:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_dict.items()
                )
                if not match:
                    continue
            results.append((doc, round(score, 4)))
            if len(results) >= k:
                break
        return results

    @staticmethod
    def _sparse_dot(query_sparse: dict, doc_sparse: dict) -> float:
        """计算两个稀疏向量的点积（只遍历交集 key，O(|交集|)）。"""
        score = 0.0
        for key, q_weight in query_sparse.items():
            if key in doc_sparse:
                score += q_weight * doc_sparse[key]
        return score

    def is_ready(self) -> bool:
        return len(self.documents) > 0 and len(self.sparse_weights) > 0
