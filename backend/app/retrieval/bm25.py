import os
import json
from typing import List, Tuple
import jieba
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from app.retrieval.base import BaseRetriever
from app.core.config import BACKEND_DIR, settings

def _law_dict_path() -> str:
    """法律自定义词典的绝对路径；缺失即抛错。

    路径基于 `BACKEND_DIR` 而非 `__file__`：词典在 `backend/data/` 下，与本文件
    不同目录。用 `__file__` 推算的写法在文件移动后会**静默失效** —— 词典加载不上，
    jieba 把「经济补偿金」切成「经济/补偿/金」，检索质量悄悄下降，且不会有任何报错。

    因此这里缺词典时**直接抛错**而不是 `if os.path.exists(...)` 包一层：
    该文件是版本控制内的资产，缺失只可能是路径算错了，静默降级比启动失败难排查得多。
    """
    path = BACKEND_DIR / "data" / "law_dict.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"法律自定义词典缺失：{path}\n"
            "该词典用于保证法律术语不被 jieba 切碎，缺失会静默降低检索质量，故直接失败。"
        )
    return str(path)


# ── 加载法律专属自定义词典（确保专业术语不被切碎） ──
_LAW_DICT_PATH = _law_dict_path()
jieba.load_userdict(_LAW_DICT_PATH)
print(f"[BM25] 已加载法律自定义词典: {_LAW_DICT_PATH}")


class BM25Retriever(BaseRetriever):
    """BM25关键词检索器，基于jieba中文分词+法律专属词典。"""

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
