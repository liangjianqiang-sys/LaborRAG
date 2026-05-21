import os
import json
import re
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings
from app.core.embeddings import get_embeddings
from app.core.text_splitter import LawArticleSplitter


# 匹配法条编号的正则，如 "第四十四条"、"第31条"
_ARTICLE_REF_PATTERN = re.compile(r"第[一二三四五六七八九十百千\d]+条")


def extract_article_ref(query: str) -> Optional[dict]:
    """从查询中提取法条编号，返回元数据过滤条件。

    示例：
        "第四十四条怎么规定的" → {"article": "第四十四条"}
        "第31条的内容是什么" → {"article": "第31条"}
        "加班费怎么算" → None（无法条编号）
    """
    match = _ARTICLE_REF_PATTERN.search(query)
    if match:
        return {"article": match.group()}
    return None


class VectorStoreManager:
    """FAISS向量库管理器，负责文档切分、向量化、检索、保存和加载。

    支持层次化父子分块（PARENT_CHILD_ENABLED=True）：
    - 向量库仅索引子块（chunk_type="child"），用于精准语义检索
    - 父块存储在 self.parent_store 字典中，不参与向量化
    - 检索后通过 promote_children_to_parents() 将子块替换为完整父块
    """

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
        # 父块存储：parent_id -> Document
        self.parent_store: dict[str, Document] = {}
        self._parent_store_path = os.path.join(
            settings.VECTOR_STORE_PATH, "parent_store.json"
        )

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """根据切分策略切分文档。

        当 PARENT_CHILD_ENABLED=True 且 CHUNK_STRATEGY="law_article" 时：
        - 仅将 chunk_type="child" 的子块返回用于向量化
        - 将 chunk_type="parent" 的父块存入 self.parent_store
        """
        if settings.CHUNK_STRATEGY == "law_article":
            all_chunks = self.law_article_splitter.split_documents(documents)
        else:
            return self.text_splitter.split_documents(documents)

        # 父子分块模式：分离父块和子块
        if settings.PARENT_CHILD_ENABLED:
            child_chunks = []
            for chunk in all_chunks:
                chunk_type = chunk.metadata.get("chunk_type", "")
                if chunk_type == "parent":
                    parent_id = chunk.metadata.get("parent_id", "")
                    if parent_id:
                        self.parent_store[parent_id] = chunk
                elif chunk_type == "child":
                    child_chunks.append(chunk)
                else:
                    # 无 chunk_type 标记的文档（如 fallback），直接作为检索块
                    child_chunks.append(chunk)
            return child_chunks

        return all_chunks

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
        self, query: str, k: int = None, score_threshold: float = None,
        filter_dict: dict = None
    ) -> List[Tuple[Document, float]]:
        """带分数的相似度检索，支持元数据过滤。

        Args:
            query: 查询文本
            k: 返回数量，None时使用配置默认值
            score_threshold: 分数阈值，None时使用配置默认值
            filter_dict: 元数据过滤条件，如 {"article": "第四十四条"}
        """
        if self.vector_store is None:
            return []

        if k is None:
            k = settings.TOP_K
        if score_threshold is None:
            score_threshold = settings.SCORE_THRESHOLD

        # FAISS返回的是L2距离，需要转换为相似度分数
        # 距离越小越相似，转换为0-1的分数：score = 1 / (1 + distance)
        raw_results = self.vector_store.similarity_search_with_score(
            query, k=k, filter=filter_dict
        )
        results = []
        for doc, distance in raw_results:
            score = 1.0 / (1.0 + distance)
            if score >= score_threshold:
                results.append((doc, score))
        return results

    def promote_children_to_parents(
        self, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """将检索结果中的子块替换为对应的完整父块，并去重重排。

        逻辑：
        1. 识别子块的 parent_id，从 parent_store 获取完整父块
        2. 多个子块命中同一父块时，取最高分数作为父块最终得分
        3. 按最终得分全局降序排列
        4. 非 child 类型的文档（如 fallback 块）直接保留

        当 PARENT_CHILD_ENABLED=False 或 parent_store 为空时，直接返回原结果。
        """
        if not settings.PARENT_CHILD_ENABLED or not self.parent_store:
            return results

        # 按 parent_id 聚合，保留每个父块的最高分数
        parent_best: dict[str, float] = {}  # parent_id -> max_score
        other_results: List[Tuple[Document, float]] = []

        for doc, score in results:
            chunk_type = doc.metadata.get("chunk_type", "")
            if chunk_type == "child":
                parent_id = doc.metadata.get("parent_id", "")
                if parent_id and parent_id in self.parent_store:
                    # 记录该父块的最高分数
                    if parent_id not in parent_best or score > parent_best[parent_id]:
                        parent_best[parent_id] = score
                else:
                    # 子块找不到父块，保留原样
                    other_results.append((doc, score))
            else:
                # 非子块文档直接保留
                other_results.append((doc, score))

        # 构建父块结果列表
        parent_results: List[Tuple[Document, float]] = []
        for parent_id, max_score in parent_best.items():
            parent_doc = self.parent_store[parent_id]
            parent_results.append((parent_doc, round(max_score, 4)))

        # 合并父块结果和其他结果，按分数降序排列
        all_results = parent_results + other_results
        all_results.sort(key=lambda x: x[1], reverse=True)

        return all_results

    def save(self):
        """保存向量库和父块存储到磁盘。"""
        if self.vector_store is not None:
            os.makedirs(settings.VECTOR_STORE_PATH, exist_ok=True)
            self.vector_store.save_local(settings.VECTOR_STORE_PATH)

        # 持久化父块存储
        if self.parent_store:
            self._save_parent_store()

    def _save_parent_store(self):
        """将父块存储保存为 JSON 文件。"""
        os.makedirs(os.path.dirname(self._parent_store_path), exist_ok=True)
        data = {}
        for parent_id, doc in self.parent_store.items():
            data[parent_id] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        with open(self._parent_store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_parent_store(self) -> bool:
        """从磁盘加载父块存储，成功返回True。"""
        if not os.path.exists(self._parent_store_path):
            return False
        try:
            with open(self._parent_store_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.parent_store = {}
            for parent_id, item in data.items():
                self.parent_store[parent_id] = Document(
                    page_content=item["page_content"],
                    metadata=item["metadata"],
                )
            return True
        except Exception as e:
            print(f"Failed to load parent store: {e}")
            return False

    def load(self) -> bool:
        """从磁盘加载向量库和父块存储，成功返回True。"""
        index_path = os.path.join(settings.VECTOR_STORE_PATH, "index.faiss")
        if os.path.exists(index_path):
            self.vector_store = FAISS.load_local(
                settings.VECTOR_STORE_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            # 同时加载父块存储
            if settings.PARENT_CHILD_ENABLED:
                self._load_parent_store()
            return True
        return False

    @property
    def is_ready(self) -> bool:
        """向量库是否可用。"""
        return self.vector_store is not None
