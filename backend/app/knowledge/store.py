import os
import json
import re
from typing import List, Tuple, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from app.core.config import settings
from app.knowledge.embeddings import get_embeddings
from app.utils.files import file_lock
from app.utils.law_refs import article_to_cn
from app.knowledge.splitter import LawArticleSplitter
# 法条编号正则的唯一真相源（如 "第四十四条"、"第31条"）
from app.utils.law_refs import ARTICLE_REF_RE as _ARTICLE_REF_PATTERN


# 法律名称 → 文件名映射
# 注意：名称越长越精确的放在前面，避免短名错误匹配长名
_LAW_NAME_MAP = {
    # ── 主干法律（.txt） ──
    "劳动合同法": "劳动合同法.txt",
    "劳动法": "劳动法.txt",
    "劳动争议调解仲裁法": "劳动争议调解仲裁法.txt",
    "工伤保险条例": "工伤保险条例.pdf",
    "社会保险法": "社会保险法.txt",
    "职工带薪年休假条例": "职工带薪年休假条例.txt",
    # ── 新增配套法规（.pdf） ──
    "中华人民共和国劳动合同法实施条例": "中华人民共和国劳动合同法实施条例.pdf",
    "劳动合同法实施条例": "中华人民共和国劳动合同法实施条例.pdf",
    "中华人民共和国职业病防治法": "中华人民共和国职业病防治法.pdf",
    "职业病防治法": "中华人民共和国职业病防治法.pdf",
    "住房公积金管理条例": "住房公积金管理条例.pdf",
    "失业保险条例": "失业保险条例.pdf",
    "女职工劳动保护特别规定": "女职工劳动保护特别规定.pdf",
    "工资支付暂行规定": "工资支付暂行规定.pdf",
    "最低工资规定": "最低工资规定.pdf",
    "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）": "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）.pdf",
    "最高人民法院关于审理劳动争议案件适用法律问题的解释二": "最高人民法院关于审理劳动争议案件适用法律问题的解释（二）.pdf",
    "最高法劳动争议司法解释一": "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）.pdf",
    "最高法劳动争议司法解释二": "最高人民法院关于审理劳动争议案件适用法律问题的解释（二）.pdf",
    "劳动争议司法解释一": "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）.pdf",
    "劳动争议司法解释二": "最高人民法院关于审理劳动争议案件适用法律问题的解释（二）.pdf",
    "职工非因工伤残鉴定标准": "职工非因工伤残或因病丧失劳动能力程度鉴定标准(试行).pdf",
    "法条适用前提速查表": "法条适用前提速查表.txt",
}

# 匹配法律名称的正则（按名称长度降序排列，长名优先匹配避免歧义）
_LAW_NAME_PATTERN = re.compile(
    "|".join(sorted(_LAW_NAME_MAP.keys(), key=len, reverse=True))
)

# 阿拉伯数字 → 中文数字映射
def _normalize_article_num(article: str) -> str:
    """将法条编号中的阿拉伯数字转为中文数字，与元数据格式统一。

    "第47条" → "第四十七条"，"第四十七条" → 不变。

    转换本身走 app.utils.law_refs（唯一真相源）；本函数只负责
    「仅当条号是阿拉伯写法时才转换」这一层判断。
    """
    match = re.search(r"第(\d+)条", article)
    if not match:
        return article
    return article_to_cn(match.group(1))


def extract_article_ref(query: str) -> Optional[dict]:
    """从查询中提取法条编号和法律名称，返回元数据过滤条件。

    自动将阿拉伯数字转为中文数字，与向量库元数据格式统一。
    同时提取法律名称用于source过滤，避免不同法律同一条号混淆。

    示例：
        "劳动合同法第47条" → {"article": "第四十七条", "source": "劳动合同法.txt"}
        "第四十四条怎么规定的" → {"article": "第四十四条"}
        "加班费怎么算" → None（无法条编号）
        "劳动法相关问题" → {"source": "劳动法.txt"}（仅法律名过滤）
    """
    filter_dict = {}

    # 提取法律名称，即使无法条编号也有过滤价值
    law_match = _LAW_NAME_PATTERN.search(query)
    if law_match:
        law_name = law_match.group()
        filter_dict["source"] = _LAW_NAME_MAP[law_name]

    # 提取法条编号
    match = _ARTICLE_REF_PATTERN.search(query)
    if match:
        article = _normalize_article_num(match.group())
        filter_dict["article"] = article

    return filter_dict if filter_dict else None


class VectorStoreManager:
    """FAISS向量库管理器，负责文档切分、向量化、检索、保存和加载。

    支持层次化父子分块（PARENT_CHILD_ENABLED=True）：
    - 向量库仅索引子块（chunk_type="child"），用于精准语义检索
    - 父块存储在 self.parent_store 字典中，不参与向量化
    - 检索后通过 promote_children_to_parents() 将子块替换为完整父块
    """

    def __init__(self):
        # 惰性导入：langchain_text_splitters 顶层导入约 40s，只有实例化时才需要
        from langchain_text_splitters import RecursiveCharacterTextSplitter

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

        当filter指定了article时，优先从parent_store精确查找，
        避免语义检索对法条编号匹配不准的问题。

        Args:
            query: 查询文本
            k: 返回数量，None时使用配置默认值
            score_threshold: 分数阈值，None时使用配置默认值
            filter_dict: 元数据过滤条件，如 {"article": "第四十七条", "source": "劳动合同法.txt"}
        """
        if self.vector_store is None:
            return []

        if k is None:
            k = settings.TOP_K
        if score_threshold is None:
            score_threshold = settings.SCORE_THRESHOLD

        # 当filter指定了article，优先从parent_store精确查找
        if filter_dict and "article" in filter_dict and self.parent_store:
            exact_results = self._exact_lookup(filter_dict, k)
            if exact_results:
                return exact_results

        # 语义检索 + Python层过滤
        fetch_k = k * 5 if filter_dict else k
        raw_results = self.vector_store.similarity_search_with_score(
            query, k=fetch_k
        )
        results = []
        for doc, distance in raw_results:
            if filter_dict:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter_dict.items()
                )
                if not match:
                    continue
            score = 1.0 / (1.0 + distance)
            if score >= score_threshold:
                results.append((doc, score))
            if len(results) >= k:
                break
        return results

    def _exact_lookup(self, filter_dict: dict, k: int) -> List[Tuple[Document, float]]:
        """从parent_store精确查找匹配的父块，未找到返回空列表。"""
        return [
            (doc, 1.0)
            for doc in self.parent_store.values()
            if all(doc.metadata.get(key) == value for key, value in filter_dict.items())
        ][:k]

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
            if doc.metadata.get("chunk_type") == "child":
                parent_id = doc.metadata.get("parent_id", "")
                if parent_id and parent_id in self.parent_store:
                    if parent_id not in parent_best or score > parent_best[parent_id]:
                        parent_best[parent_id] = score
                else:
                    other_results.append((doc, score))
            else:
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
        """将父块存储保存为 JSON 文件（线程安全）。"""
        os.makedirs(os.path.dirname(self._parent_store_path), exist_ok=True)
        data = {}
        for parent_id, doc in self.parent_store.items():
            data[parent_id] = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
        with file_lock:
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
