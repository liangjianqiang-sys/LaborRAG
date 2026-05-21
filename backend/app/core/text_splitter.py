import os
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


class LawArticleSplitter:
    """法条级结构化切分器，按法律条文（第X条）切分，保持法条完整性。

    支持层次化父子分块模式（PARENT_CHILD_ENABLED=True）：
    - 父块：整条法条完整内容，chunk_type="parent"
    - 子块：法条内每个自然段落，chunk_type="child"，通过 parent_id 映射回父块
    - parent_id 格式：{doc_id}_{article_number}，确保跨文档唯一
    """

    # 匹配 "第X条" 的正则
    ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千\d]+条\s")
    # 匹配法条内的款项标记，如 （一）、（二）、1.、2. 等
    CLAUSE_PATTERN = re.compile(r"^[（(][一二三四五六七八九十\d]+[）)]|^\d+[.、]")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """按法条结构切分文档。

        当 PARENT_CHILD_ENABLED=True 时，返回的列表同时包含父块和子块，
        通过 metadata 中的 chunk_type 字段区分。
        """
        all_chunks = []
        for doc in documents:
            source = doc.metadata.get("source", "未知")
            # 用 source 作为 doc_id，构建跨文档唯一的 parent_id
            doc_id = self._sanitize_doc_id(source)
            if settings.PARENT_CHILD_ENABLED:
                chunks = self._split_by_article_parent_child(doc.page_content, source, doc_id)
            else:
                chunks = self._split_by_article(doc.page_content, source)
            all_chunks.extend(chunks)
        return all_chunks

    @staticmethod
    def _sanitize_doc_id(source: str) -> str:
        """将 source 路径转换为安全的 doc_id，用于 parent_id 构建。"""
        # 取文件名（不含扩展名），去除特殊字符
        basename = os.path.basename(source)
        name_without_ext = os.path.splitext(basename)[0]
        # 只保留字母、数字、下划线、中文
        return re.sub(r"[^\w\u4e00-\u9fff]", "_", name_without_ext)

    def _split_by_article_parent_child(
        self, text: str, source: str, doc_id: str
    ) -> List[Document]:
        """层次化父子分块：每个法条生成一个父块 + 多个子块。"""
        lines = text.split("\n")
        articles: List[str] = []
        current_article: List[str] = []
        current_article_num = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = self.ARTICLE_PATTERN.match(stripped)
            if match:
                if current_article:
                    articles.append("\n".join(current_article))
                    current_article = []
                current_article_num = stripped[:stripped.index("条") + 1]
                current_article.append(stripped)
            else:
                current_article.append(stripped)

        if current_article:
            articles.append("\n".join(current_article))

        # 没有识别到法条结构，回退
        if not articles:
            return self._fallback_split(text, source)

        chunks: List[Document] = []

        for article_text in articles:
            # 提取法条编号
            article_num = ""
            first_line = article_text.split("\n")[0]
            num_match = self.ARTICLE_PATTERN.match(first_line)
            if num_match:
                article_num = first_line[:first_line.index("条") + 1]

            # 构建 parent_id：{doc_id}_{article_number}
            parent_id = f"{doc_id}_{article_num}" if article_num else f"{doc_id}_unknown"

            # --- 父块：完整法条，严禁截断 ---
            parent_doc = Document(
                page_content=article_text,
                metadata={
                    "source": source,
                    "article": article_num,
                    "chunk_type": "parent",
                    "parent_id": parent_id,
                },
            )
            chunks.append(parent_doc)

            # --- 子块：法条内每个自然段落 ---
            child_lines = [l.strip() for l in article_text.split("\n") if l.strip()]
            if len(child_lines) <= 1:
                # 法条只有一行，子块与父块内容相同
                child_doc = Document(
                    page_content=article_text,
                    metadata={
                        "source": source,
                        "article": article_num,
                        "chunk_type": "child",
                        "parent_id": parent_id,
                    },
                )
                chunks.append(child_doc)
            else:
                # 多行法条：尝试按款项/段落拆分为子块
                child_paragraphs = self._split_article_into_paragraphs(child_lines)
                for para_text in child_paragraphs:
                    child_doc = Document(
                        page_content=para_text,
                        metadata={
                            "source": source,
                            "article": article_num,
                            "chunk_type": "child",
                            "parent_id": parent_id,
                        },
                    )
                    chunks.append(child_doc)

        return chunks

    def _split_article_into_paragraphs(self, lines: List[str]) -> List[str]:
        """将法条行列表拆分为自然段落（子块）。

        拆分策略：
        1. 法条标题行（第X条 ...）单独作为第一个子块
        2. 以款项标记开头的行（如（一）、1.）单独作为子块
        3. 其他连续行合并为一个子块
        """
        paragraphs: List[str] = []
        current_para: List[str] = []

        for i, line in enumerate(lines):
            is_article_title = (i == 0 and self.ARTICLE_PATTERN.match(line))
            is_clause_start = self.CLAUSE_PATTERN.match(line)

            if is_article_title or is_clause_start:
                # 遇到新的段落起点，先保存当前段落
                if current_para:
                    paragraphs.append("\n".join(current_para))
                    current_para = []
                current_para.append(line)
            else:
                current_para.append(line)

        # 保存最后一个段落
        if current_para:
            paragraphs.append("\n".join(current_para))

        # 过滤空段落
        return [p for p in paragraphs if p.strip()]

    def _split_by_article(self, text: str, source: str) -> List[Document]:
        """原始法条切分逻辑（PARENT_CHILD_ENABLED=False 时使用）。"""
        lines = text.split("\n")
        articles: List[str] = []
        current_article: List[str] = []
        current_article_num = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = self.ARTICLE_PATTERN.match(stripped)
            if match:
                if current_article:
                    articles.append("\n".join(current_article))
                    current_article = []
                current_article_num = stripped[:stripped.index("条") + 1]
                current_article.append(stripped)
            else:
                current_article.append(stripped)

        if current_article:
            articles.append("\n".join(current_article))

        if not articles:
            return self._fallback_split(text, source)

        chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )

        for article_text in articles:
            article_num = ""
            first_line = article_text.split("\n")[0]
            num_match = self.ARTICLE_PATTERN.match(first_line)
            if num_match:
                article_num = first_line[:first_line.index("条") + 1]

            if len(article_text) <= settings.CHUNK_SIZE:
                chunks.append(Document(
                    page_content=article_text,
                    metadata={"source": source, "article": article_num},
                ))
            else:
                sub_chunks = splitter.split_text(article_text)
                for i, sub in enumerate(sub_chunks):
                    chunks.append(Document(
                        page_content=sub,
                        metadata={
                            "source": source,
                            "article": article_num,
                            "chunk_index": i,
                        },
                    ))

        return chunks

    def _fallback_split(self, text: str, source: str) -> List[Document]:
        """回退到固定长度切分。"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )
        docs = splitter.create_documents([text], metadatas=[{"source": source}])
        return docs
