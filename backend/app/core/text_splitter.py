import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


class LawArticleSplitter:
    """法条级结构化切分器，按法律条文（第X条）切分，保持法条完整性。"""

    # 匹配 "第X条" 的正则
    ARTICLE_PATTERN = re.compile(r"^第[一二三四五六七八九十百千\d]+条\s")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """按法条结构切分文档。"""
        all_chunks = []
        for doc in documents:
            source = doc.metadata.get("source", "未知")
            chunks = self._split_by_article(doc.page_content, source)
            all_chunks.extend(chunks)
        return all_chunks

    def _split_by_article(self, text: str, source: str) -> List[Document]:
        """将文本按法条切分。"""
        lines = text.split("\n")
        articles: List[str] = []
        current_article: List[str] = []
        current_article_num = ""

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 检测是否是法条开头
            match = self.ARTICLE_PATTERN.match(stripped)
            if match:
                # 保存上一条
                if current_article:
                    articles.append("\n".join(current_article))
                    current_article = []
                # 提取法条编号
                current_article_num = stripped[:stripped.index("条") + 1]
                current_article.append(stripped)
            else:
                current_article.append(stripped)

        # 保存最后一条
        if current_article:
            articles.append("\n".join(current_article))

        # 如果没有识别到法条结构，回退到固定长度切分
        if not articles:
            return self._fallback_split(text, source)

        # 将过长的法条进一步切分（保持法条编号元数据）
        chunks = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
            length_function=len,
        )

        for article_text in articles:
            # 提取法条编号
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
                # 过长法条进一步切分，每段都保留法条编号
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
