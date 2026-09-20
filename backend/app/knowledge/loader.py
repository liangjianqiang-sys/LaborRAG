import os
from typing import List

from langchain_core.documents import Document


class DocumentLoader:
    """文档加载器，支持PDF/DOCX/TXT/MD格式。"""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    @staticmethod
    def _resolve_loader(ext: str):
        """按扩展名取 langchain 的 loader 类。

        惰性导入：`langchain_community.document_loaders` 顶层导入实测 30s+，
        但它只在真正加载文档（建知识库）时才需要，问答链路完全用不到。
        放在模块级会让每个 `import app.knowledge.loader` 的调用方都付出这个代价。
        """
        from langchain_community.document_loaders import (
            Docx2txtLoader,
            TextLoader,
            UnstructuredMarkdownLoader,
        )

        return {
            ".docx": Docx2txtLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
        }[ext]

    def load_documents(self) -> List[Document]:
        """加载data_dir下所有支持的文档。"""
        documents = []
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            return documents

        for filename in os.listdir(self.data_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in self.SUPPORTED_EXTENSIONS:
                continue

            filepath = os.path.join(self.data_dir, filename)
            try:
                docs = self._load_file(filepath)
                documents.extend(docs)
                print(f"Loaded {len(docs)} chunks from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        return documents

    def load_single_file(self, filepath: str) -> List[Document]:
        """加载单个文件。"""
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {ext}")

        return self._load_file(filepath)

    def _load_file(self, filepath: str) -> List[Document]:
        """内部加载方法。"""
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)

        if ext == ".pdf":
            # PyMuPDF4LLM: PDF → Markdown，结构更完整
            # 惰性导入：pymupdf4llm 顶层导入约 50s，而只有 PDF 分支需要它，
            # 放在模块级会让所有 import app.knowledge.loader 的代码都付出这个代价
            import pymupdf4llm

            md_text = pymupdf4llm.to_markdown(filepath)
            docs = [Document(page_content=md_text, metadata={"source": filename})]
        elif ext == ".txt":
            loader = self._resolve_loader(ext)(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
        else:
            loader = self._resolve_loader(ext)(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename

        return docs
