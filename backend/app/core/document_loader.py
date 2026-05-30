import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
import pymupdf4llm


class DocumentLoader:
    """文档加载器，支持PDF/DOCX/TXT/MD格式。"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": "pymupdf4llm",
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

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
        loader_cls = self.SUPPORTED_EXTENSIONS[ext]
        filename = os.path.basename(filepath)

        if ext == ".pdf":
            # PyMuPDF4LLM: PDF → Markdown，结构更完整
            md_text = pymupdf4llm.to_markdown(filepath)
            docs = [Document(page_content=md_text, metadata={"source": filename})]
        elif ext == ".txt":
            loader = loader_cls(filepath, encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename
        else:
            loader = loader_cls(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = filename

        return docs
