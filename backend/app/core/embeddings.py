from langchain_core.embeddings import Embeddings
from app.config import settings
import os


def _setup_hf_mirror():
    """设置HuggingFace镜像源（国内加速下载模型）。"""
    if settings.HF_ENDPOINT and "mirror" in settings.HF_ENDPOINT:
        os.environ["HF_ENDPOINT"] = settings.HF_ENDPOINT


def get_embeddings() -> Embeddings:
    """根据配置返回Embedding模型实例。
    
    优先级：
    1. 如果配置了EMBEDDING_API_KEY和EMBEDDING_API_BASE，使用OpenAI兼容API
    2. 如果只配置了EMBEDDING_API_KEY，使用OpenAI官方API
    3. 默认使用HuggingFace本地模型（BGE-M3）

    注意：langchain_huggingface 顶层会连带导入 torch / transformers（约 60s），
    所以两个 Embedding 实现都改为惰性导入——只有真正实例化时才付出这个代价，
    让 import app.core.vectorstore 这类只做类型/常量引用的场景保持秒级。
    """
    _setup_hf_mirror()

    if settings.EMBEDDING_API_KEY and settings.EMBEDDING_API_BASE:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key=settings.EMBEDDING_API_KEY,
            openai_api_base=settings.EMBEDDING_API_BASE,
        )
    elif settings.EMBEDDING_API_KEY:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key=settings.EMBEDDING_API_KEY,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
