from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
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
    """
    _setup_hf_mirror()

    if settings.EMBEDDING_API_KEY and settings.EMBEDDING_API_BASE:
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key=settings.EMBEDDING_API_KEY,
            openai_api_base=settings.EMBEDDING_API_BASE,
        )
    elif settings.EMBEDDING_API_KEY:
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            openai_api_key=settings.EMBEDDING_API_KEY,
        )
    else:
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
