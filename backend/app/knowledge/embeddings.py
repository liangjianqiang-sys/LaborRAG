from langchain_core.embeddings import Embeddings
from app.core.config import settings


def get_embeddings() -> Embeddings:
    """根据配置返回Embedding模型实例。
    
    优先级：
    1. 如果配置了EMBEDDING_API_KEY和EMBEDDING_API_BASE，使用OpenAI兼容API
    2. 如果只配置了EMBEDDING_API_KEY，使用OpenAI官方API
    3. 默认使用HuggingFace本地模型（BGE-M3）

    注意：langchain_huggingface 顶层会连带导入 torch / transformers（约 60s），
    所以两个 Embedding 实现都改为惰性导入——只有真正实例化时才付出这个代价，
    让 import app.knowledge.store 这类只做类型/常量引用的场景保持秒级。

    另注：HF_HOME（模型缓存目录）与 HF_ENDPOINT（国内镜像）**不在这里设置**。
    它们必须在 `huggingface_hub` 被导入之前落定，而本模块被导入时早已过了那个时机。
    统一在 `app/core/config.py` 的模块体里设置，理由见该文件顶部注释。
    """
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
