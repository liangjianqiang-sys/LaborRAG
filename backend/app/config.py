import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录（backend/的父目录）
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_BASE_DIR = BASE_DIR / "data"


class Settings:
    PROJECT_NAME: str = "LaborRAG - 劳动法智能问答系统"
    VERSION: str = "4.0.0"
    API_PREFIX: str = "/api/v1"

    # LLM配置（生成用）
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    # LLM配置（评估用，可与生成用不同模型）
    EVAL_LLM_API_KEY: str = os.getenv("EVAL_LLM_API_KEY", "") or os.getenv("LLM_API_KEY", "")
    EVAL_LLM_API_BASE: str = os.getenv("EVAL_LLM_API_BASE", "") or os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    EVAL_LLM_MODEL_NAME: str = os.getenv("EVAL_LLM_MODEL_NAME", "") or os.getenv("LLM_MODEL_NAME", "deepseek-chat")

    # Embedding配置
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-m3"
    )
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_API_BASE: str = os.getenv("EMBEDDING_API_BASE", "")

    # 向量库配置（绝对路径，不受工作目录影响）
    # 注意: FAISS底层C++不支持中文路径，向量库存放到D盘纯英文路径
    VECTOR_STORE_PATH: str = os.getenv(
        "VECTOR_STORE_PATH",
        "D:/LaborRAG_data/vector_store"
    )
    DATA_DIR: str = os.getenv("DATA_DIR", str(DATA_BASE_DIR / "labor_laws"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "law_article")  # recursive | law_article

    # 层次化父子分块配置
    PARENT_CHILD_ENABLED: bool = os.getenv("PARENT_CHILD_ENABLED", "true").lower() == "true"

    # 检索配置
    RETRIEVER_TYPE: str = os.getenv("RETRIEVER_TYPE", "vector")  # vector | hybrid | reranked
    TOP_K: int = int(os.getenv("TOP_K", "8"))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.2"))

    # 混合检索配置 (V2)
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.5"))  # 向量检索权重
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.5"))      # BM25检索权重
    RRF_K: int = int(os.getenv("RRF_K", "60"))                       # RRF常数

    # 重排序配置 (V2)
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))          # 重排序后返回数量

    # 上下文压缩配置 (V3)
    COMPRESSION_THRESHOLD: float = float(os.getenv("COMPRESSION_THRESHOLD", "0.5"))  # EmbeddingsFilter相似度阈值

    # RAG模式配置 (V3/V4)
    RAG_MODE: str = os.getenv("RAG_MODE", "simple")  # simple | crag | agent

    # HuggingFace镜像源（国内加速）
    HF_ENDPOINT: str = os.getenv("HF_ENDPOINT", "https://huggingface.co")

    # CORS配置
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


settings = Settings()
