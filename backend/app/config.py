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

    # LLM配置（RAGAS评估系统专用，可与上述不同）
    RAGAS_API_KEY: str = os.getenv("RAGAS_API_KEY", "") or os.getenv("EVAL_LLM_API_KEY", "")
    RAGAS_API_BASE: str = os.getenv("RAGAS_API_BASE", "") or os.getenv("EVAL_LLM_API_BASE", "https://api.deepseek.com/v1")
    RAGAS_MODEL_NAME: str = os.getenv("RAGAS_MODEL_NAME", "") or os.getenv("EVAL_LLM_MODEL_NAME", "deepseek-chat")

    # Embedding配置
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "BAAI/bge-m3"
    )
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_API_BASE: str = os.getenv("EMBEDDING_API_BASE", "")

    # 运行时数据目录（跨平台默认值，优先使用环境变量）
    # 注意: FAISS底层C++不支持中文路径，若项目路径含中文需通过环境变量指定纯英文路径
    _RUNTIME_DATA_DIR = os.getenv(
        "RUNTIME_DATA_DIR",
        str(BASE_DIR / "runtime_data")
    )
    VECTOR_STORE_PATH: str = os.getenv(
        "VECTOR_STORE_PATH",
        str(Path(_RUNTIME_DATA_DIR) / "vector_store")
    )
    DATA_DIR: str = os.getenv("DATA_DIR", str(DATA_BASE_DIR / "labor_laws"))
    CONVERSATION_DATA_PATH: str = os.getenv(
        "CONVERSATION_DATA_PATH",
        str(Path(_RUNTIME_DATA_DIR) / "conversations")
    )
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    CHUNK_STRATEGY: str = os.getenv("CHUNK_STRATEGY", "law_article")  # recursive | law_article

    # 层次化父子分块配置
    PARENT_CHILD_ENABLED: bool = os.getenv("PARENT_CHILD_ENABLED", "true").lower() == "true"

    # 检索配置
    RETRIEVER_TYPE: str = os.getenv("RETRIEVER_TYPE", "vector")  # vector | hybrid | reranked
    TOP_K: int = int(os.getenv("TOP_K", "12"))
    SCORE_THRESHOLD: float = float(os.getenv("SCORE_THRESHOLD", "0.15"))

    # 混合检索配置 (V2)
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.45"))  # 向量检索权重
    BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.55"))      # BM25检索权重
    RRF_K: int = int(os.getenv("RRF_K", "60"))                       # RRF常数

    # 重排序配置 (V2)
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "8"))          # 重排序后返回数量
    RERANK_SCORE_THRESHOLD: float = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.0"))  # CrossEncoder原始分数阈值(>0相关,<0不相关)

    # 上下文压缩配置 (V3)
    COMPRESSION_THRESHOLD: float = float(os.getenv("COMPRESSION_THRESHOLD", "0.35"))  # EmbeddingsFilter相似度阈值

    # RAG模式配置 (V3/V4)
    RAG_MODE: str = os.getenv("RAG_MODE", "simple")  # simple | crag | agent

    # 检索验证配置（两步生成开关）
    RETRIEVAL_VALIDATION: bool = os.getenv("RETRIEVAL_VALIDATION", "true").lower() == "true"  # 检索充分性验证，关闭则跳过直接生成

    # 评估并发配置
    EVAL_MAX_CONCURRENT: int = int(os.getenv("EVAL_MAX_CONCURRENT", "4"))  # 评估答案生成并发度
    EVAL_TASK_TIMEOUT: int = int(os.getenv("EVAL_TASK_TIMEOUT", "1800"))  # 评估任务超时秒数

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
