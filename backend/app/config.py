import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# load_dotenv 把 .env 注入 os.environ（向后兼容：某些库直接读 os.getenv）。
# BaseSettings 的 env_file 也读 .env，双重保险，CWD 在 backend/ 下即可命中。
load_dotenv()

# 项目根目录（backend/的父目录）
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_BASE_DIR = BASE_DIR / "data"

# CORS 缺省白名单（env 未设 CORS_ORIGINS 时用）
_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]


class Settings(BaseSettings):
    """应用配置。环境变量驱动，类型自动校验/转换。

    迁移自手写 os.getenv 类：BaseSettings 在实例化时按注解转 int/float/bool，
    配错（如非数字填进 int 字段）启动即崩，而非运行到一半才暴露。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PROJECT_NAME: str = "LaborRAG - 劳动法智能问答系统"
    VERSION: str = "4.0.0"
    API_PREFIX: str = "/api/v1"

    # LLM配置（生成用）
    LLM_API_KEY: str = ""
    LLM_API_BASE: str = "https://api.deepseek.com/v1"
    LLM_MODEL_NAME: str = "deepseek-chat"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048

    # LLM配置（评估用，缺省回退到生成用，见 _apply_fallbacks）
    EVAL_LLM_API_KEY: str = ""
    EVAL_LLM_API_BASE: str = ""
    EVAL_LLM_MODEL_NAME: str = ""

    # RAGAS评估专用（缺省回退到 EVAL，再回退到 LLM）
    RAGAS_API_KEY: str = ""
    RAGAS_API_BASE: str = ""
    RAGAS_MODEL_NAME: str = ""

    # Embedding配置（默认本地 BGE-M3；填 API_KEY+API_BASE 则走 OpenAI 兼容 API）
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_API_BASE: str = ""

    # 运行时数据目录与派生路径
    RUNTIME_DATA_DIR: str = str(BASE_DIR / "runtime_data")
    VECTOR_STORE_PATH: str = ""        # 留空则由 _apply_fallbacks 派生
    DATA_DIR: str = str(DATA_BASE_DIR / "labor_laws")
    CONVERSATION_DATA_PATH: str = ""   # 留空则派生
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    CHUNK_STRATEGY: str = "law_article"  # recursive | law_article
    PARENT_CHILD_ENABLED: bool = True

    # 检索配置
    RETRIEVER_TYPE: str = "vector"  # vector | hybrid | reranked
    TOP_K: int = 12
    SCORE_THRESHOLD: float = 0.15

    # 混合检索配置
    VECTOR_WEIGHT: float = 0.5
    BM25_WEIGHT: float = 0.5
    RRF_K: int = 60

    # 重排序配置
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    RERANK_TOP_K: int = 12
    RERANK_SCORE_THRESHOLD: float = 0.3  # CrossEncoder原始分数阈值(>0相关,<0不相关)

    # 检索验证配置（两步生成开关）
    RETRIEVAL_VALIDATION: bool = True

    # 评估配置
    EVAL_MAX_CONCURRENT: int = 4
    EVAL_TASK_TIMEOUT: int = 1800

    # HuggingFace镜像源（国内加速）
    HF_ENDPOINT: str = "https://huggingface.co"

    # CORS配置：env 可传逗号分隔或 JSON 数组；缺省走 localhost 白名单
    CORS_ORIGINS: list[str] = list(_DEFAULT_CORS_ORIGINS)

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        """支持 env 用逗号分隔或 JSON 数组；缺省回退 localhost 列表。"""
        if v in (None, "", []):
            return list(_DEFAULT_CORS_ORIGINS)
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json
                return json.loads(v)
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @model_validator(mode="after")
    def _apply_fallbacks(self):
        """集中处理跨字段 fallback 与派生路径，替代散落的 `or os.getenv` 链。"""
        # EVAL 回退到 LLM
        if not self.EVAL_LLM_API_KEY:
            self.EVAL_LLM_API_KEY = self.LLM_API_KEY
        if not self.EVAL_LLM_API_BASE:
            self.EVAL_LLM_API_BASE = self.LLM_API_BASE
        if not self.EVAL_LLM_MODEL_NAME:
            self.EVAL_LLM_MODEL_NAME = self.LLM_MODEL_NAME
        # RAGAS 回退到 EVAL（此时 EVAL 已回退完）
        if not self.RAGAS_API_KEY:
            self.RAGAS_API_KEY = self.EVAL_LLM_API_KEY
        if not self.RAGAS_API_BASE:
            self.RAGAS_API_BASE = self.EVAL_LLM_API_BASE
        if not self.RAGAS_MODEL_NAME:
            self.RAGAS_MODEL_NAME = self.EVAL_LLM_MODEL_NAME
        # 派生路径
        if not self.VECTOR_STORE_PATH:
            self.VECTOR_STORE_PATH = str(Path(self.RUNTIME_DATA_DIR) / "vector_store")
        if not self.CONVERSATION_DATA_PATH:
            self.CONVERSATION_DATA_PATH = str(Path(self.RUNTIME_DATA_DIR) / "conversations")
        return self

    def warn_on_insecure_secrets(self) -> list[str]:
        """检查密钥是否缺失/占位符/疑似弱口令，返回告警列表。

        为什么只告警不崩：建知识库、跑单测等场景不需要 LLM key，
        直接崩会误伤；改为启动期收集告警交由 main.py 提示。
        """
        warnings: list[str] = []
        placeholders = {"", "your_api_key_here", "sk-xxx", "your_api_key"}
        if self.LLM_API_KEY in placeholders:
            warnings.append(
                "LLM_API_KEY 为空或占位符——聊天/评估将无法调用 LLM。"
                "请复制 backend/.env.example 为 backend/.env 并填入真实 key。"
            )
        if self.LLM_API_KEY and len(self.LLM_API_KEY) < 16:
            warnings.append(
                f"LLM_API_KEY 长度仅 {len(self.LLM_API_KEY)}，疑似非真实密钥，请确认。"
            )
        return warnings


settings = Settings()
