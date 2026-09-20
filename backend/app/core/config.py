import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# load_dotenv 把 .env 注入 os.environ（向后兼容：某些库直接读 os.getenv）。
# BaseSettings 的 env_file 也读 .env，双重保险，CWD 在 backend/ 下即可命中。
load_dotenv()


# ── HuggingFace 缓存目录：必须在 huggingface_hub 被导入之前落定 ──────────
#
# 为什么写在模块体里，而不是某个「加载模型」的函数里
# --------------------------------------------------
# `huggingface_hub` 在 **import 时** 就把 HF_HOME 读成模块级常量
# （`constants.HF_HUB_CACHE`）。之后再写 `os.environ["HF_HOME"]` 是**无效的** ——
# 不报错，只是模型照样下到 `C:\Users\<user>\.cache\huggingface`。
# 本文件是 app 里最早被导入的模块之一，且早于任何 langchain / transformers 导入，
# 因此这里是唯一能保证「早于 huggingface_hub」的位置。
#
# 为什么默认放 D 盘
# ----------------
# `C:\Users\<user>\.cache` 是清理软件（360 / 电脑管家 / Windows 存储感知）的
# 常见清理目标。本项目实测被整体清空过一次：约 8GB 模型缓存消失
# （bge-m3 6.5GB + bge-reranker-v2-m3 2.2GB + bert-base-chinese 0.4GB），
# 表现为启动时静默重新下载、或离线环境下直接失败。
# 放到非系统盘、且目录名不含 `.cache`，可以避开这类一键清理。
#
# 外部已设置（系统环境变量）时不覆盖，便于临时切回默认位置排查。
_DEFAULT_HF_HOME = "D:/hf_cache"
HF_HOME = os.getenv("HF_HOME") or _DEFAULT_HF_HOME
os.environ["HF_HOME"] = HF_HOME
_HF_HOME_ERROR = ""
try:
    Path(HF_HOME).mkdir(parents=True, exist_ok=True)
except OSError as e:
    # 不阻断导入：config 被所有模块依赖（含测试会话），在这里抛会让整个 app 起不来。
    # 但也**不能静默** —— 目录不可写意味着模型永远下不下来，而报错会晚到
    # 「下载时」才出现，且现场是 huggingface_hub 的内部异常，很难联想到路径问题。
    # 因此记下来，交给启动期的告警通道（见 warn_on_insecure_secrets）。
    _HF_HOME_ERROR = f"{type(e).__name__}: {e}"

# HF_ENDPOINT（国内镜像）不需要在这里额外处理：
# 上面的 `load_dotenv()` 已经把 .env 里的 HF_ENDPOINT 注入 os.environ，
# 而它同样早于 huggingface_hub 的导入。
# 原先 `knowledge/embeddings.py` 与 `retrieval/reranked.py` 里各有一份
# 「若含 mirror 则写 os.environ」的代码，实为重复（load_dotenv 已覆盖），已删除。


# 目录基准。用 parents[N] 而非链式 .parent，深度一目了然。
# 本文件位于 backend/app/core/config.py：
#   parents[0] = backend/app/core   parents[1] = backend/app
#   parents[2] = backend            parents[3] = 项目根（LaborRAG/）
#
# ⚠️ 目录重构若改变了本文件深度，这两个索引必须同步改。算错不会报错，
# 只会让 DATA_DIR / RUNTIME_DATA_DIR 指向错误位置，表现为"找不到数据"。
_THIS_FILE = Path(__file__).resolve()
BACKEND_DIR = _THIS_FILE.parents[2]
BASE_DIR = _THIS_FILE.parents[3]

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
    VERSION: str = "5.0.0"
    API_PREFIX: str = "/api/v1"

    # 访问鉴权：留空 = 不启用（本地开发零配置）。
    # 填值后除 /health 外的所有 API 端点都要求 Authorization: Bearer <AUTH_SECRET>。
    AUTH_SECRET: str = ""

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

    # HuggingFace 缓存目录。
    # ⚠️ **真正生效的是模块级常量 `config.HF_HOME`**（见本文件顶部）：
    # huggingface_hub 在 import 时就固化了该路径，设晚了无效。
    # 这里保留同名字段，是为了能在状态接口里展示、并让 .env 的值参与校验。
    HF_HOME: str = ""

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
        # HF_HOME：这里赋值的是**模块级常量**（文件顶部那个），不是本字段本身。
        # 让字段与真正生效的路径保持一致，便于状态接口展示 / 排查。
        if not self.HF_HOME:
            self.HF_HOME = HF_HOME
        return self

    def warn_on_insecure_secrets(self) -> list[str]:
        """检查密钥是否缺失/占位符/疑似弱口令，返回告警列表。

        为什么只告警不崩：建知识库、跑单测等场景不需要 LLM key，
        直接崩会误伤；改为启动期收集告警交由 main.py 提示。
        """
        warnings: list[str] = []
        placeholders = {"", "your_api_key_here", "sk-xxx", "your_api_key"}
        if _HF_HOME_ERROR:
            warnings.append(
                f"HF_HOME 目录不可用（{HF_HOME}）：{_HF_HOME_ERROR}。"
                "模型将无法下载/加载。请确认该盘符存在并可写，"
                "或把 .env 里的 HF_HOME 改到别的目录。"
            )
        if self.LLM_API_KEY in placeholders:
            warnings.append(
                "LLM_API_KEY 为空或占位符——聊天/评估将无法调用 LLM。"
                "请复制 backend/.env.example 为 backend/.env 并填入真实 key。"
            )
        if self.LLM_API_KEY and len(self.LLM_API_KEY) < 16:
            warnings.append(
                f"LLM_API_KEY 长度仅 {len(self.LLM_API_KEY)}，疑似非真实密钥，请确认。"
            )
        if self.AUTH_SECRET and len(self.AUTH_SECRET) < 16:
            warnings.append(
                f"AUTH_SECRET 长度仅 {len(self.AUTH_SECRET)}，密钥过弱易被暴力猜测，"
                "建议改用 32 位以上随机串（如 `python -c \"import secrets;print(secrets.token_urlsafe(32))\"`）。"
            )
        return warnings


settings = Settings()
