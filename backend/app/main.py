from contextlib import asynccontextmanager
from datetime import datetime
import uuid
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.core.rag_engine import RAGEngine
from app.api.routes import router, set_engine

logger = get_logger(__name__)

# 全局RAG引擎
rag_engine = RAGEngine()


def _preload_eval_models():
    """预加载评估相关模型，避免首次评估时等待下载。"""
    logger.info("预加载评估模型...")
    try:
        from langchain_openai import ChatOpenAI
        from app.core.embeddings import get_embeddings

        # 1. 预初始化 RAGAS LLM（触发 ragas 内部 prompt 模板加载）
        eval_llm = ChatOpenAI(
            model=settings.EVAL_LLM_MODEL_NAME,
            openai_api_key=settings.EVAL_LLM_API_KEY,
            openai_api_base=settings.EVAL_LLM_API_BASE,
            temperature=0, max_tokens=64, request_timeout=30,
            extra_body={"enable_thinking": False},
        )
        # warmup：发送一个极短请求，确保 API 连通 + ragas 内部资源就绪
        try:
            eval_llm.invoke("hi")
        except Exception:
            pass  # 连通性验证失败不影响后续（可能网络延迟）

        # 2. 预加载 Embeddings（HuggingFace 本地模型首次需下载权重）
        emb = get_embeddings()
        try:
            emb.embed_query("warmup")
        except Exception:
            pass

        # 3. 预导入 ragas metrics（触发 ragas 内部 metric 注册）
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall  # noqa: F401

        logger.info("评估模型预加载完成")
    except Exception as e:
        logger.warning(f"评估模型预加载失败（不影响主流程）: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化RAG引擎，关闭时清理资源。"""
    # 统一日志配置（reload 子进程在此 setup；setup_logging 幂等）
    setup_logging()

    logger.info(f"🚀 {settings.PROJECT_NAME} v{settings.VERSION} starting...")
    logger.info(f"   LLM: {settings.LLM_MODEL_NAME}")
    logger.info(f"   Embedding: {settings.EMBEDDING_MODEL_NAME}")
    logger.info(f"   Data dir: {settings.DATA_DIR}")

    # 密钥安全告警：缺 key / 占位符 / 弱口令启动即提示，避免带空 key 跑生产
    for w in settings.warn_on_insecure_secrets():
        logger.warning(f"[密钥安全] {w}")

    # 初始化RAG引擎（加载已有向量库或构建新的）
    rag_engine.initialize()
    set_engine(rag_engine)

    logger.info("RAG引擎初始化完成")

    # 将上次未完成的评估任务标记为paused（不自动恢复，需手动续评）
    from app.evaluation.eval_persistent import load_all_tasks, save_task, file_lock
    with file_lock:
        for task in load_all_tasks():
            if task.get("status") in ("running", "scoring"):
                task["status"] = "paused"
                task["error"] = "服务重启，任务已暂停"
                task["updated_at"] = datetime.now().isoformat()
                save_task(task)
                logger.info(f"  ⏸ 任务 {task['task_id']} 标记为paused（服务重启）")

    # 预加载评估模型（同步阻塞，启动后不等待）
    _preload_eval_models()

    logger.info("所有模型预加载完成")
    yield

    logger.info("👋 Shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)


# ── request_id 中间件：每请求注入 uuid，响应头回传，便于线上串联排障 ──
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── 全局异常处理：统一错误格式 + 完整堆栈记日志（替代 routes 里散落的 catch-all-500）──
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(f"[{request_id}] 未处理异常 {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误", "request_id": request_id},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=422,
        content={"detail": "请求参数校验失败", "request_id": request_id, "errors": exc.errors()},
    )


# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix=settings.API_PREFIX)
