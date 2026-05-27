from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.core.rag_engine import RAGEngine
from app.api.routes import router, set_engine

# 全局RAG引擎
rag_engine = RAGEngine()


def _preload_eval_models():
    """预加载评估相关模型，避免首次评估时等待下载。"""
    print("⏳ 预加载评估模型...")
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

        print("✅ 评估模型预加载完成")
    except Exception as e:
        print(f"⚠️ 评估模型预加载失败（不影响主流程）: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化RAG引擎，关闭时清理资源。"""
    # 过滤高频轮询日志（reload子进程中也生效）
    import logging
    class _PollingFilter(logging.Filter):
        _HIDDEN = ("/evaluation/status", "/evaluation/persistent/progress")
        def filter(self, record):
            return not any(p in record.getMessage() for p in self._HIDDEN)
    logging.getLogger("uvicorn.access").addFilter(_PollingFilter())

    print(f"🚀 {settings.PROJECT_NAME} v{settings.VERSION} starting...")
    print(f"   LLM: {settings.LLM_MODEL_NAME}")
    print(f"   Embedding: {settings.EMBEDDING_MODEL_NAME}")
    print(f"   Data dir: {settings.DATA_DIR}")

    # 初始化RAG引擎（加载已有向量库或构建新的）
    rag_engine.initialize()
    set_engine(rag_engine)

    print("✅ RAG引擎初始化完成")

    # 将上次未完成的评估任务标记为paused（不自动恢复，需手动续评）
    from app.evaluation.eval_persistent import _load_all_tasks, _save_task, file_lock
    with file_lock:
        for task in _load_all_tasks():
            if task.get("status") in ("running", "scoring"):
                task["status"] = "paused"
                task["error"] = "服务重启，任务已暂停"
                task["updated_at"] = datetime.now().isoformat()
                _save_task(task)
                print(f"  ⏸ 任务 {task['task_id']} 标记为paused（服务重启）")

    # 预加载评估模型（后台不阻塞启动）
    import threading
    t = threading.Thread(target=_preload_eval_models, daemon=True)
    t.start()

    yield

    print("👋 Shutting down...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
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
