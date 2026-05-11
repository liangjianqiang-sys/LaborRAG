from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.core.rag_engine import RAGEngine
from app.api.routes import router, set_engine

# 全局RAG引擎
rag_engine = RAGEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化RAG引擎，关闭时清理资源。"""
    print(f"🚀 {settings.PROJECT_NAME} v{settings.VERSION} starting...")
    print(f"   LLM: {settings.LLM_MODEL_NAME}")
    print(f"   Embedding: {settings.EMBEDDING_MODEL_NAME}")
    print(f"   Data dir: {settings.DATA_DIR}")

    # 初始化RAG引擎（加载已有向量库或构建新的）
    rag_engine.initialize()
    set_engine(rag_engine)

    print("✅ RAG引擎初始化完成")

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
