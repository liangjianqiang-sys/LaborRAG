"""API 层共享状态：引擎单例引用 + 通用辅助。

engine 的注入约定（重要）
------------------------
路由函数必须通过**模块属性访问** `deps.engine` 来读引擎，**不能**用
`from app.api.deps import engine` 然后在本地读那个名字 —— 后者是 import 时的
快照，`main.py` 的 lifespan 或测试的 monkeypatch 在运行时替换引擎时，
本地名字不会跟着变（测试里 `monkeypatch.setattr(deps, "engine", ...)` 就失效了）。

本模块位于 api 层内部，被 chat / knowledge / evaluation 三个路由子模块共享。
"""
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    # 仅供类型检查。RAGEngine 只在注解里出现（无 isinstance 等运行时用法），
    # 模块级导入会把 rag_engine → vectorstore → bm25(jieba 词典) 整条链拉进
    # app.api 的导入开销（实测约 11s）。
    from app.services.rag_engine import RAGEngine

# 支持的文档扩展名
SUPPORTED_EXT = {".pdf", ".docx", ".txt", ".md"}

# RAG 引擎实例（由 main.py 的 lifespan 注入）。注解写成字符串，避免运行时求值 RAGEngine。
engine: "RAGEngine | None" = None


def set_engine(rag_engine: "RAGEngine") -> None:
    """注入 RAG 引擎实例。"""
    global engine
    engine = rag_engine


def require_engine() -> None:
    """校验引擎就绪，否则抛 503。"""
    if not engine or not engine.is_ready:
        raise HTTPException(status_code=503, detail="知识库未就绪")
