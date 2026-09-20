"""API 层测试的公共 fixture。

关于导入成本：
`app/main.py` 在模块级执行 `rag_engine = RAGEngine()`，而 `app/core/rag_engine.py`
的导入链会拉起 langchain_core.prompts / langchain_text_splitters /
langchain_community.document_loaders / jieba 词典，实测约 11s 的**纯导入**开销。
API 契约测试用不到真实引擎，所以在 import app.main 之前把 `app.services.rag_engine`
替换成轻量替身，让整个测试会话保持在秒级。

需要验证真实引擎行为的用例，用 `real_rag_engine_module` fixture 取回真实模块
（它会临时摘掉替身，用完还原）。
"""
import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.schemas.chat import ChatResponse, SourceDocument


class _StubRAGEngine:
    """占位引擎：真实引擎由 client fixture 通过 monkeypatch 注入。"""

    def __init__(self, *args, **kwargs):
        self.is_ready = False
        self.vector_store_manager = None

    def initialize(self) -> bool:
        return False

    def build_knowledge_base(self, rebuild: bool = False) -> bool:
        return False


def _install_engine_stub() -> None:
    """把 app.services.rag_engine 换成不含重依赖的替身模块。"""
    if "app.services.rag_engine" in sys.modules:
        return
    stub = types.ModuleType("app.services.rag_engine")
    stub.RAGEngine = _StubRAGEngine
    sys.modules["app.services.rag_engine"] = stub


_install_engine_stub()


class FakeEngine:
    """替身引擎：不加载模型、不读写磁盘，行为可通过属性注入。"""

    def __init__(self):
        self.is_ready = True
        self.vector_store_manager = MagicMock(is_ready=False, vector_store=None)
        self.chat_calls = []
        self.chat_error = None
        self.upload_chunk_count = 3

    def initialize(self) -> bool:
        return True

    def build_knowledge_base(self, rebuild: bool = False) -> bool:
        return True

    def add_document(self, filepath: str) -> int:
        return self.upload_chunk_count

    async def chat(self, request, use_reranker: bool = False) -> ChatResponse:
        self.chat_calls.append(request)
        if self.chat_error is not None:
            raise self.chat_error
        return ChatResponse(
            answer=f"关于「{request.question}」的回答",
            sources=[
                SourceDocument(
                    content="用人单位安排加班应支付不低于150%的工资报酬…",
                    source="劳动法.txt",
                    score=0.91,
                )
            ],
            conversation_id=request.conversation_id or "conv-fake",
            rag_steps=["rewrite_query", "route_intent", "validate"],
            rewritten_question=request.question,
            confidence=0.88,
        )


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(fake_engine, monkeypatch):
    """TestClient，注入替身引擎。

    刻意不使用 `with TestClient(...)`：那会触发 lifespan，进而预加载评估模型、
    扫描评估任务文件 —— 这些与 HTTP 契约测试无关。
    """
    import app.main as main_module
    from app.api import deps

    monkeypatch.setattr(main_module, "rag_engine", fake_engine)
    monkeypatch.setattr(deps, "engine", fake_engine)
    return TestClient(main_module.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _auth_off_by_default(monkeypatch):
    """默认关闭鉴权，避免开发机 .env 里的 AUTH_SECRET 污染用例。"""
    monkeypatch.setattr(settings, "AUTH_SECRET", "")


@pytest.fixture
def auth_secret(monkeypatch):
    """开启鉴权并返回密钥。"""
    secret = "test-secret-0123456789abcdef"
    monkeypatch.setattr(settings, "AUTH_SECRET", secret)
    return secret


@pytest.fixture
def auth_headers(auth_secret):
    return {"Authorization": f"Bearer {auth_secret}"}


_real_engine_module = None


@pytest.fixture
def real_rag_engine_module():
    """取回真实的 `app.services.rag_engine` 模块（临时摘掉导入期装的替身）。

    替身是在本文件导入时就装进 sys.modules 的，之后无法通过 monkeypatch 撤销，
    所以这里直接摘掉它、加载真实模块，用完立刻把替身放回去，避免影响其他用例。

    真实模块只在首次调用时加载一次（约 11s，主要来自 jieba 词典与 langchain_core），
    之后走缓存 —— 否则每个用例都要重新执行一遍模块体。
    """
    global _real_engine_module
    if _real_engine_module is None:
        stub = sys.modules.pop("app.services.rag_engine", None)
        try:
            _real_engine_module = importlib.import_module("app.services.rag_engine")
        finally:
            if stub is not None:
                sys.modules["app.services.rag_engine"] = stub
    return _real_engine_module
