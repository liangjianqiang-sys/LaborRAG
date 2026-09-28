"""启动期容错：引擎初始化失败时服务**仍能起来**（降级模式），并暴露失败原因。

为什么需要
----------
改前 `lifespan` 里的 `RAGEngine()` / `initialize()` 异常会冒泡出去，uvicorn 直接
启动失败 —— 表现为「服务起不来」。而失败原因（模型缓存缺失、FAISS 索引损坏）
**在运行时是可恢复的**：重新下载模型、或调 `POST /knowledge-base/build` 重建知识库。
起不来则连 `/health` 都访问不到，用户只看到一段栈回溯，没有任何恢复入口。

本项目**本来就设计好了「未就绪」状态**：`engine.is_ready`、`require_engine()` 的 503、
前端的「构建知识库」按钮。但启动期异常让这个状态永远无法到达 ——
本文件守住「能到达」这条不变量。

顺带守住一个反向不变量：**降级不能是静默的**。
如果只暴露 `knowledge_base_ready: false`，调用方分不清
「知识库为空，去构建一下」和「模型加载失败，服务实际不可用」——
这两种情况的处理完全不同，所以 `startup_error` 必须带出来。
"""
import contextlib

import pytest
from fastapi.testclient import TestClient


# ── 替身 ────────────────────────────────────────────────────────


class _FakeEngine:
    """最小引擎替身：不加载任何模型。"""

    def __init__(self, ready: bool = True, init_result: bool = True):
        self.is_ready = ready
        self._init_result = init_result
        self.initialize_called = False

    def initialize(self) -> bool:
        self.initialize_called = True
        return self._init_result


class _StoreStub:
    """`VectorStoreManager` 替身：只提供 initialize() 会碰到的东西。"""

    def load(self) -> bool:
        return True

    @property
    def is_ready(self) -> bool:
        return True


class _Bm25Stub:
    def load_index(self) -> bool:
        return True

    def is_ready(self) -> bool:
        return True


def _raise(exc: Exception):
    """构造一个「调用即抛」的替身（lambda 里不能写 raise 语句）。"""
    def _fn(*_a, **_kw):
        raise exc
    return _fn


# ── 启动 helper ─────────────────────────────────────────────────


@contextlib.contextmanager
def _started_app(monkeypatch, engine_factory=None):
    """进入 TestClient 上下文（**会触发 lifespan**），并隔离副作用。

    必须隔离的两处：
    - `_preload_eval_models()` —— 会真的向 LLM 发一次 warmup 请求
    - `load_all_tasks()` —— 会把 running 状态的任务**写回真实评估数据目录**

    ⚠️ 替换引擎工厂时必须 patch `sys.modules` 里的那个模块对象，**不能**用
    `from app.services import rag_engine`：

        conftest.py 把替身装进 sys.modules["app.services.rag_engine"]
        → main.py 的 `from app.services.rag_engine import RAGEngine` 读到**替身**
        → 而 `from app.services import rag_engine` 读到的是**包属性**（真实模块）

    两者不是同一个对象，patch 后者对 lifespan 毫无影响 —— 表现为
    「单独跑能过、全量跑变红」（因为包属性是否已存在取决于导入顺序）。
    """
    import sys

    import app.main as main_module
    from app.api import deps
    from app.evaluation import persistent as P

    monkeypatch.setattr(main_module, "_preload_eval_models", lambda: None)
    monkeypatch.setattr(P, "load_all_tasks", lambda: [])
    monkeypatch.setattr(main_module, "rag_engine", None)
    monkeypatch.setattr(main_module, "_startup_error", None)
    monkeypatch.setattr(deps, "engine", None)

    if engine_factory is not None:
        # 与 main.py 的 `from app.services.rag_engine import RAGEngine` 读同一个对象
        re_mod = sys.modules["app.services.rag_engine"]
        monkeypatch.setattr(re_mod, "RAGEngine", engine_factory)

    with TestClient(main_module.app, raise_server_exceptions=False) as c:
        yield c, main_module


# ── 启动失败 → 降级启动 ─────────────────────────────────────────


def test_引擎构造失败时服务仍能启动(monkeypatch):
    """核心不变量：构造抛异常 ≠ 服务起不来。"""
    with _started_app(monkeypatch, _raise(RuntimeError("模拟模型缓存缺失"))) as (c, _m):
        r = c.get("/health")
        assert r.status_code == 200, "服务必须仍然能响应 /health"
        assert r.json()["knowledge_base_ready"] is False


def test_启动失败原因被暴露出来_不是静默降级(monkeypatch):
    with _started_app(monkeypatch, _raise(RuntimeError("模拟模型缓存缺失"))) as (c, _m):
        err = c.get("/health").json()["startup_error"]
    assert err, "降级必须带出原因，否则就是静默降级"
    assert "RuntimeError" in err and "模拟模型缓存缺失" in err, err


def test_启动失败时业务端点返回_503(monkeypatch):
    with _started_app(monkeypatch, _raise(RuntimeError("boom"))) as (c, _m):
        r = c.post("/api/v1/chat", json={"question": "试用期被辞退有补偿吗"})
    assert r.status_code == 503, r.text


def test_存活状态不因降级而变成失败(monkeypatch):
    """liveness 与 readiness 必须分开。

    若降级时把 `status` 改成非 "ok"，负载均衡会直接摘掉实例 ——
    连查看 `startup_error` 的机会都没有，等于把「可恢复的降级」
    变成了「静默下线」。
    """
    with _started_app(monkeypatch, _raise(RuntimeError("boom"))) as (c, _m):
        body = c.get("/health").json()
    assert body["status"] == "ok"
    assert body["knowledge_base_ready"] is False


# ── 正常与半正常启动 ────────────────────────────────────────────


def test_正常启动时_startup_error_为_None(monkeypatch):
    fake = _FakeEngine(ready=True, init_result=True)
    with _started_app(monkeypatch, lambda: fake) as (c, _m):
        body = c.get("/health").json()
    assert fake.initialize_called, "lifespan 必须调用 initialize()"
    assert body["startup_error"] is None
    assert body["knowledge_base_ready"] is True


def test_知识库为空时引擎仍被注入_不是_None(monkeypatch):
    """`initialize()` 返回 False = 知识库为空，**不是**不可用。

    这种情况下引擎对象是好的、只是 `is_ready=False`，
    应当提示「先构建知识库」而不是让服务降级成 engine=None ——
    否则用户连 `/knowledge-base/build` 都调不了（它也要引擎）。
    """
    fake = _FakeEngine(ready=False, init_result=False)
    with _started_app(monkeypatch, lambda: fake) as (c, main_module):
        body = c.get("/health").json()
        assert main_module.rag_engine is fake, "引擎不该被置为 None"
    assert body["knowledge_base_ready"] is False
    assert "知识库为空" in (body["startup_error"] or "")


# ── initialize() 内部：可选增强失败不致命 ───────────────────────


def test_可选增强失败不影响引擎就绪(real_rag_engine_module, monkeypatch):
    """`_preload_reranker` / 评估检索器都是**可选增强**。

    失败只影响首次调用的延迟或评估精度，不该让一个知识库完好的引擎启动失败。

    两点说明：
    - 必须用 `real_rag_engine_module` —— `conftest.py` 在导入期把
      `app.services.rag_engine` 换成了替身，直接 import 拿到的是替身（没有这些方法）
    - 用 `__new__` 绕过 `__init__`（否则会加载 BGE-M3，约 60s）
    """
    RAGEngine = real_rag_engine_module.RAGEngine

    engine = RAGEngine.__new__(RAGEngine)
    engine.vector_store_manager = _StoreStub()
    engine.bm25_retriever = _Bm25Stub()

    # 两个可选步骤**都**失败 —— 仍然要求 initialize() 返回 True
    monkeypatch.setattr(
        RAGEngine, "_preload_reranker",
        _raise(RuntimeError("模拟 reranker 模型缺失")),
    )
    monkeypatch.setattr(
        RAGEngine, "eval_retriever",
        property(_raise(RuntimeError("模拟评估检索器构造失败"))),
    )

    assert engine.initialize() is True, "可选增强失败不该让 initialize 返回 False"


def test_向量库加载失败时_initialize_抛异常_由_lifespan_兜住(real_rag_engine_module, monkeypatch):
    """职责划分：`initialize()` **不吞**必需项的异常，由 lifespan 统一兜。

    这样分层是刻意的 —— 引擎层如实报告失败，应用层决定「降级启动还是退出」。
    若 initialize 自己吞掉，调用方就永远不知道出了什么事。
    """
    RAGEngine = real_rag_engine_module.RAGEngine

    engine = RAGEngine.__new__(RAGEngine)
    engine.vector_store_manager = _StoreStub()
    engine.bm25_retriever = _Bm25Stub()
    monkeypatch.setattr(_StoreStub, "load", _raise(OSError("模拟 FAISS 索引损坏")))

    with pytest.raises(OSError, match="FAISS 索引损坏"):
        engine.initialize()
