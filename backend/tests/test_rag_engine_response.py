"""RAGEngine.chat() 的响应装配契约测试。

为什么需要它：conftest 在导入期把 `app.services.rag_engine` 整体换成替身
（为了避免加载向量库与 jieba 词典，把测试会话从分钟级降到秒级），代价是
`chat()` 从未被真正执行过。于是「取错 dict 键」这类**不抛异常、只把字段填错**
的问题可以一直存活 —— `rewritten_question` 曾长期返回 `intent` 值
（retrieve/calculate/compare），前端「改写后的问题」面板因此显示的是意图而非改写结果。

这里通过 `real_rag_engine_module` fixture 取回真实模块。本文件只测字段装配：
不碰检索、不碰生成、不加载任何模型（构造时跳过 __init__）。
"""
import asyncio

import pytest

from app.core.errors import LLMUnavailableError
from app.schemas.chat import ChatRequest


class _FakeGraph:
    """替掉 AgenticRAGGraph：run() 的返回值就是 chat() 要装配的原始 dict。"""

    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    async def run(self, question, conversation_context="", skip_guardrails=False):
        if self._error:
            raise self._error
        return self._payload


class _FakeConversationManager:
    """替掉真实的 JSON 持久化，避免测试往磁盘写会话文件。"""

    def __init__(self):
        self.messages = []

    def add_message(self, conversation_id, role, content):
        self.messages.append((conversation_id, role, content))

    def build_context_string(self, conversation_id):
        return ""


@pytest.fixture
def engine(real_rag_engine_module, monkeypatch):
    """构造不加载向量库的真实 RAGEngine：跳过 __init__，只装配 chat() 用到的属性。"""
    module = real_rag_engine_module
    monkeypatch.setattr(module, "conversation_manager", _FakeConversationManager())
    instance = module.RAGEngine.__new__(module.RAGEngine)  # 不走 __init__
    instance._fallback_llm = None
    return instance


def _payload(**overrides) -> dict:
    payload = {"answer": "答", "context_docs": [], "steps": [], "intent": "retrieve"}
    payload.update(overrides)
    return payload


def test_rewritten_question_取改写结果而非意图(engine):
    """回归：曾写成 result.get("intent")，前端显示的是 retrieve 而非改写结果。"""
    engine.agent_graph = _FakeGraph(
        _payload(rewritten_question="劳动合同法第四十七条 经济补偿金 计算")
    )

    res = asyncio.run(engine.chat(ChatRequest(question="被辞退有补偿吗")))

    assert res.rewritten_question == "劳动合同法第四十七条 经济补偿金 计算"
    assert res.rewritten_question != "retrieve"


@pytest.mark.parametrize("intent", ["retrieve", "calculate", "compare"])
def test_三种意图值都不会漏进_rewritten_question(engine, intent):
    engine.agent_graph = _FakeGraph(_payload(intent=intent, rewritten_question="改写结果"))
    res = asyncio.run(engine.chat(ChatRequest(question="问题")))
    assert res.rewritten_question == "改写结果"


def test_图未返回改写问题时为空串而非意图(engine):
    """缺失键必须降级为空串，不能顺手回落到 intent。"""
    engine.agent_graph = _FakeGraph(_payload(intent="retrieve"))
    res = asyncio.run(engine.chat(ChatRequest(question="问题")))
    assert res.rewritten_question == ""


def test_降级路径的字段契约(engine, monkeypatch):
    """图抛异常时走 _fallback_simple；此处只验证 chat() 的字段装配。"""
    engine.agent_graph = _FakeGraph(error=RuntimeError("boom"))
    monkeypatch.setattr(
        type(engine),
        "_fallback_simple",
        lambda self, q, msg, use_reranker=False: ("降级答案", [], [msg], ""),
    )

    res = asyncio.run(engine.chat(ChatRequest(question="问题")))

    assert res.answer == "降级答案"
    assert res.rewritten_question == ""
    assert res.disclaimer is True


# ── LLM 不可用时的降级行为 ────────────────────────────────────
#
# 背景：降级路径 `_fallback_simple` **同样要调 LLM**。所以当失败原因就是 LLM
# 不可用时（额度耗尽 / key 失效 / 网络不通），降级必然也失败 ——
# 原实现会白等一次注定失败的调用（还带超时），最终仍冒泡成 500。


def test_LLM不可用时跳过降级重试(engine, monkeypatch):
    """LLM 挂掉时不应再调 `_fallback_simple`，而是直接归类为 LLMUnavailableError。"""
    fallback_calls = []
    monkeypatch.setattr(
        type(engine),
        "_fallback_simple",
        lambda self, q, msg, use_reranker=False: fallback_calls.append(msg) or ("", [], [], ""),
    )

    class PermissionDeniedError(Exception):
        """模拟 openai 的 403（免费额度耗尽）。"""

    engine.agent_graph = _FakeGraph(error=PermissionDeniedError("Free quota exhausted"))

    with pytest.raises(LLMUnavailableError) as ei:
        asyncio.run(engine.chat(ChatRequest(question="问题")))

    assert fallback_calls == [], "LLM 不可用时不应再调用降级路径（必然失败且要等超时）"
    assert "LLM" in str(ei.value)


def test_降级路径因LLM不可用失败也归类上报(engine, monkeypatch):
    """agent 因非 LLM 原因失败 → 走降级；降级又因 LLM 失败 → 仍归类为 503。"""

    class APIConnectionError(Exception):
        """模拟 openai 的网络错误。"""

    def boom(self, q, msg, use_reranker=False):
        raise APIConnectionError("connection refused")

    monkeypatch.setattr(type(engine), "_fallback_simple", boom)
    engine.agent_graph = _FakeGraph(error=RuntimeError("编排失败"))

    with pytest.raises(LLMUnavailableError):
        asyncio.run(engine.chat(ChatRequest(question="问题")))


def test_降级路径因非LLM原因失败则原样抛出(engine, monkeypatch):
    """防误判：降级因普通异常失败时应暴露真缺陷，而不是伪装成 503。"""

    def boom(self, q, msg, use_reranker=False):
        raise ValueError("真 bug")

    monkeypatch.setattr(type(engine), "_fallback_simple", boom)
    engine.agent_graph = _FakeGraph(error=RuntimeError("编排失败"))

    with pytest.raises(ValueError, match="真 bug"):
        asyncio.run(engine.chat(ChatRequest(question="问题")))
