"""LLM 运维性故障的分类与 HTTP 契约测试。

背景（实测踩到）：百炼免费额度耗尽时，`/chat` 返回 **500「内部服务器错误」**。
调用方看到的是一条既无法定位也无法行动的错误 —— 而「额度用完 / key 失效 /
网络不通」恰恰是线上最常见的一类故障。

修复后：这类故障归类为 `LLMUnavailableError` → 503 + 可操作提示。
本文件守住三件事：分类判定的准确性（含防误判）、HTTP 契约、以及
「LLM 挂掉时不再白跑一次注定失败的降级」。
"""
import pytest

from app.core.errors import (
    LLMUnavailableError,
    is_llm_unavailable,
    llm_unavailable_message,
)


# ── 故障分类 ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name",
    [
        "PermissionDeniedError",   # 403 额度耗尽
        "AuthenticationError",     # 401 key 无效
        "RateLimitError",          # 429
        "APIConnectionError",      # 网络不通
        "APITimeoutError",
        "InternalServerError",     # 上游 5xx
    ],
)
def test_按类名识别_LLM_运维性故障(name):
    """按类名判断（不 isinstance）—— openai 是可选依赖，本模块要保持零依赖。"""
    exc = type(name, (Exception,), {})("boom")
    assert is_llm_unavailable(exc)


def test_沿异常链识别():
    """真实故障常被 langchain 包好几层，只看最外层会漏判。"""

    class PermissionDeniedError(Exception):
        pass

    try:
        try:
            raise PermissionDeniedError("Free quota exhausted")
        except Exception as inner:
            raise RuntimeError("llm chain failed") from inner
    except Exception as outer:
        assert is_llm_unavailable(outer)


def test_沿_context_链也能识别():
    """未显式 `raise ... from` 时，异常挂在 __context__ 上。"""

    class APIConnectionError(Exception):
        pass

    try:
        try:
            raise APIConnectionError("refused")
        except Exception:
            raise RuntimeError("wrapped")
    except Exception as outer:
        assert is_llm_unavailable(outer)


@pytest.mark.parametrize(
    "exc",
    [ValueError("真 bug"), KeyError("k"), RuntimeError("编排失败"), TypeError("x")],
)
def test_普通异常不得被误判(exc):
    """防误判：把代码缺陷伪装成 503 会掩盖真问题。"""
    assert not is_llm_unavailable(exc)


def test_自引用异常链不死循环():
    """防御性：异常链理论上可能成环，遍历必须能终止。"""
    exc = RuntimeError("loop")
    exc.__cause__ = exc
    assert not is_llm_unavailable(exc)


def test_提示信息可操作且不含上游长文本():
    msg = llm_unavailable_message(ValueError("x" * 500))
    assert "LLM" in msg and "LLM_API_KEY" in msg
    assert len(msg) < 200, "不应把上游长文本塞进用户可见的提示里"


# ── HTTP 契约 ─────────────────────────────────────────────────


def test_LLM不可用返回_503_而非_500(client, fake_engine):
    """回归：额度耗尽时曾返回 500「内部服务器错误」。"""
    fake_engine.chat_error = LLMUnavailableError("LLM 服务暂不可用（额度耗尽）：PermissionDeniedError")

    r = client.post("/api/v1/chat", json={"question": "没签合同被辞了"})

    assert r.status_code == 503
    body = r.json()
    assert "LLM" in body["detail"]
    assert body["request_id"], "错误响应必须带 request_id，便于按追踪 ID 定位"


def test_503_响应也带_X_Request_ID_头(client, fake_engine):
    """异常路径绕过了中间件设置响应头的那行，需在处理器里显式回传。"""
    fake_engine.chat_error = LLMUnavailableError("LLM 不可用")

    r = client.post("/api/v1/chat", json={"question": "x"})

    assert r.status_code == 503
    assert r.headers.get("X-Request-ID") == r.json()["request_id"]


def test_普通异常仍返回_500(client, fake_engine):
    """防误判：真缺陷必须照常暴露为 500，不能被 503 掩盖。"""
    fake_engine.chat_error = ValueError("真 bug")

    r = client.post("/api/v1/chat", json={"question": "x"})

    assert r.status_code == 500
    assert r.json()["detail"] == "内部服务器错误"
